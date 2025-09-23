#!/usr/bin/env python3
"""
Daily price fetcher with foldered CSVs + multi-source support.

- Reads data/reference/assets.csv
  Required columns (min): asset_id,currency,price_source
  Optional but recommended: source_symbol,current_price,short_name,name,type,market,sector

- For rows with price_source == 'funddoctor':
    Uses asset_id as the FundDoctor CODE and fetches 기준가(원) for the most recent available day
    (looks back up to MAX_BACK_DAYS).

- For rows with price_source == 'yahoo':
    Uses source_symbol with yfinance to get the last close in native currency.

- Appends (date, asset_id, close) to data/prices/prices_daily.csv (no duplicates per date+asset).
- Updates assets.csv current_price with the fetched close for convenience.

- NO currency conversion. All prices remain in each asset's native trading currency.

Tip: If you ever add a new data source, extend `price_source` values and add another fetcher.
"""

from __future__ import annotations

# -----------------------
# Imports (keep on top!)
# -----------------------
import datetime as dt
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # yfinance not available in runner? We'll skip Yahoo sources.

# -----------------------
# Constants & Paths
# -----------------------
ROOT = Path(__file__).resolve().parent
ASSETS_FILE = ROOT / "data/reference/assets.csv"
PRICES_FILE = ROOT / "data/prices/prices_daily.csv"
PRICES_FILE.parent.mkdir(parents=True, exist_ok=True)

DATE_TODAY: dt.date = dt.date.today()
MAX_BACK_DAYS = 14  # look back this many days for FundDoctor (weekends/holidays)
YF_BATCH_SIZE = 10

UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
]
BASE_HEADERS = {
    "User-Agent": random.choice(UA_LIST),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.funddoctor.co.kr/",
    "Cache-Control": "no-cache",
}

# -----------------------
# Utilities
# -----------------------
def yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def iter_back_dates(start: dt.date, days: int) -> List[dt.date]:
    return [start - dt.timedelta(days=i) for i in range(days + 1)]


def load_assets() -> pd.DataFrame:
    df = pd.read_csv(ASSETS_FILE)
    required = {"asset_id", "currency", "price_source"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"assets.csv is missing columns: {missing}")
    # Ensure these columns exist even if blank
    for col in ("source_symbol", "current_price"):
        if col not in df.columns:
            df[col] = ""
    return df


def load_existing_prices() -> pd.DataFrame:
    if PRICES_FILE.exists():
        return pd.read_csv(PRICES_FILE)
    return pd.DataFrame(columns=["date", "asset_id", "close"])


def append_prices(existing: pd.DataFrame, rows: List[Tuple[str, str, float]]) -> pd.DataFrame:
    """Append rows and drop duplicates on (date, asset_id)."""
    new_df = pd.DataFrame(rows, columns=["date", "asset_id", "close"])
    out = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    out.drop_duplicates(subset=["date", "asset_id"], keep="last", inplace=True)
    return out


def update_assets_current_price(assets: pd.DataFrame, latest_by_asset: Dict[str, Tuple[dt.date, float]]) -> pd.DataFrame:
    updated = assets.copy()
    for aid, (_d, px) in latest_by_asset.items():
        updated.loc[updated["asset_id"] == aid, "current_price"] = px
    return updated


# -----------------------
# FundDoctor (KR) fetcher
# -----------------------
def funddoctor_url(code: str, on_date: dt.date) -> str:
    return f"https://www.funddoctor.co.kr/afn/fund/fprofile.jsp?fund_cd={code}&gijun_ymd={yyyymmdd(on_date)}"


def parse_funddoctor_price(html: str) -> Optional[float]:
    """
    Heuristic parser:
    1) Prefer a number near the '기준가(원)' label.
    2) Fallback: pick the largest numeric candidate in tables.
    """
    text = re.sub(r"\s+", " ", html)

    # Try to find number near '기준가' (NAV)
    for m in re.finditer(r"기준가[^(]*\(?원\)?[^0-9]{0,60}([0-9][0-9,]*\.?[0-9]*)", text):
        try:
            return float(m.group(1).replace(",", ""))
        except Exception:
            pass

    # Fallback: extract numeric-looking cells and take the largest
    candidates = [c.replace(",", "") for c in re.findall(r">([\s0-9,]+\d(?:\.\d+)?)<", html)]
    nums: List[float] = []
    for c in candidates:
        c = c.strip()
        if not c or not re.search(r"\d", c):
            continue
        try:
            nums.append(float(c))
        except Exception:
            pass
    return max(nums) if nums else None


def fetch_funddoctor(code: str, on_date: dt.date) -> Tuple[dt.date, float]:
    s = requests.Session()
    s.headers.update(BASE_HEADERS)

    for d in iter_back_dates(on_date, MAX_BACK_DAYS):
        url = funddoctor_url(code, d)
        try:
            resp = s.get(url, timeout=20)
            if resp.status_code != 200 or not resp.text:
                continue
            price = parse_funddoctor_price(resp.text)
            if price and price > 0:
                print(f"[FD] {code} {d} → {price}")
                return d, float(price)
        except requests.RequestException:
            time.sleep(0.3)
            continue
    raise RuntimeError(f"FundDoctor: could not fetch price for {code} within {MAX_BACK_DAYS} days")


# -----------------------
# Yahoo Finance fetcher
# -----------------------
def fetch_yahoo(symbol_map: Dict[str, str]) -> Dict[str, Tuple[dt.date, float]]:
    """
    Returns a map: asset_id -> (date, close)
    """
    results: Dict[str, Tuple[dt.date, float]] = {}
    if not symbol_map or yf is None:
        return results

    items = list(symbol_map.items())
    for i in range(0, len(items), YF_BATCH_SIZE):
        chunk = items[i : i + YF_BATCH_SIZE]
        ids = [aid for aid, _ in chunk]
        syms = [sym for _, sym in chunk]

        data = yf.download(
            syms,
            period="10d",
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=False,
        )

        for aid, sym in zip(ids, syms):
            try:
                # Multi-symbol returns MultiIndex columns (sym, field)
                ser = data[(sym, "Close")] if len(syms) > 1 else data["Close"]
                ser = ser.dropna()
                if ser.empty:
                    continue
                last_ts = ser.index.max()
                last_date = last_ts.date() if hasattr(last_ts, "date") else DATE_TODAY
                last_val = float(ser.loc[last_ts])
                results[aid] = (last_date, last_val)
                print(f"[YF] {aid} ({sym}) {last_date} → {last_val}")
            except Exception:
                # Fallback for odd shapes
                try:
                    ser = data["Close"].dropna()
                    if not ser.empty:
                        last_ts = ser.index.max()
                        last_date = last_ts.date() if hasattr(last_ts, "date") else DATE_TODAY
                        last_val = float(ser.loc[last_ts])
                        results[aid] = (last_date, last_val)
                        print(f"[YF*] {aid} ({sym}) {last_date} → {last_val}")
                except Exception:
                    print(f"WARN: yahoo failed for {aid} ({sym})")
    return results


# -----------------------
# Main
# -----------------------
def main() -> None:
    print("== Fetch start ==")
    assets = load_assets()
    prices = load_existing_prices()

    # Build lists by source
    yahoo_map: Dict[str, str] = {}
    fd_codes: List[str] = []
    for _, row in assets.iterrows():
        aid = str(row["asset_id"])
        src = str(row.get("price_source", "")).lower()
        if src == "yahoo":
            sym = (str(row.get("source_symbol", "")).strip() or aid)
            yahoo_map[aid] = sym
        elif src == "funddoctor":
            fd_codes.append(aid)

    latest: Dict[str, Tuple[dt.date, float]] = {}

    # FundDoctor
    for code in fd_codes:
        try:
            d, px = fetch_funddoctor(code, DATE_TODAY)
            latest[code] = (d, px)
        except Exception as e:
            print(f"WARN: {e}")

    # Yahoo
    latest.update(fetch_yahoo(yahoo_map))

    print("DEBUG latest_by_asset =", {k: (v[0].isoformat(), v[1]) for k, v in latest.items()})

    if not latest:
        print("No prices fetched. Check price_source/source_symbol settings or site availability.")
        return

    # Append/Update prices_daily
    out_rows = [(d.isoformat(), aid, px) for aid, (d, px) in latest.items()]
    prices_updated = append_prices(prices, out_rows)
    prices_updated.sort_values(["date", "asset_id"], inplace=True)
    prices_updated.to_csv(PRICES_FILE, index=False)

    # Update current_price in assets
    assets_updated = update_assets_current_price(assets, latest)
    assets_updated.to_csv(ASSETS_FILE, index=False)

    print(f"OK: appended {len(out_rows)} prices (FundDoctor: {len(fd_codes)}, Yahoo: {len(yahoo_map)})")
    print("== Fetch done ==")


if __name__ == "__main__":
    # Tiny sanity test for date formatting
    assert yyyymmdd(dt.date(2025, 9, 23)) == "20250923"
    main()
