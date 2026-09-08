"""Incrementally download daily OHLCV bars into data/<TICKER>.csv.

Reads the ticker list from data/tickers.txt, fetches only the bars missing
since the last stored date (re-fetching a small overlap window so Yahoo's
post-close corrections overwrite stale rows), and rewrites each CSV.

Run daily by .github/workflows/update-data.yml; safe to run by hand.
Exits non-zero if any ticker failed so the workflow run shows up red,
but successful tickers are still written first.
"""
from __future__ import annotations

import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
OVERLAP_DAYS = 7
RETRIES = 3
RETRY_WAIT_SECONDS = 30


def load_existing(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = "Date"
    return frame


def drop_live_bar(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop today's in-progress bar while the US session is open — running
    mid-session must not store a provisional price as an official close."""
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    if now_ny.hour >= 16:
        return frame
    return frame[frame.index < pd.Timestamp(now_ny.date())]


def fetch(ticker: str, start: date | None) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            if start is None:
                frame = yf.download(ticker, period="max", progress=False, auto_adjust=False, repair=True, threads=False)
            else:
                frame = yf.download(ticker, start=start, progress=False, auto_adjust=False, repair=True, threads=False)
            if frame is None or frame.empty:
                raise RuntimeError(f"{ticker}: empty response")
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            frame = frame[[col for col in COLUMNS if col in frame.columns]]
            frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
            frame.index.name = "Date"
            return drop_live_bar(frame[frame["Close"].notna()])
        except Exception as exc:  # noqa: BLE001 - retry any yfinance failure
            last_error = exc
            if attempt < RETRIES:
                print(f"{ticker}: attempt {attempt} failed ({exc}); retrying in {RETRY_WAIT_SECONDS}s")
                time.sleep(RETRY_WAIT_SECONDS)
    raise RuntimeError(f"{ticker}: all {RETRIES} attempts failed") from last_error


def update_ticker(ticker: str) -> bool:
    path = DATA_DIR / f"{ticker}.csv"
    existing = load_existing(path)
    start = None
    if existing is not None and not existing.empty:
        start = (existing.index.max() - timedelta(days=OVERLAP_DAYS)).date()

    fresh = fetch(ticker, start)
    if fresh.empty:  # e.g. run mid-session when only today's live bar is new
        print(f"{ticker}: no new closed bars, last {existing.index.max().date() if existing is not None else 'n/a'}")
        return False
    if existing is not None:
        merged = pd.concat([existing[existing.index < fresh.index.min()], fresh])
    else:
        merged = fresh
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    changed = existing is None or not merged.equals(existing)
    if changed:
        path.write_text(merged.to_csv())
        print(f"{ticker}: {len(merged)} rows, last {merged.index.max().date()} (updated)")
    else:
        print(f"{ticker}: unchanged, last {merged.index.max().date()}")
    return changed


def main() -> int:
    tickers = [
        line.strip()
        for line in (DATA_DIR / "tickers.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    failures = []
    for ticker in tickers:
        try:
            update_ticker(ticker)
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            failures.append(ticker)
            print(f"{ticker}: FAILED - {exc}", file=sys.stderr)
    if failures:
        print(f"failed tickers: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
