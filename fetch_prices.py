"""Download historical price data for a given stock ticker."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import yfinance as yf

DEFAULT_OUTPUT_DIR = Path("market_data")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical OHLCV data via Yahoo Finance.")
    parser.add_argument("ticker", help="Symbol to download, e.g. SOXL")
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Pricing interval understood by yfinance (default: 1d)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path. Defaults to market_data/<ticker>_<start>_<end>_<interval>.csv",
    )
    return parser.parse_args(argv)


def fetch_price_frame(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No price data returned for {ticker} between {start} and {end}")
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.set_index(pd.to_datetime(data.index))
    if data.index.name is None:
        data.index.name = "Date"
    frame = data.reset_index()
    frame.columns = [_sanitize_column(col) for col in frame.columns]
    if "date" not in frame.columns:
        raise ValueError("Unexpected response format: missing 'Date' column")
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    required = {"open", "high", "low", "close", "adj_close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing expected columns in price data: {sorted(missing)}")
    ordered_cols = ["date"] + sorted(required)
    frame = frame[ordered_cols]
    return frame


def _sanitize_column(column: object) -> str:
    if isinstance(column, tuple):
        column = column[0]
    if column is None:
        return ""
    return str(column).lower().replace(" ", "_")


def _default_output_path(ticker: str, start: str, end: str, interval: str) -> Path:
    ticker_slug = ticker.replace("/", "-")
    filename = f"{ticker_slug}_{start}_{end}_{interval}.csv"
    return DEFAULT_OUTPUT_DIR / filename


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_path = args.output or _default_output_path(args.ticker, args.start, args.end, args.interval)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = fetch_price_frame(args.ticker, args.start, args.end, args.interval)
    frame.to_csv(output_path, index=False)
    print(f"Saved {len(frame)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
