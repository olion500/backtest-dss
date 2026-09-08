from __future__ import annotations

import os
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[4]

# Where daily bars committed by .github/workflows/update-data.yml are read from.
# May be an https URL (raw GitHub) or a local directory path.
DEFAULT_DATA_SOURCE = "https://raw.githubusercontent.com/olion500/backtest-dss/main/data"

# Bars newer than the dataset's last row are fetched live from Yahoo and
# appended (the committed dataset lags whenever the updater is not running).
# Only if that live top-up also fails AND the dataset is older than this many
# days is the dataset treated as broken, falling back to a full Yahoo fetch.
MAX_DATASET_AGE_DAYS = 7


class MarketDataError(RuntimeError):
    """Raised when a requested price series is unavailable."""


def _data_source() -> str:
    configured = os.environ.get("DONGPA_DATA_URL", "").strip()
    if configured:
        return configured.rstrip("/")
    local = REPO_ROOT / "data"
    if local.is_dir():
        return str(local)
    return DEFAULT_DATA_SOURCE


class MarketDataClient:
    def __init__(self, ttl_seconds: int = 600, max_entries: int = 64):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
        self._lock = threading.Lock()

    def download(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Return daily bars for [start, end) — dataset first, Yahoo as fallback."""
        frame = self._from_dataset(ticker, start, end)
        if frame is None:
            frame = self._from_yahoo(ticker, start, end)
        if frame is not None:
            # Applied to every path: the startup updater may have written
            # today's provisional bar into the dataset while the US session
            # is still open.
            frame = self._drop_live_bar(frame)
        if frame is None or frame.empty:
            raise MarketDataError(f"{ticker} 가격 데이터가 비어 있습니다.")
        return frame

    def _cached(self, key: tuple) -> pd.DataFrame | None:
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] < self.ttl_seconds:
                return cached[1].copy(deep=True)
        return None

    def _store(self, key: tuple, frame: pd.DataFrame) -> None:
        with self._lock:
            if len(self._cache) >= self.max_entries:
                oldest_key = min(self._cache, key=lambda item: self._cache[item][0])
                self._cache.pop(oldest_key, None)
            self._cache[key] = (time.monotonic(), frame.copy(deep=True))

    def _from_dataset(self, ticker: str, start: date, end: date) -> pd.DataFrame | None:
        source = _data_source()
        key = ("dataset", source, ticker)
        frame = self._cached(key)
        if frame is None:
            location = f"{source}/{ticker}.csv" if "://" in source else str(Path(source) / f"{ticker}.csv")
            try:
                frame = pd.read_csv(location, index_col=0, parse_dates=True)
            except Exception:
                return None  # untracked ticker or unreachable source → live fallback
            if frame.empty or "Close" not in frame.columns:
                return None
            self._store(key, frame)
        latest_needed = min(end - timedelta(days=1), date.today())
        if frame.index.max().date() < latest_needed:
            frame = self._extend_from_yahoo(ticker, frame, end)
        if frame.index.max().date() < date.today() - timedelta(days=MAX_DATASET_AGE_DAYS):
            return None
        sliced = frame[(frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end))]
        return sliced.copy(deep=True) if not sliced.empty else None

    def _extend_from_yahoo(self, ticker: str, frame: pd.DataFrame, end: date) -> pd.DataFrame:
        """Append bars newer than the dataset's last row, fetched live."""
        gap_start = frame.index.max().date() + timedelta(days=1)
        try:
            fresh = self._from_yahoo(ticker, gap_start, end)
        except MarketDataError:
            return frame
        fresh = self._flatten(fresh)
        fresh = fresh[[col for col in frame.columns if col in fresh.columns]]
        if fresh.empty:
            return frame
        merged = pd.concat([frame, fresh])
        return merged[~merged.index.duplicated(keep="last")].sort_index()

    @staticmethod
    def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
        frame.index.name = "Date"
        return frame

    @staticmethod
    def _drop_live_bar(frame: pd.DataFrame) -> pd.DataFrame:
        """Drop today's in-progress bar while the US session is still open,
        so the newest close is always an official close (matches the
        Streamlit pages' market-hours cutoff)."""
        try:
            now_ny = datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            return frame
        if now_ny.hour >= 16:
            return frame
        return frame[frame.index < pd.Timestamp(now_ny.date())]

    def _from_yahoo(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        key = ("yahoo", ticker, start, end)
        frame = self._cached(key)
        if frame is not None:
            return frame
        try:
            frame = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                repair=True,
                threads=False,
            )
        except Exception as exc:
            raise MarketDataError(f"{ticker} 가격 데이터를 불러오지 못했습니다.") from exc
        if frame is None or frame.empty:
            raise MarketDataError(f"{ticker} 가격 데이터가 비어 있습니다.")
        self._store(key, frame)
        return frame


market_data_client = MarketDataClient()
