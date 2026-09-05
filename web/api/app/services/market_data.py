from __future__ import annotations

import os
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[4]

# Where daily bars committed by .github/workflows/update-data.yml are read from.
# May be an https URL (raw GitHub) or a local directory path.
DEFAULT_DATA_SOURCE = "https://raw.githubusercontent.com/olion500/backtest-dss/main/data"

# A dataset whose newest bar is older than this is treated as broken (the
# updater stopped running) and the request falls back to live Yahoo data.
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
        if frame.index.max().date() < date.today() - timedelta(days=MAX_DATASET_AGE_DAYS):
            return None
        sliced = frame[(frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end))]
        return sliced.copy(deep=True) if not sliced.empty else None

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
