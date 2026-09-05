from __future__ import annotations

import threading
import time
from datetime import date

import pandas as pd
import yfinance as yf


class MarketDataError(RuntimeError):
    """Raised when a requested Yahoo Finance series is unavailable."""


class MarketDataClient:
    def __init__(self, ttl_seconds: int = 600, max_entries: int = 64):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._cache: dict[tuple[str, date, date], tuple[float, pd.DataFrame]] = {}
        self._lock = threading.Lock()

    def download(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        key = (ticker, start, end)
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] < self.ttl_seconds:
                return cached[1].copy(deep=True)

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

        with self._lock:
            if len(self._cache) >= self.max_entries:
                oldest_key = min(self._cache, key=lambda item: self._cache[item][0])
                self._cache.pop(oldest_key, None)
            self._cache[key] = (now, frame.copy(deep=True))
        return frame


market_data_client = MarketDataClient()
