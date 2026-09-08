"""Tests for the viewer's dataset + Yahoo gap-fill data client."""
from datetime import date, timedelta

import pandas as pd
import pytest

from web.api.app.services.market_data import MarketDataClient, MarketDataError

COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _bars(dates: list[str], close_start: float = 100.0) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    rows = {
        col: [close_start + i for i in range(len(dates))]
        for col in COLUMNS
    }
    return pd.DataFrame(rows, index=index)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DONGPA_DATA_URL", str(tmp_path))
    # After-hours: keep _drop_live_bar inert so tests are time-independent.
    monkeypatch.setattr(MarketDataClient, "_drop_live_bar", staticmethod(lambda frame: frame))
    instance = MarketDataClient()
    return instance, tmp_path


def test_dataset_gap_filled_from_yahoo(client, monkeypatch):
    instance, data_dir = client
    dataset_last = date.today() - timedelta(days=3)
    dataset = _bars([str(dataset_last - timedelta(days=1)), str(dataset_last)])
    (data_dir / "SOXL.csv").write_text(dataset.to_csv())

    fresh_day = dataset_last + timedelta(days=1)
    fresh = _bars([str(fresh_day)], close_start=200.0)
    calls = {}

    def fake_yahoo(self, ticker, start, end):
        calls["range"] = (start, end)
        return fresh

    monkeypatch.setattr(MarketDataClient, "_from_yahoo", fake_yahoo)
    result = instance.download("SOXL", dataset_last - timedelta(days=30), date.today() + timedelta(days=1))

    assert calls["range"][0] == fresh_day
    assert result.index.max().date() == fresh_day
    assert result.loc[pd.Timestamp(fresh_day), "Close"] == 200.0
    assert len(result) == 3


def test_dataset_served_as_is_when_gap_fetch_fails(client, monkeypatch):
    instance, data_dir = client
    dataset_last = date.today() - timedelta(days=3)
    dataset = _bars([str(dataset_last)])
    (data_dir / "SOXL.csv").write_text(dataset.to_csv())

    def failing_yahoo(self, ticker, start, end):
        raise MarketDataError("blocked")

    monkeypatch.setattr(MarketDataClient, "_from_yahoo", failing_yahoo)
    result = instance.download("SOXL", dataset_last - timedelta(days=30), date.today() + timedelta(days=1))
    assert result.index.max().date() == dataset_last


def test_stale_dataset_without_yahoo_raises(client, monkeypatch):
    instance, data_dir = client
    dataset_last = date.today() - timedelta(days=30)
    (data_dir / "SOXL.csv").write_text(_bars([str(dataset_last)]).to_csv())

    def failing_yahoo(self, ticker, start, end):
        raise MarketDataError("blocked")

    monkeypatch.setattr(MarketDataClient, "_from_yahoo", failing_yahoo)
    with pytest.raises(MarketDataError):
        instance.download("SOXL", dataset_last - timedelta(days=30), date.today() + timedelta(days=1))


def test_no_gap_fetch_when_dataset_is_current(client, monkeypatch):
    instance, data_dir = client
    (data_dir / "SOXL.csv").write_text(_bars([str(date.today())]).to_csv())

    def unexpected_yahoo(self, ticker, start, end):
        raise AssertionError("Yahoo should not be called")

    monkeypatch.setattr(MarketDataClient, "_from_yahoo", unexpected_yahoo)
    result = instance.download("SOXL", date.today() - timedelta(days=10), date.today() + timedelta(days=1))
    assert result.index.max().date() == date.today()
