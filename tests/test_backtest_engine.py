import tempfile
from pathlib import Path

import pytest

from backtest_engine import BacktestEngine, BacktestSettings, StrategyParameters
from backtest_local import read_price_series


def test_engine_generates_positive_return_from_rebound_cycle():
    prices = [100.0, 97.0, 95.0, 99.0, 102.0]
    settings = BacktestSettings(initial_capital=1000.0, commission_rate=0.0, trading_days_per_year=5)
    params = StrategyParameters(divisions=5, max_hold_days=5, buy_threshold_pct=1.0, sell_threshold_pct=1.5)
    engine = BacktestEngine(prices)

    metrics = engine.run(settings, params, params)

    assert metrics.final_capital > settings.initial_capital
    assert metrics.cagr_pct > 0
    assert metrics.max_drawdown_pct >= 0
    assert len(metrics.equity_curve) == len(prices)


def test_read_price_series_requires_close_column(tmp_path: Path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("date,open\n2024-01-01,10\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_price_series(csv_path)


def test_read_price_series_parses_closes(tmp_path: Path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("date,close\n2024-01-01,10\n2024-01-02,11\n", encoding="utf-8")
    closes = read_price_series(csv_path)
    assert closes == [10.0, 11.0]
