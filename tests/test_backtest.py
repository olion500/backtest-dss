"""Integration tests for run_backtest."""
import numpy as np
import pandas as pd
import pytest

from engines.dongpa_engine import (
    BacktestResult,
    CapitalParams,
    ModeParams,
    StrategyParams,
    run_backtest,
    compute_indicators,
    Indicators,
)
from tests.conftest import _make_price_df


@pytest.fixture
def long_prices():
    """100 trading days of uptrending data."""
    np.random.seed(42)
    base = 50.0
    prices = [base]
    for _ in range(99):
        prices.append(prices[-1] * (1 + np.random.normal(0.002, 0.03)))
    return _make_price_df(prices, start="2024-01-02")


@pytest.fixture
def long_momo():
    """100 trading days of momentum data."""
    np.random.seed(7)
    base = 400.0
    prices = [base]
    for _ in range(99):
        prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.01)))
    return _make_price_df(prices, start="2024-01-02")


class TestRunBacktest:
    def test_returns_backtest_result(self, long_prices, long_momo, default_params, default_cap):
        result = run_backtest(long_prices, long_momo, default_params, default_cap)
        assert isinstance(result, BacktestResult)
        assert isinstance(result.equity, pd.Series)
        assert isinstance(result.journal, pd.DataFrame)
        assert isinstance(result.trade_log, pd.DataFrame)
        assert isinstance(result.cash_end, float)
        assert isinstance(result.open_positions, int)

    def test_equity_length_matches_dates(self, long_prices, long_momo, default_params, default_cap):
        result = run_backtest(long_prices, long_momo, default_params, default_cap)
        # One equity point and one journal row per trading day
        assert list(result.equity.index) == list(long_prices.index)
        assert len(result.journal) == len(long_prices)

    def test_initial_equity_equals_cash(self, long_prices, long_momo, default_params, default_cap):
        result = run_backtest(long_prices, long_momo, default_params, default_cap)
        # First day: no buy (i=0), equity = cash
        assert result.equity.iloc[0] == default_cap.initial_cash

    def test_empty_data_returns_empty_result(self, default_params, default_cap):
        empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        empty_df.index = pd.DatetimeIndex([], name="Date")
        momo = _make_price_df([400.0], start="2024-01-02")
        result = run_backtest(empty_df, momo, default_params, default_cap)
        assert result.equity.empty
        assert result.journal.empty

    def test_end_state_matches_last_journal_row(self, long_prices, long_momo, default_params, default_cap):
        result = run_backtest(long_prices, long_momo, default_params, default_cap)
        last = result.journal.iloc[-1]
        assert result.cash_end >= 0
        assert result.cash_end == last["현금"]
        assert result.equity.iloc[-1] == pytest.approx(last["현금"] + last["평가금액"])
        open_lots = result.trade_log[result.trade_log["상태"] == "보유중"]
        assert result.open_positions == len(open_lots)
        assert last["보유수량"] == open_lots["매수수량"].sum()


class TestComputeIndicators:
    def test_returns_tuple(self, long_prices, long_momo, default_params):
        df, ind = compute_indicators(long_prices, long_momo, default_params)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(ind, Indicators)
        assert ind.strategy == "rsi"

    def test_target_trimmed_to_momentum_period(self, default_params):
        target = _make_price_df([50.0] * 20, start="2024-01-02")   # Jan 2 – Jan 29
        momo = _make_price_df([400.0] * 10, start="2024-01-09")    # Jan 9 – Jan 22
        df, ind = compute_indicators(target, momo, default_params)
        assert df.index[0] == pd.Timestamp("2024-01-09")
        assert df.index[-1] == pd.Timestamp("2024-01-22")
        assert ind.daily_rsi.index.equals(df.index)

    def test_daily_rsi_is_last_completed_week(self, long_prices, long_momo, default_params):
        _, ind = compute_indicators(long_prices, long_momo, default_params)
        # Thursday 2024-02-08 still carries the RSI stamped on Friday 2024-02-02
        assert ind.daily_rsi.loc["2024-02-08"] == ind.weekly_rsi.loc["2024-02-02"]
        assert ind.daily_prev_week.loc["2024-02-08"] == ind.weekly_rsi.loc["2024-01-26"]
        assert ind.daily_rsi_delta.loc["2024-02-08"] == pytest.approx(
            ind.weekly_rsi.loc["2024-02-02"] - ind.weekly_rsi.loc["2024-01-26"]
        )

    def test_rsi_indicators_populated(self, long_prices, long_momo, default_params):
        _, ind = compute_indicators(long_prices, long_momo, default_params)
        assert ind.daily_rsi is not None
        assert ind.daily_rsi_delta is not None
        assert ind.daily_prev_week is not None

    def test_ma_cross_indicators(self, long_prices, long_momo, default_defense, default_offense):
        params = StrategyParams(
            target_ticker="TEST", momentum_ticker="MOMO",
            mode_switch_strategy="ma_cross",
            ma_short_period=3, ma_long_period=7,
            defense=default_defense, offense=default_offense,
        )
        _, ind = compute_indicators(long_prices, long_momo, params)
        assert ind.strategy == "ma_cross"
        assert ind.daily_ma_short is not None
        assert ind.daily_ma_long is not None
        assert ind.daily_golden is not None

    def test_roc_indicators(self, long_prices, long_momo, default_defense, default_offense):
        params = StrategyParams(
            target_ticker="TEST", momentum_ticker="MOMO",
            mode_switch_strategy="roc", roc_period=4,
            defense=default_defense, offense=default_offense,
        )
        _, ind = compute_indicators(long_prices, long_momo, params)
        assert ind.strategy == "roc"
        assert ind.daily_roc is not None
