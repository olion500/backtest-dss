"""Tests for metrics functions."""
import pandas as pd
import pytest

from engines.dongpa_engine import (
    CAGR,
    max_drawdown,
    summarize,
    compute_buy_and_hold_return,
    compute_equity_return,
    compute_trade_metrics,
    compute_mode_bands,
)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        s = pd.Series([100, 101, 102, 103], index=pd.date_range("2024-01-01", periods=4))
        assert max_drawdown(s) == pytest.approx(0.0)

    def test_drawdown(self):
        s = pd.Series([100, 90, 80, 100], index=pd.date_range("2024-01-01", periods=4))
        assert max_drawdown(s) == pytest.approx(-0.2)  # 80/100 - 1

    def test_full_recovery(self):
        s = pd.Series([100, 50, 100], index=pd.date_range("2024-01-01", periods=3))
        assert max_drawdown(s) == pytest.approx(-0.5)


class TestCAGR:
    def test_positive_return(self):
        dates = pd.date_range("2020-01-01", "2021-01-01", periods=2)
        s = pd.Series([100, 110], index=dates)
        # 366 days → (110/100)^(365.25/366) − 1
        assert CAGR(s) == pytest.approx(1.1 ** (365.25 / 366) - 1)

    def test_two_years_doubling(self):
        dates = pd.DatetimeIndex(["2020-01-01", "2021-12-31"])
        s = pd.Series([100.0, 400.0], index=dates)
        years = (dates[-1] - dates[0]).days / 365.25
        assert CAGR(s) == pytest.approx(4 ** (1 / years) - 1)

    def test_short_period_and_small_start(self):
        # half a year (182 days) from 0.5 to 1.0 → 2^(365.25/182) − 1
        s = pd.Series([0.5, 1.0], index=pd.DatetimeIndex(["2020-01-01", "2020-07-01"]))
        assert CAGR(s) == pytest.approx(2 ** (365.25 / 182) - 1)

    def test_single_point_returns_zero(self):
        assert CAGR(pd.Series([100.0], index=pd.DatetimeIndex(["2020-01-01"]))) == 0.0

    def test_zero_start_returns_zero(self):
        dates = pd.date_range("2020-01-01", periods=2)
        s = pd.Series([0, 100], index=dates)
        assert CAGR(s) == 0.0

    def test_empty_returns_zero(self):
        assert CAGR(pd.Series(dtype=float)) == 0.0


class TestSummarize:
    def test_returns_dict(self):
        s = pd.Series([100, 101, 102], index=pd.date_range("2024-01-01", periods=3))
        result = summarize(s)
        assert "Final Equity" in result
        assert "CAGR" in result
        assert "Max Drawdown" in result
        assert "Sharpe (rf=0)" in result
        assert "Calmar Ratio" in result

    def test_values(self):
        idx = pd.DatetimeIndex(["2020-01-01", "2020-07-01", "2021-01-01"])
        s = pd.Series([100.0, 80.0, 120.0], index=idx)
        result = summarize(s)
        ret = pd.Series([-0.2, 0.5])
        cagr = 1.2 ** (365.25 / 366) - 1
        assert result["Final Equity"] == 120.0
        assert result["CAGR"] == pytest.approx(cagr)
        assert result["Max Drawdown"] == pytest.approx(-0.2)
        assert result["Calmar Ratio"] == pytest.approx(cagr / 0.2)
        assert result["Volatility (ann)"] == pytest.approx(ret.std() * 252 ** 0.5)
        assert result["Sharpe (rf=0)"] == pytest.approx(ret.mean() / ret.std() * 252 ** 0.5)

    def test_flat_equity_has_zero_ratios(self):
        s = pd.Series([100.0, 100.0, 100.0], index=pd.date_range("2024-01-01", periods=3))
        result = summarize(s)
        assert result["Sharpe (rf=0)"] == 0.0
        assert result["Calmar Ratio"] == 0.0
        assert result["Max Drawdown"] == 0.0


class TestComputeBuyAndHoldReturn:
    def test_positive_return(self):
        df = pd.DataFrame({"Close": [100, 110]}, index=pd.date_range("2024-01-01", periods=2))
        assert compute_buy_and_hold_return(df) == pytest.approx(10.0)

    def test_uses_first_and_last_valid_close(self):
        df = pd.DataFrame({"Close": [None, 80, 90, 60, None]}, index=pd.date_range("2024-01-01", periods=5))
        assert compute_buy_and_hold_return(df) == pytest.approx(-25.0)

    def test_empty_df(self):
        assert compute_buy_and_hold_return(pd.DataFrame()) is None
        no_close = pd.DataFrame({"Open": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))
        assert compute_buy_and_hold_return(no_close) is None

    def test_zero_start(self):
        df = pd.DataFrame({"Close": [0, 100]}, index=pd.date_range("2024-01-01", periods=2))
        assert compute_buy_and_hold_return(df) is None


class TestComputeEquityReturn:
    def test_positive(self):
        s = pd.Series([100, 120])
        assert compute_equity_return(s) == pytest.approx(20.0)
        assert compute_equity_return(pd.Series([100, 150, 90])) == pytest.approx(-10.0)

    def test_zero_start(self):
        assert compute_equity_return(pd.Series([0.0, 100.0])) is None

    def test_empty(self):
        assert compute_equity_return(pd.Series(dtype=float)) is None


class TestComputeTradeMetrics:
    def test_none_input(self):
        assert compute_trade_metrics(None, 10000) is None

    def test_empty_df(self):
        assert compute_trade_metrics(pd.DataFrame(), 10000) is None

    def test_closed_trades(self):
        df = pd.DataFrame({
            "상태": ["완료", "완료", "완료", "보유중"],
            "실현손익": [100.0, -50.0, 40.0, None],
            "보유기간(일)": [5, 3, 1, 9],
            "수익률(%)": [2.0, -1.0, 4.0, None],
            "청산사유": ["TP", "MOC", "TP", None],
        })
        result = compute_trade_metrics(df, 10000)
        assert result["trade_count"] == 3  # the open lot is not a trade yet
        assert result["moc_count"] == 1
        assert result["net_profit"] == pytest.approx(90.0)
        assert result["avg_hold_days"] == pytest.approx(3.0)
        assert result["avg_return_pct"] == pytest.approx(5.0 / 3)
        assert result["avg_gain"] == pytest.approx(70.0)
        assert result["avg_loss"] == pytest.approx(-50.0)
        assert result["avg_gain_pct"] == pytest.approx(3.0)
        assert result["avg_loss_pct"] == pytest.approx(-1.0)
        assert result["period_return_pct"] == pytest.approx(0.9)

    def test_break_even_trade_is_neither_gain_nor_loss(self):
        df = pd.DataFrame({
            "상태": ["완료", "완료"], "실현손익": [0.0, 30.0], "보유기간(일)": [2, 2],
            "수익률(%)": [0.0, 1.0], "청산사유": ["TP", "TP"],
        })
        result = compute_trade_metrics(df, 10000)
        assert result["avg_gain"] == pytest.approx(30.0)
        assert result["avg_loss"] is None
        assert result["avg_gain_pct"] == pytest.approx(1.0)
        assert result["avg_loss_pct"] is None
        assert compute_trade_metrics(df, 0)["period_return_pct"] is None

    def test_only_open_trades(self):
        df = pd.DataFrame({"상태": ["보유중"], "실현손익": [None]})
        result = compute_trade_metrics(df, 10000)
        assert result["trade_count"] == 0
        assert result["net_profit"] == 0.0
        assert result["avg_gain"] is None


class TestComputeModeBands:
    def test_empty_journal(self):
        result = compute_mode_bands(pd.DataFrame())
        assert result.empty
        assert compute_mode_bands(pd.DataFrame({"거래일자": ["2024-01-01"]})).empty

    def test_single_mode(self):
        df = pd.DataFrame({
            "거래일자": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "모드": ["안전", "안전", "안전"],
        })
        result = compute_mode_bands(df)
        assert len(result) == 1
        assert result.iloc[0]["mode"] == "안전"

    def test_mode_switch(self):
        df = pd.DataFrame({
            "거래일자": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "모드": ["안전", "안전", "공세", "공세"],
        })
        result = compute_mode_bands(df)
        assert len(result) == 2
        assert result.iloc[0]["mode"] == "안전"
        assert result.iloc[1]["mode"] == "공세"
        # each band runs from its first day to the day after its last
        assert result.iloc[0]["start"] == pd.Timestamp("2024-01-01")
        assert result.iloc[0]["end"] == pd.Timestamp("2024-01-03")
        assert result.iloc[1]["start"] == pd.Timestamp("2024-01-03")
        assert result.iloc[1]["end"] == pd.Timestamp("2024-01-05")
