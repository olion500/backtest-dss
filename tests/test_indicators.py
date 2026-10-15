"""Tests for indicator functions."""
import numpy as np
import pandas as pd
import pytest

from engines.dongpa_engine import (
    wilder_rsi,
    cross_up,
    cross_down,
    moving_average,
    golden_cross,
    death_cross,
    weekly_roc,
    to_weekly_close,
)


class TestWilderRSI:
    def test_returns_series(self):
        prices = pd.Series([100, 101, 102, 101, 100, 99, 100, 101, 102, 103,
                            104, 103, 102, 101, 100, 99, 98, 99, 100, 101])
        rsi = wilder_rsi(prices, period=14)
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(prices)

    def test_hand_computed_value(self):
        # period 2 → alpha 0.5. Δ = [+2, −1]: avg up 2 → 1, avg down 0 → 0.5, RS = 2
        rsi = wilder_rsi(pd.Series([10.0, 12.0, 11.0]), period=2)
        assert rsi.iloc[-1] == pytest.approx(100 - 100 / 3)
        # one more −1: avg up 0.5, avg down 0.75 → RS 2/3 → RSI 40
        rsi = wilder_rsi(pd.Series([10.0, 12.0, 11.0, 10.0]), period=2)
        assert rsi.iloc[-1] == pytest.approx(40.0)

    def test_fills_nan_with_50(self):
        prices = pd.Series([100.0])
        rsi = wilder_rsi(prices, period=14)
        assert rsi.iloc[0] == 50.0

    def test_mostly_up_high_rsi(self):
        # Mostly up with small dips to avoid 0 denominator
        prices = [100, 102, 101, 103, 104, 103.5, 105, 106, 105.5, 107,
                  108, 107.5, 109, 110, 109.5, 111, 112, 111.5, 113, 114]
        rsi = wilder_rsi(pd.Series(prices), period=5)
        assert rsi.iloc[-1] > 65

    def test_all_down_near_0(self):
        prices = pd.Series(range(120, 100, -1))
        rsi = wilder_rsi(prices, period=5)
        assert rsi.iloc[-1] < 10


class TestCrossUpDown:
    def test_cross_up(self):
        assert cross_up(49.0, 51.0, 50.0) is True
        assert cross_up(50.0, 51.0, 50.0) is False  # prev not < level
        assert cross_up(49.0, 49.5, 50.0) is False  # curr < level
        assert cross_up(49.0, 50.0, 50.0) is True   # landing on the level counts

    def test_cross_down(self):
        assert cross_down(51.0, 49.0, 50.0) is True
        assert cross_down(50.0, 49.0, 50.0) is False  # prev not > level
        assert cross_down(51.0, 50.5, 50.0) is False  # curr > level
        assert cross_down(51.0, 50.0, 50.0) is True   # landing on the level counts


class TestMovingAverage:
    def test_single_value(self):
        s = pd.Series([10.0])
        result = moving_average(s, 5)
        assert result.iloc[0] == 10.0

    def test_matches_manual(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moving_average(s, 3)
        assert result.iloc[-1] == pytest.approx(4.0)  # (3+4+5)/3


class TestGoldenDeathCross:
    def test_golden_cross(self):
        short = pd.Series([10.0, 10.0, 11.0])
        long = pd.Series([11.0, 11.0, 10.0])
        gc = golden_cross(short, long)
        assert gc.tolist() == [False, False, True]

    def test_golden_cross_from_touching(self):
        # equal yesterday, above today → cross; already above → no new cross
        gc = golden_cross(pd.Series([10.0, 11.0, 12.0]), pd.Series([10.0, 10.0, 10.0]))
        assert gc.tolist() == [False, True, False]

    def test_death_cross(self):
        short = pd.Series([11.0, 11.0, 10.0])
        long = pd.Series([10.0, 10.0, 11.0])
        dc = death_cross(short, long)
        assert dc.tolist() == [False, False, True]

    def test_death_cross_from_touching(self):
        dc = death_cross(pd.Series([10.0, 9.0, 8.0]), pd.Series([10.0, 10.0, 10.0]))
        assert dc.tolist() == [False, True, False]


class TestWeeklyROC:
    def test_positive_roc(self):
        s = pd.Series([100, 101, 102, 103, 110])
        roc = weekly_roc(s, period=1)
        assert roc.iloc[-1] == pytest.approx(110 / 103 - 1)
        assert weekly_roc(s, period=2).iloc[-1] == pytest.approx(110 / 102 - 1)
        assert np.isnan(weekly_roc(s, period=2).iloc[1])

    def test_negative_roc(self):
        s = pd.Series([110, 109, 108, 107, 100])
        roc = weekly_roc(s, period=1)
        assert roc.iloc[-1] == pytest.approx(100 / 107 - 1)


class TestToWeeklyClose:
    def test_last_close_of_each_friday_week(self):
        # Mon 2024-01-01 … Wed 2024-01-10
        idx = pd.bdate_range("2024-01-01", periods=8)
        df = pd.DataFrame({"Close": [1.0, 2, 3, 4, 5, 6, 7, 8]}, index=idx)
        weekly = to_weekly_close(df)
        assert weekly.index.tolist() == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")]
        assert weekly.tolist() == [5.0, 8.0]
