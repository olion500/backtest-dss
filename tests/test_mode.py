"""Tests for mode decision pure functions."""
import pytest

from engines.dongpa_engine import (
    ModeParams,
    StrategyParams,
    eval_rsi_conditions,
)


@pytest.fixture
def rsi_params():
    d = ModeParams(buy_cond_pct=3.0, tp_pct=0.2, max_hold_days=30, slices=7)
    o = ModeParams(buy_cond_pct=5.0, tp_pct=2.5, max_hold_days=7, slices=7)
    return StrategyParams(
        target_ticker="TEST", momentum_ticker="MOMO",
        defense=d, offense=o,
        rsi_high_threshold=65.0,
        rsi_mid_low=40.0,
        rsi_mid_high=60.0,
        rsi_low_threshold=35.0,
        rsi_neutral=50.0,
    )


class TestEvalRsiConditions:
    def test_high_rsi_declining_triggers_defense(self, rsi_params):
        # RSI=70, delta=-2 → defense (high RSI + declining)
        result = eval_rsi_conditions(70.0, 72.0, -2.0, rsi_params)
        assert result == "defense"

    def test_low_rsi_rising_triggers_offense(self, rsi_params):
        # RSI=30, delta=+3 → offense (below low threshold + rising)
        result = eval_rsi_conditions(30.0, 27.0, 3.0, rsi_params)
        assert result == "offense"

    def test_cross_up_50_triggers_offense(self, rsi_params):
        # prev_w=49, rsi=51, delta=+2 → cross_up(49, 51, 50) = True
        result = eval_rsi_conditions(51.0, 49.0, 2.0, rsi_params)
        assert result == "offense"

    def test_cross_down_50_triggers_defense(self, rsi_params):
        # prev_w=51, rsi=49, delta=-2 → cross_down(51, 49, 50) = True
        result = eval_rsi_conditions(49.0, 51.0, -2.0, rsi_params)
        assert result == "defense"

    def test_mid_range_neutral_returns_none(self, rsi_params):
        # RSI=55, delta=0 → no clear signal
        result = eval_rsi_conditions(55.0, 55.0, 0.0, rsi_params)
        assert result is None

    def test_mid_low_declining_triggers_defense(self, rsi_params):
        # RSI=45 (between mid_low=40 and neutral=50), delta=-1
        result = eval_rsi_conditions(45.0, 46.0, -1.0, rsi_params)
        assert result == "defense"

    def test_mid_high_rising_triggers_offense(self, rsi_params):
        # RSI=55 (between neutral=50 and mid_high=60), delta=+1
        result = eval_rsi_conditions(55.0, 54.0, 1.0, rsi_params)
        assert result == "offense"

    def test_high_threshold_is_inclusive(self, rsi_params):
        # RSI exactly 65 and declining → defense (RSI ≥ 65)
        assert eval_rsi_conditions(65.0, 66.0, -1.0, rsi_params) == "defense"

    def test_high_rsi_rising_holds(self, rsi_params):
        # RSI ≥ 65 only matters while declining; 64.9 declining is outside every band
        assert eval_rsi_conditions(70.0, 68.0, 2.0, rsi_params) is None
        assert eval_rsi_conditions(64.9, 66.0, -1.1, rsi_params) is None

    def test_mid_bands_are_exclusive(self, rsi_params):
        # 40 < RSI < 50 declining and 50 < RSI < 60 rising: the edges themselves don't count
        assert eval_rsi_conditions(40.0, 41.0, -1.0, rsi_params) is None
        assert eval_rsi_conditions(60.0, 59.0, 1.0, rsi_params) is None
        assert eval_rsi_conditions(59.9, 59.0, 0.9, rsi_params) == "offense"
        assert eval_rsi_conditions(40.1, 41.0, -0.9, rsi_params) == "defense"

    def test_mid_low_rising_and_mid_high_declining_hold(self, rsi_params):
        assert eval_rsi_conditions(45.0, 44.0, 1.0, rsi_params) is None
        assert eval_rsi_conditions(55.0, 56.0, -1.0, rsi_params) is None

    def test_low_threshold_is_exclusive(self, rsi_params):
        assert eval_rsi_conditions(35.0, 34.0, 1.0, rsi_params) is None
        assert eval_rsi_conditions(34.9, 34.0, 0.9, rsi_params) == "offense"
        # below 35 but still falling → no signal
        assert eval_rsi_conditions(30.0, 32.0, -2.0, rsi_params) is None

    def test_cross_onto_neutral_line(self, rsi_params):
        assert eval_rsi_conditions(50.0, 49.0, 1.0, rsi_params) == "offense"
        assert eval_rsi_conditions(50.0, 51.0, -1.0, rsi_params) == "defense"

    def test_thresholds_come_from_params(self, rsi_params):
        from dataclasses import replace
        custom = replace(rsi_params, rsi_high_threshold=80.0, rsi_low_threshold=20.0)
        assert eval_rsi_conditions(70.0, 72.0, -2.0, custom) is None
        assert eval_rsi_conditions(30.0, 27.0, 3.0, custom) is None


def _rsi_indicators(rsi: list[float], dates):
    import pandas as pd
    from engines.dongpa_engine import Indicators
    w = pd.Series(rsi, index=dates)
    return Indicators(
        strategy="rsi",
        weekly_rsi=w,
        weekly_rsi_delta=w.diff(),
        daily_rsi=w,
        daily_rsi_delta=w.diff(),
        daily_prev_week=w.shift(1),
    )


class TestRsiModeReplay:
    def test_decide_mode_keeps_previous_mode_without_signal(self, rsi_params):
        import pandas as pd
        from engines.dongpa_engine import decide_mode
        dates = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
        ind = _rsi_indicators([30.0, 33.0, 33.0], dates)
        # week 2: RSI < 35 and rising → offense; week 3: Δ = 0 → keep whatever we had
        assert decide_mode(dates[1], "defense", ind, rsi_params) == "offense"
        assert decide_mode(dates[2], "offense", ind, rsi_params) == "offense"
        assert decide_mode(dates[2], "defense", ind, rsi_params) == "defense"
        # first week: no previous week, Δ NaN → 0 → keep
        assert decide_mode(dates[0], "offense", ind, rsi_params) == "offense"

    def test_initial_mode_replays_weeks_before_start(self, rsi_params):
        import pandas as pd
        from engines.dongpa_engine import determine_initial_mode
        dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
        # 30 → 33 rising below 35: offense; 70 → 68 declining above 65: defense
        ind = _rsi_indicators([30.0, 33.0, 70.0, 68.0], dates)
        assert determine_initial_mode(dates[1] + pd.Timedelta(days=1), ind, rsi_params) == "offense"
        # the week stamped on the start day itself is not known yet
        assert determine_initial_mode(dates[3], ind, rsi_params) == "offense"
        assert determine_initial_mode(dates[3] + pd.Timedelta(days=1), ind, rsi_params) == "defense"
        assert determine_initial_mode(dates[0], ind, rsi_params) == "defense"  # no history

    def test_initial_mode_replay_uses_the_previous_week_for_crosses(self, rsi_params):
        import pandas as pd
        from engines.dongpa_engine import determine_initial_mode
        dates = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
        # 70 → 48: 40 < RSI < 50 declining → defense; 48 → 62: 50-line cross up → offense
        # (62 is above the 50–60 band, so only the cross against last week's 48 fires)
        ind = _rsi_indicators([70.0, 48.0, 62.0], dates)
        assert determine_initial_mode(dates[2] + pd.Timedelta(days=1), ind, rsi_params) == "offense"

    def test_missing_rsi_keeps_previous_mode(self, rsi_params):
        import pandas as pd
        from engines.dongpa_engine import decide_mode
        dates = pd.date_range("2024-01-05", periods=2, freq="W-FRI")
        ind = _rsi_indicators([float("nan"), 30.0], dates)
        assert decide_mode(dates[0], "offense", ind, rsi_params) == "offense"
        assert decide_mode(dates[0], None, ind, rsi_params) == "defense"


class TestRsiStrictness:
    def test_flat_rsi_is_neither_rising_nor_falling(self, rsi_params):
        assert eval_rsi_conditions(70.0, 70.0, 0.0, rsi_params) is None
        assert eval_rsi_conditions(30.0, 30.0, 0.0, rsi_params) is None

    def test_sitting_on_the_neutral_line_is_not_a_band(self, rsi_params):
        # prev 50 → 50: no cross, and 50 is outside both open bands
        assert eval_rsi_conditions(50.0, 50.0, -1.0, rsi_params) is None
        assert eval_rsi_conditions(50.0, 50.0, 1.0, rsi_params) is None


class TestTrendVixMode:
    def test_needs_both_signals(self):
        from engines.dongpa_engine import _trend_vix_mode
        nan = float("nan")
        assert _trend_vix_mode(1.0, 1.0) == "defense"
        assert _trend_vix_mode(1.0, 0.0) == "offense"
        assert _trend_vix_mode(0.0, 1.0) == "offense"
        assert _trend_vix_mode(nan, 1.0) is None
        assert _trend_vix_mode(1.0, nan) is None

    def test_initial_mode_is_last_state_before_start(self):
        import pandas as pd
        from engines.dongpa_engine import Indicators, determine_initial_mode
        idx = pd.date_range("2024-01-01", periods=4)
        state = pd.Series([None, "offense", "defense", "offense"], index=idx, dtype=object)
        ind = Indicators(strategy="trend_vix", trend_vix_state=state)
        params = StrategyParams(target_ticker="T", momentum_ticker="M", mode_switch_strategy="trend_vix",
                                defense=ModeParams(1, 1, 1, 1), offense=ModeParams(1, 1, 1, 1))
        assert determine_initial_mode(idx[3], ind, params) == "defense"   # idx[3] itself not known yet
        assert determine_initial_mode(idx[2], ind, params) == "offense"
        assert determine_initial_mode(idx[1], ind, params) == "defense"   # only None before → default
