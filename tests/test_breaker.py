"""Stop-loss streak breaker (StrategyParams.sl_breaker_*), hand-checked scenarios.

The base mode signal is pinned with monkeypatch so only the breaker can move the mode.
Offense buys every session (limit +50%) with a 10% stop; defense never buys (limit -50%),
so a forced defense session shows as 모드 안전 with no buy.
"""
import pandas as pd
import pytest

import engines.dongpa_engine as E
from engines.dongpa_engine import CapitalParams, ModeParams, StrategyParams, run_backtest
from tests.conftest import _make_price_df

OFFENSE = ModeParams(buy_cond_pct=50.0, tp_pct=50.0, max_hold_days=99, slices=4, stop_loss_pct=10.0)
DEFENSE = ModeParams(buy_cond_pct=-50.0, tp_pct=50.0, max_hold_days=99, slices=4)
START = "2024-03-04"


def _pin_base(monkeypatch, mode_for):
    """Base signal: mode_for(i) is the mode of session i (decided at the end of session i - 1)."""
    dates = pd.bdate_range(START, periods=40)
    monkeypatch.setattr(E, "determine_initial_mode", lambda start, ind, p: mode_for(0))
    monkeypatch.setattr(E, "decide_mode", lambda idx, prev, ind, p: mode_for(dates.get_loc(idx) + 1))


def _run(closes, offense=OFFENSE, defense=DEFENSE, **breaker):
    params = StrategyParams(target_ticker="TEST", momentum_ticker="MOMO", defense=defense, offense=offense,
                            enable_netting=False, **breaker)
    momo = _make_price_df([400.0] * 150, start="2023-10-02")
    return run_backtest(_make_price_df(closes, start=START), momo, params, CapitalParams(initial_cash=10_000))


# sessions 1, 2 buy @100 (stop 90); session 3 closes 85: both stop → 2 offense stops on session 3
TWO_STOPS = [100.0, 100.0, 100.0, 85.0] + [85.0] * 8


def test_two_offense_stops_force_defense_for_the_next_sessions(monkeypatch):
    _pin_base(monkeypatch, lambda i: "offense")
    r = _run(TWO_STOPS, sl_breaker_count=2, sl_breaker_window=10, sl_breaker_days=5)
    modes = r.journal["모드"].tolist()
    assert (r.trade_log["청산사유"] == "SL").sum() == 2
    assert modes[:4] == ["공세"] * 4
    assert modes[4:9] == ["안전"] * 5
    assert modes[9:] == ["공세"] * 3
    assert r.journal["안전강제(일)"].tolist()[3:9] == [5, 4, 3, 2, 1, 0]
    assert r.journal["다음모드"].iloc[3] == "안전"
    assert (r.journal["매수수량"].iloc[4:9] == 0).all()


def test_off_by_default_changes_nothing(monkeypatch):
    _pin_base(monkeypatch, lambda i: "offense")
    r = _run(TWO_STOPS)
    assert r.journal["모드"].tolist() == ["공세"] * len(TWO_STOPS)
    assert "안전강제(일)" not in r.journal.columns


def test_count_not_reached_keeps_the_signal(monkeypatch):
    _pin_base(monkeypatch, lambda i: "offense")
    r = _run(TWO_STOPS, sl_breaker_count=3, sl_breaker_window=10, sl_breaker_days=5)
    assert r.journal["모드"].tolist() == ["공세"] * len(TWO_STOPS)


# session 1 buys @100 (stop 90) → stops on session 2 (89); session 2 buys @89 (stop 80.10) → stops on session 6 (80)
SPREAD_STOPS = [100.0, 100.0, 89.0, 88.5, 88.5, 88.5, 80.0] + [80.0] * 4


@pytest.mark.parametrize("window, forced", [(3, False), (10, True)])
def test_only_stops_inside_the_window_count(monkeypatch, window, forced):
    _pin_base(monkeypatch, lambda i: "offense")
    r = _run(SPREAD_STOPS, sl_breaker_count=2, sl_breaker_window=window, sl_breaker_days=2)
    stops = r.trade_log[r.trade_log["청산사유"] == "SL"]
    assert stops["매도일자"].tolist()[:2] == ["2024-03-06", "2024-03-12"]
    assert (r.journal["모드"].iloc[7:9] == "안전").all() == forced


def test_stops_of_defense_lots_do_not_count(monkeypatch):
    # sessions 0-3 defense, offense from session 4; defense buys here and stops like offense
    _pin_base(monkeypatch, lambda i: "defense" if i < 4 else "offense")
    defense_buying = ModeParams(buy_cond_pct=50.0, tp_pct=50.0, max_hold_days=99, slices=4, stop_loss_pct=10.0)
    r = _run(TWO_STOPS, defense=defense_buying, sl_breaker_count=2, sl_breaker_window=10, sl_breaker_days=5)
    stopped = r.trade_log[r.trade_log["청산사유"] == "SL"]
    assert set(stopped["매수모드"]) == {"안전"} and len(stopped) == 2
    assert r.journal["모드"].iloc[4:].eq("공세").all()


def test_stops_before_the_start_date_do_not_count(monkeypatch):
    _pin_base(monkeypatch, lambda i: "offense")
    start = pd.bdate_range(START, periods=5)[4]
    r = _run(TWO_STOPS, sl_breaker_count=2, sl_breaker_window=10, sl_breaker_days=5, sl_breaker_start=start)
    assert r.journal["모드"].tolist() == ["공세"] * len(TWO_STOPS)


def test_btc_overnight_is_rejected():
    with pytest.raises(ValueError):
        StrategyParams(target_ticker="T", momentum_ticker="M", defense=DEFENSE, offense=OFFENSE,
                       mode_switch_strategy="btc_overnight", sl_breaker_count=2)
