"""trend_vix mode: momentum SMA band + VIX band, strong market → defense, otherwise offense."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from engines.dongpa_engine import (
    ModeParams,
    StrategyParams,
    CapitalParams,
    compute_indicators,
    determine_initial_mode,
    hysteresis_state,
    run_backtest,
)
from tests.conftest import _make_price_df
from web.api.app.schemas import BacktestRequest, ModeSettings, StrategySettings
from web.api.app.services.backtest import run_backtest_view, run_order_book_view


def test_hysteresis_state_holds_inside_the_band():
    enter = pd.Series([False, True, False, False, False, False])
    leave = pd.Series([False, False, False, True, False, False])
    state = hysteresis_state(enter, leave)
    assert state.isna().iloc[0]
    assert state.iloc[1:].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def params():
    return StrategyParams(
        target_ticker="TEST",
        momentum_ticker="MOMO",
        mode_switch_strategy="trend_vix",
        trend_sma_days=10,
        trend_band_pct=2.0,
        vix_low=20.0,
        vix_high=30.0,
        defense=ModeParams(buy_cond_pct=3.5, tp_pct=1.25, max_hold_days=24, slices=8, stop_loss_pct=5.0),
        offense=ModeParams(buy_cond_pct=3.5, tp_pct=3.25, max_hold_days=16, slices=2, stop_loss_pct=20.0),
    )


@pytest.fixture
def market():
    """Momentum rallies, then sells off; VIX stays calm except for a spike mid-rally."""
    n = 160
    momo = np.r_[np.linspace(100, 160, 80), np.linspace(160, 110, 80)]
    vix = np.full(n, 15.0)
    vix[40:50] = 35.0   # stressed
    vix[50:60] = 25.0   # inside the band → stays stressed
    target = 50 * (1 + 0.01 * np.sin(np.arange(n) / 3))
    return (
        _make_price_df(list(target), start="2024-01-02"),
        _make_price_df(list(momo), start="2024-01-02"),
        _make_price_df(list(vix), start="2024-01-02"),
    )


def test_modes_follow_trend_and_vix(params, market):
    target, momo, vix = market
    _, ind = compute_indicators(target, momo, params, vix_data=vix)
    days = target.index
    assert ind.daily_trend_up.iloc[30] == 1 and ind.daily_vix_calm.iloc[30] == 1
    assert ind.trend_vix_state[days[30]] == "defense"     # rally, calm VIX
    assert ind.trend_vix_state[days[45]] == "offense"     # VIX spike
    assert ind.trend_vix_state[days[55]] == "offense"     # VIX back inside the band: still stressed
    assert ind.trend_vix_state[days[65]] == "defense"     # VIX calm again
    assert ind.trend_vix_state[days[120]] == "offense"    # momentum sold off below the band


def test_backtest_trades_the_previous_sessions_state(params, market):
    target, momo, vix = market
    start = target.index[20]
    result = run_backtest(target.loc[start:], momo, params, CapitalParams(initial_cash=10_000), vix_data=vix)
    _, ind = compute_indicators(target.loc[start:], momo, params, vix_data=vix)
    j = result.journal
    expected = ind.trend_vix_state.reindex(target.index).shift(1).loc[start:]
    labels = expected.map({"defense": "안전", "offense": "공세"})
    assert (j["모드"].iloc[1:].values == labels.iloc[1:].values).all()
    assert determine_initial_mode(start, ind, params) == ind.trend_vix_state[target.index[19]]


def test_requires_vix_data(params, market):
    target, momo, _ = market
    with pytest.raises(ValueError, match="vix_data"):
        compute_indicators(target, momo, params)


def _web_request() -> BacktestRequest:
    mode = ModeSettings(slices=4, buy_cond_pct=3.5, tp_pct=2, max_hold_days=20)
    return BacktestRequest(
        start_date=date(2024, 3, 1),
        end_date=date(2024, 5, 20),
        initial_cash=10_000,
        strategy=StrategySettings(
            target_ticker="TEST", momentum_ticker="MOMO", mode_switch_strategy="trend_vix",
            trend_sma_days=10, trend_band_pct=0.5, vix_ticker="^VIX", defense=mode, offense=mode,
        ),
    )


def _provider(ticker: str, start: date, end: date) -> pd.DataFrame:
    if ticker == "^VIX":
        return _make_price_df([15.0] * 200, start="2023-10-02")
    if ticker == "TEST":
        return _make_price_df([50 * (1 + 0.001 * i) for i in range(100)], start="2024-01-02")
    return _make_price_df([400 * (1 + 0.002 * i) for i in range(200)], start="2023-10-02")


def test_web_views_fetch_vix_for_trend_vix():
    calls = []
    def provider(ticker, start, end):
        calls.append(ticker)
        return _provider(ticker, start, end)
    payload = run_backtest_view(_web_request(), provider=provider)
    assert "^VIX" in calls
    # Steady rally (momentum ~0.9% above its 10-day SMA) with calm VIX → strong market → 안전.
    assert payload["journal"]["rows"][-1]["다음모드"] == "안전"
    assert run_order_book_view(_web_request(), provider=_provider)["order_book"]["state"]["current_mode"] == "defense"


def test_schema_rejects_inverted_vix_band():
    with pytest.raises(ValueError):
        StrategySettings(mode_switch_strategy="trend_vix", vix_low=40, vix_high=30)
