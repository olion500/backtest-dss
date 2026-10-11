"""Weekly mode signals must only steer the session after the Friday close they include."""
import numpy as np
import pandas as pd
import pytest

from engines.dongpa_engine import run_backtest
from engines.order_book_engine import extract_state
from tests.conftest import _make_price_df


@pytest.fixture
def swinging_momo():
    """Momentum that trends up and down so the weekly RSI flips modes several times."""
    t = np.arange(400)
    prices = 400 * (1 + 0.25 * np.sin(t / 25))
    return _make_price_df(list(prices), start="2023-01-02")


@pytest.fixture
def target_prices():
    np.random.seed(3)
    prices = [50.0]
    for _ in range(399):
        prices.append(prices[-1] * (1 + np.random.normal(0.0, 0.02)))
    return _make_price_df(prices, start="2023-01-02")


def _journal(target_prices, swinging_momo, default_params, default_cap):
    start = target_prices.index[100]
    result = run_backtest(target_prices.loc[start:], swinging_momo, default_params, default_cap)
    j = result.journal.copy()
    j["d"] = pd.to_datetime(j["거래일자"])
    return j


def test_mode_switches_only_after_a_friday(target_prices, swinging_momo, default_params, default_cap):
    j = _journal(target_prices, swinging_momo, default_params, default_cap)
    switched = j["모드"] != j["모드"].shift()
    switched.iloc[0] = False
    assert switched.sum() >= 2
    prev_days = j["d"].shift()[switched]
    assert (prev_days.dt.weekday == 4).all()


def test_mode_is_previous_rows_next_mode(target_prices, swinging_momo, default_params, default_cap):
    j = _journal(target_prices, swinging_momo, default_params, default_cap)
    assert (j["모드"].iloc[1:].values == j["다음모드"].iloc[:-1].values).all()


def test_order_book_uses_next_mode(target_prices, swinging_momo, default_params, default_cap):
    j = _journal(target_prices, swinging_momo, default_params, default_cap)
    flip = j.index[j["모드"] != j["다음모드"]][0]
    journal = j.loc[:flip].drop(columns="d")
    ui_values = {"defense_slices": 7, "offense_slices": 7}
    from engines.dongpa_engine import compute_indicators
    _, ind = compute_indicators(target_prices, swinging_momo, default_params)
    state = extract_state(journal, ind, ui_values, default_cap.initial_cash)
    expected = "offense" if journal.iloc[-1]["다음모드"] == "공세" else "defense"
    assert state.current_mode == expected
