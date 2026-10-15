"""benchmarks/ harness: the injected mode series must reproduce the engine (no data download)."""
import numpy as np
import pandas as pd
import pytest

import engines.dongpa_engine as E
from benchmarks.__main__ import suites
from benchmarks.lib import data
from benchmarks.lib.engine import mp, run
from benchmarks.lib.signals import rsi_mode
from benchmarks.lib.windows import metrics
from tests.conftest import _make_price_df


def _walk(seed, n, vol):
    rng = np.random.default_rng(seed)
    return list(100 * np.cumprod(1 + rng.normal(0.0005, vol, n)))


@pytest.fixture
def market(monkeypatch):
    monkeypatch.setitem(data._cache, 'TGT', _make_price_df(_walk(1, 600, 0.04), start='2020-01-01'))
    monkeypatch.setitem(data._cache, 'QQQ', _make_price_df(_walk(2, 700, 0.015), start='2019-07-01'))


def test_injected_rsi_series_matches_engine_rsi(market):
    D, O = mp(7, 3.0, 0.2, 30), mp(7, 5.0, 2.5, 7)
    idx = data.load('TGT').index
    native = run('TGT', {}, D, O, idx[150], idx[-1], scale=False)
    injected = run('TGT', rsi_mode(idx), D, O, idx[150], idx[-1], scale=False)
    assert set(native.journal['모드']) == {'안전', '공세'}  # the window really switches modes
    pd.testing.assert_series_equal(native.equity, injected.equity)


def test_injection_is_undone_after_run(market):
    idx = data.load('TGT').index
    decide = E.decide_mode
    run('TGT', pd.Series('offense', index=idx), mp(7, 3.0, 0.2, 30), mp(7, 5.0, 2.5, 7), idx[0], idx[-1], scale=False)
    assert E.decide_mode is decide


def test_always_offense_series_trades_offense_params(market):
    idx = data.load('TGT').index
    r = run('TGT', pd.Series('offense', index=idx), mp(7, 3.0, 0.2, 30), mp(7, 5.0, 2.5, 7), idx[0], idx[-1], scale=False)
    assert set(r.journal['모드']) == {'공세'}


def test_metrics_are_cagr_and_negative_mdd():
    eq = pd.Series([100.0, 50.0, 200.0], index=pd.to_datetime(['2020-01-01', '2020-07-01', '2021-01-01']))
    cagr, mdd = metrics(eq)
    assert mdd == pytest.approx(-0.5)
    assert cagr == pytest.approx(1.0, rel=0.01)


def test_every_suite_has_a_description():
    listed = suites()
    assert {'verify_harness', 'search', 'mix'} <= set(listed)
    assert all(listed.values())
