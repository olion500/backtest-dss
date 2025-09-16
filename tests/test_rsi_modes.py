import pytest

import backtest_local


def test_compute_wilder_rsi_bounds():
    prices = [100.0, 102.0, 101.0, 103.0, 104.0]
    rsi = backtest_local._compute_wilder_rsi(prices, period=2)
    assert len(rsi) == len(prices)
    assert rsi[0] is None and rsi[1] is None
    for value in rsi[2:]:
        assert value is not None
        assert 0.0 <= value <= 100.0


def test_entry_flag_derivation_follows_rules():
    rsi = [
        None,
        68.0,
        64.0,
        52.0,
        49.0,
        47.0,
        34.0,
        36.0,
        52.0,
        55.0,
        40.0,
    ]
    safe_cfg = backtest_local._parse_safe_config(
        {"falling_above": 65, "falling_between": [40, 50], "cross_below": 50}
    )
    aggressive_cfg = backtest_local._parse_aggressive_config(
        {"rising_below": 35, "rising_between": [50, 60], "cross_above": 50}
    )
    safe_flags, aggressive_flags = backtest_local._derive_entry_flags(rsi, safe_cfg, aggressive_cfg)
    safe_indices = [index for index, allowed in enumerate(safe_flags) if allowed]
    aggressive_indices = [index for index, allowed in enumerate(aggressive_flags) if allowed]
    assert safe_indices == [2, 4, 5, 6, 10]
    assert aggressive_indices == [7, 8, 9]


def test_build_entry_gates_uses_mode_rules(monkeypatch):
    prices = [100.0, 99.0, 98.0]
    fake_rsi = [None, 68.0, 64.0]

    def fake_compute(prices_arg, period):
        assert prices_arg == prices
        assert period == 14
        return fake_rsi

    monkeypatch.setattr(backtest_local, "_compute_wilder_rsi", fake_compute)
    mode_rules = {
        "indicator": "wilder_rsi",
        "period": 14,
        "safe": {"falling_above": 65},
        "aggressive": {"rising_below": 30},
    }

    gates = backtest_local._build_entry_gates(prices, mode_rules)
    assert list(gates.safe) == [False, False, True]
    assert list(gates.aggressive) == [False, False, False]


def test_build_entry_gates_returns_none_without_rules():
    prices = [100.0, 101.0, 102.0]
    assert backtest_local._build_entry_gates(prices, None) is None
