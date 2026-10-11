from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd

from tests.conftest import _make_price_df
from web.api.app import main as api_main
from web.api.app.schemas import BacktestRequest, ModeSettings, StrategySettings
from web.api.app.serializers import dataframe_payload, json_value
from web.api.app.services.accounts import load_accounts
from web.api.app.services.backtest import run_backtest_view, run_order_book_view


def _request() -> BacktestRequest:
    return BacktestRequest(
        start_date=date(2024, 1, 2),
        end_date=date(2024, 5, 20),
        initial_cash=10_000,
        strategy=StrategySettings(
            target_ticker="TEST",
            momentum_ticker="MOMO",
            defense=ModeSettings(
                slices=7,
                buy_cond_pct=3,
                tp_pct=0.2,
                max_hold_days=30,
            ),
            offense=ModeSettings(
                slices=7,
                buy_cond_pct=5,
                tp_pct=2.5,
                max_hold_days=7,
            ),
        ),
    )


def _provider(ticker: str, start: date, end: date) -> pd.DataFrame:
    periods = 100 if ticker == "TEST" else 850
    first_price = 50 if ticker == "TEST" else 400
    values = [first_price * (1 + 0.001 * index) for index in range(periods)]
    first_date = "2024-01-02" if ticker == "TEST" else "2021-04-07"
    return _make_price_df(values, start=first_date)


def test_json_value_normalizes_non_json_scalars():
    assert json_value(Decimal("12.34")) == 12.34
    assert json_value(pd.Timestamp("2024-01-02")) == "2024-01-02T00:00:00"
    assert json_value(np.float64(2.5)) == 2.5
    assert json_value(float("nan")) is None


def test_dataframe_payload_keeps_columns_and_normalizes_rows():
    payload = dataframe_payload(pd.DataFrame({"when": [pd.Timestamp("2024-01-02")], "value": [np.nan]}))
    assert payload["columns"] == ["when", "value"]
    assert payload["rows"] == [{"when": "2024-01-02T00:00:00", "value": None}]


def test_backtest_view_returns_chart_and_tables():
    payload = run_backtest_view(_request(), provider=_provider)
    assert payload["meta"]["target_ticker"] == "TEST"
    assert payload["equity"]
    assert payload["journal"]["rows"]
    assert "Final Equity" in payload["summary"]


def test_order_book_view_returns_read_only_preview():
    payload = run_order_book_view(_request(), provider=_provider)
    assert "order_book" in payload
    assert payload["order_book"]["state"]["last_date"]
    assert "orders" in payload["order_book"]


def test_order_book_view_for_account_without_a_session_yet():
    # TEST has 100 sessions from 2024-01-02, so 2024-05-21 is the session after the last close.
    request = _request().model_copy(update={"start_date": date(2024, 5, 21), "end_date": date(2024, 5, 21)})

    payload = run_order_book_view(request, provider=_provider)

    state = payload["order_book"]["state"]
    assert state["last_date"] == "2024-05-20"
    assert state["current_cash"] == 10_000
    assert state["current_position_qty"] == 0
    assert state["tranche_budget"] == 10_000 / 7
    assert payload["order_book"]["orders"]["rows"]
    assert payload["order_book"]["holdings"]["rows"] == []
    assert payload["journal"]["rows"] == [] and payload["equity"] == []


def test_health_route():
    assert api_main.health() == {"status": "ok"}


def test_backtest_route_uses_validated_payload(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "run_backtest_view",
        lambda request: {"target": request.strategy.target_ticker},
    )
    response = api_main.backtest(_request())
    assert response == {"target": "TEST"}


def test_schema_rejects_invalid_period():
    payload = _request().model_dump(mode="json")
    payload["start_date"], payload["end_date"] = payload["end_date"], payload["start_date"]
    try:
        BacktestRequest.model_validate(payload)
    except ValueError as exc:
        assert "start_date" in str(exc)
    else:
        raise AssertionError("invalid period was accepted")


def test_accounts_load_sorted_and_skip_broken_files(tmp_path):
    (tmp_path / "b_sub.json").write_text('{"name": "서브", "start_date": "2026-09-01", "init_cash": 20000}')
    (tmp_path / "a_main.json").write_text('{"name": "메인", "start_date": "2026-06-01", "init_cash": 50000, "spread_buy_step": 3}')
    (tmp_path / "broken.json").write_text('{"init_cash": -1}')

    accounts = load_accounts(tmp_path)

    assert [a.id for a in accounts] == ["a_main", "b_sub"]
    main, sub = accounts
    assert (main.name, main.initial_cash, main.spread_buy_step) == ("메인", 50000, 3)
    assert (sub.spread_buy_levels, sub.spread_buy_step) == (5, 1)
    assert sub.start_date == date(2026, 9, 1)


def test_accounts_default_to_single_account(tmp_path):
    accounts = load_accounts(tmp_path / "missing")

    assert [a.id for a in accounts] == ["main"]
    assert accounts[0].initial_cash == 10_000


def test_list_presets_puts_live_first_and_skips_invalid(tmp_path, monkeypatch):
    import json
    from web.api.app.services import backtest as svc
    live = tmp_path / "strategy.json"
    live.write_text(json.dumps({"target": "SOXL", "defense_slices": 11}))
    presets = tmp_path / "presets"
    presets.mkdir()
    (presets / "trend.json").write_text(json.dumps({"mode_switch_strategy": "trend_vix", "offense_slices": 2}))
    (presets / "broken.json").write_text(json.dumps({"rsi_low_threshold": 90, "rsi_high_threshold": 10}))
    monkeypatch.setattr(svc, "STRATEGY_PATH", live)
    monkeypatch.setattr(svc, "strategy_files", lambda: [live, presets / "trend.json", presets / "broken.json"])

    listed = svc.list_presets()

    assert [p["id"] for p in listed] == ["strategy", "trend"]
    assert listed[0]["live"] and listed[0]["strategy"]["defense"]["slices"] == 11
    assert listed[1]["strategy"]["mode_switch_strategy"] == "trend_vix"
    assert listed[1]["strategy"]["offense"]["slices"] == 2
