from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

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
    # TEST has 100 sessions from 2024-01-02 (through 2024-05-20): one point and one row each
    assert len(payload["equity"]) == len(payload["journal"]["rows"]) == 100
    first, last = payload["equity"][0], payload["equity"][-1]
    assert first == {"date": "2024-01-02T00:00:00", "equity": 10_000.0, "price": 50.0}
    assert last["price"] == 50 * (1 + 0.001 * 99)
    assert last["equity"] == payload["journal"]["rows"][-1]["Equity"]
    assert payload["summary"]["Final Equity"] == last["equity"]
    assert payload["summary"]["Strategy Return"] == (last["equity"] / 10_000 - 1) * 100
    assert payload["summary"]["Target Hold Return"] == ((50 * 1.099) / 50 - 1) * 100
    assert payload["state"]["cash_end"] == payload["journal"]["rows"][-1]["현금"]


def test_order_book_view_returns_read_only_preview():
    payload = run_order_book_view(_request(), provider=_provider)
    book = payload["order_book"]
    last = payload["journal"]["rows"][-1]
    state = book["state"]
    assert state["last_date"] == "2024-05-20"
    assert state["prev_close"] == last["종가"]
    assert state["current_cash"] == last["현금"]
    mode = "offense" if last["다음모드"] == "공세" else "defense"
    assert state["current_mode"] == mode
    assert state["tranche_budget"] == last["트렌치기반현금"] / 7
    prices = [row["주문가"] for row in book["orders"]["rows"]]
    assert prices and prices == sorted(prices, reverse=True)
    # The base buy (limit = close × 1.03, rounded to the cent like the backtest) nets fully
    # against a cheaper TP sell, which moves one cent above the limit as a TP sell.
    assert state["current_mode"] == "defense"
    ceiling = [row for row in book["orders"]["rows"] if row["구분"] == "매도 (익절)" and "퉁치기로 옮긴" in row["비고"]]
    assert len(ceiling) == 1
    assert last["종가"] == 54.95  # 54.95 × 1.03 = 56.5985 → limit 56.60
    assert ceiling[0]["주문가"] == pytest.approx(56.61)
    assert book["netting_message"].startswith("퉁치기 적용")
    # Holdings and TP sells come from the open lots only: three 18-share lots, and the cheapest
    # (TP 54.96, below the 56.60 buy limit) is consumed by netting, so two TP sells remain.
    open_lots = [t for t in payload["trade_log"]["rows"] if t["상태"] != "완료"]
    assert len(book["holdings"]["rows"]) == len(open_lots) == state["current_position_qty"] // 18
    assert sorted(r["주문가"] for r in book["orders"]["rows"] if r["구분"] == "매도 (익절)" and r is not ceiling[0]) == sorted(
        t["TP목표가"] for t in open_lots if t["TP목표가"] > 54.96
    )


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


def test_strategy_from_raw_defaults_match_the_recommended_strategy():
    from web.api.app.services.backtest import strategy_from_raw
    # CLAUDE.md: defense 7 slices / 3% / TP 0.2% / 30 days, offense 7 / 5% / 2.5% / 7 days, no SL
    assert strategy_from_raw({}) == StrategySettings()
    raw = {"target": "TQQQ", "defense_buy": 2.5, "offense_hold": 9, "defense_sl": 4, "enable_netting": False}
    strategy = strategy_from_raw(raw)
    assert strategy.target_ticker == "TQQQ"
    assert strategy.defense.buy_cond_pct == 2.5 and strategy.defense.stop_loss_pct == 4
    assert strategy.offense.max_hold_days == 9
    assert strategy.enable_netting is False


def test_default_request_uses_first_account(monkeypatch):
    from web.api.app.services import backtest as svc
    from web.api.app.services.accounts import Account
    monkeypatch.setattr(svc, "market_today", lambda: date(2026, 10, 15))
    monkeypatch.setattr(svc, "load_strategy", lambda: {"target": "TQQQ"})
    monkeypatch.setattr(svc, "load_accounts", lambda: [])
    request = svc.default_request()
    assert (request.start_date, request.end_date) == (date(2022, 1, 1), date(2026, 10, 15))
    assert (request.initial_cash, request.spread_buy_levels, request.spread_buy_step) == (10_000, 5, 1)
    assert request.strategy.target_ticker == "TQQQ"
    account = Account(id="a", name="A", start_date=date(2026, 6, 1), initial_cash=5000,
                      spread_buy_levels=3, spread_buy_step=2)
    monkeypatch.setattr(svc, "load_accounts", lambda: [account])
    request = svc.default_request()
    assert request.start_date == date(2026, 6, 1)
    assert (request.initial_cash, request.spread_buy_levels, request.spread_buy_step) == (5000, 3, 2)


def test_backtest_view_fetches_warmup_and_reports_momentum_period():
    calls = []

    def provider(ticker, start, end):
        calls.append((ticker, start, end))
        return _provider(ticker, start, end)

    payload = run_backtest_view(_request(), provider=provider)
    # target over the request (end exclusive), momentum with 1000 days of warm-up
    assert calls == [
        ("TEST", date(2024, 1, 2), date(2024, 5, 21)),
        ("MOMO", date(2021, 4, 7), date(2024, 5, 21)),
    ]
    momo = _provider("MOMO", None, None)["Close"].loc["2024-01-02":"2024-05-20"]
    assert payload["summary"]["Momentum Hold Return"] == (momo.iloc[-1] / momo.iloc[0] - 1) * 100


def test_order_book_view_seeds_from_the_two_sessions_before_start():
    calls = []

    def provider(ticker, start, end):
        calls.append((ticker, start, end))
        return _provider(ticker, start, end)

    request = _request().model_copy(update={"start_date": date(2024, 5, 21), "end_date": date(2024, 5, 21)})
    run_order_book_view(request, provider=provider)
    # session lookup 14 days back, then a run over the last two sessions (May 17 and May 20)
    assert calls[0] == ("TEST", date(2024, 5, 7), date(2024, 5, 22))
    assert calls[1] == ("TEST", date(2024, 5, 17), date(2024, 5, 21))


def test_benchmark_summary_reads_the_committed_results(tmp_path):
    from web.api.app.services.benchmarks import SUMMARY_PATH, load_benchmark_summary

    summary = load_benchmark_summary()
    assert summary is not None and SUMMARY_PATH.name == "summary.json"
    ids = {s["id"] for s in summary["strategies"]}
    for window in summary["windows"]:
        assert set(window["curves"]) == ids == set(window["stats"])
        assert all(len(values) == len(window["dates"]) for values in window["curves"].values())
        assert all(values[0] == 1.0 for values in window["curves"].values())
    assert load_benchmark_summary(tmp_path / "missing.json") is None


def test_benchmark_summary_endpoint_404_without_results(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(api_main, "load_benchmark_summary", lambda: None)
    try:
        api_main.benchmark_summary()
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("expected 404")
