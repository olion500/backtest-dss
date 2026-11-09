"""Deposits / withdrawals (CapitalParams.cash_flows, account config ``cash_flows``)."""
from datetime import date

import pandas as pd

from engines.dongpa_engine import CapitalParams, ModeParams, StrategyParams, run_backtest
from tests.conftest import _make_price_df
from web.api.app.services.accounts import load_accounts

# budget = cash / 4; closes: 2024-03-04 (Mon) .. 2024-03-08 (Fri), 2024-03-11 (Mon)
MODE = ModeParams(buy_cond_pct=3.0, tp_pct=50.0, max_hold_days=30, slices=4)
CLOSES = [100.0, 200.0, 200.0, 200.0, 200.0, 200.0]


def _run(flows=()):
    target = _make_price_df(CLOSES, start="2024-03-04")
    momo = _make_price_df([400.0] * 150, start="2023-10-02")
    params = StrategyParams(target_ticker="TEST", momentum_ticker="MOMO", defense=MODE, offense=MODE)
    return run_backtest(target, momo, params, CapitalParams(initial_cash=10_000, cash_flows=tuple(flows)))


def _row(result, day: str) -> pd.Series:
    journal = result.journal
    return journal[pd.to_datetime(journal["거래일자"]) == pd.Timestamp(day)].iloc[0]


def test_withdrawal_lands_before_the_session_and_resets_the_tranche_base():
    # 03-06: limit 200 × 1.03 = 206 ≥ 200 → buy. Without the flow: 2500 / 200 = 12 shares.
    assert _row(_run(), "2024-03-06")["매수수량"] == 12
    # Withdraw 2000 before 03-06: cash 8000, budget 2000 → 10 shares, cash 6000 after.
    day = _row(_run([(date(2024, 3, 6), -2000.0)]), "2024-03-06")
    assert day["입출금"] == -2000.0
    assert day["트렌치기반현금"] == 8000.0
    assert day["매수수량"] == 10
    assert day["현금"] == 6000.0


def test_weekend_flow_applies_to_the_next_session_and_early_flows_on_the_first():
    result = _run([(date(2024, 3, 9), 1000.0), (date(2024, 1, 1), -500.0)])
    assert _row(result, "2024-03-04")["입출금"] == -500.0
    assert _row(result, "2024-03-08")["입출금"] == 0.0
    assert _row(result, "2024-03-11")["입출금"] == 1000.0
    # The ledger before a flow replays unchanged (lots and fills stay as they were).
    later = _run([(date(2024, 3, 11), 1000.0)])
    plain = _run()
    assert later.trade_log.iloc[:3].equals(plain.trade_log.iloc[:3])
    assert _row(later, "2024-03-08")["현금"] == _row(plain, "2024-03-08")["현금"]


def test_withdrawal_is_not_a_drawdown():
    day = _row(_run([(date(2024, 3, 7), -2000.0)]), "2024-03-07")
    assert day["낙폭(DD%)"] == 0.0


def test_account_config_reads_cash_flows(tmp_path):
    (tmp_path / "a.json").write_text(
        '{"start_date": "2026-10-14", "init_cash": 5000, "cash_flows": [{"date": "2026-11-09", "amount": -700}]}'
    )
    [account] = load_accounts(tmp_path)
    assert [(flow.date, flow.amount) for flow in account.cash_flows] == [(date(2026, 11, 9), -700.0)]
    (tmp_path / "a.json").write_text('{"start_date": "2026-10-14", "init_cash": 5000}')
    assert load_accounts(tmp_path)[0].cash_flows == []


def test_flow_after_the_last_close_goes_into_todays_order_sheet():
    from engines.order_book_engine import BacktestState
    from web.api.app.schemas import CashFlow
    from web.api.app.services.backtest import _pending_cash_flows, default_request

    request = default_request().model_copy(update={
        "start_date": date(2026, 10, 14),
        "end_date": date(2026, 11, 9),
        "cash_flows": [CashFlow(date=date(2026, 11, 2), amount=-100.0), CashFlow(date=date(2026, 11, 9), amount=-734.19)],
    })
    state = BacktestState(date(2026, 11, 6), "defense", 3385.22, 8, 45.97, 300.0, 50.0)
    after = _pending_cash_flows(request, state)
    assert after.current_cash == 2651.03  # the 11-02 flow is already in the ledger
    assert after.tranche_budget == 2651.03 / request.strategy.defense.slices
    assert _pending_cash_flows(request.model_copy(update={"cash_flows": []}), state) is state
