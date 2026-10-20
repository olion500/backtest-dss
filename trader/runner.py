"""Run steps. Each step writes one run record, whatever happens, so the web can show it.

Steps (NY time):
  check  09:40  calendar, engine ledger vs broker account  (read only)
  orders 09:45  check again, then the order sheet as LOC/MOC once a person approves it,
                or unattended (``--auto``) when the account's ``auto_trade`` is on
  stop   close−15m  stop-loss check → MOC                   (not built yet)

Record ``status``: ``ok`` (would trade), ``halted`` (a check failed, nothing sent),
``closed`` (no regular session under way), ``error`` (a call failed).
The orders step adds ``declined`` (not approved or ``auto_trade`` off, nothing sent), ``sent`` (every order
taken) and ``partial`` (stopped at the first order the broker did not take).
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Callable

from trader.broker import make_broker
from trader.broker.base import Broker, BrokerError, OrderRequest, OrderState
from trader.calendar import Session, last_closed_session, ny_now, us_session
from trader.config import TraderAccount, TraderConfigError, load_trader_account
from trader.expected import Expected, UnknownAccount, expected_state
from trader.plan import PlanError, plan_orders
from trader.reconcile import DEFAULT_CASH_TOLERANCE, reconcile
from trader.runlog import LocalRunStore, RunStore, run_store

BrokerBuilder = Callable[[TraderAccount], Broker]
ExpectedBuilder = Callable[[str, date], Expected]
Approver = Callable[[list[OrderRequest], Expected], bool]


def _default_broker(account: TraderAccount) -> Broker:
    return make_broker(account.broker, account.credentials(), account.listings)


def _session_info(session: Session, broker_open: bool) -> dict:
    return {**session.to_dict(), "broker_open": broker_open}


def _expected_payload(expected: Expected) -> dict:
    data = asdict(expected)
    data["lots"] = [asdict(lot) for lot in expected.lots]
    data["orders"] = list(expected.orders)
    return data


def _new_record(account_id: str, step: str, now: datetime, force: bool) -> dict:
    return {
        "account": account_id,
        "step": step,
        "session": now.date().isoformat(),
        "started_at": now.isoformat(),
        "forced": force,
    }


def _check(
    record: dict, account_id: str, now: datetime, force: bool, broker: Broker, expected_builder: ExpectedBuilder,
) -> tuple[str, str, Session, Expected | None]:
    """Calendar, then the engine ledger against the account. Fills ``record``; returns the outcome."""
    today = now.date()
    session = us_session(today)
    trading = session.trading_at(now)
    broker_open = broker.market_open()
    record["calendar"] = _session_info(session, broker_open)
    # Outside regular hours on a session day the broker's flag says nothing about today.
    if (trading or not session.is_open) and broker_open != trading:
        return "halted", f"달력이 다릅니다: 거래소 달력 {'장중' if trading else '휴장'}, 증권사 {'장중' if broker_open else '장 닫힘'}", session, None
    if not trading and not force:
        if session.is_open:
            return "closed", f"정규장 시간이 아닙니다 (뉴욕 {session.open:%H:%M}–{session.close:%H:%M})", session, None
        return "closed", "휴장", session, None

    expected = expected_builder(account_id, today)
    record["expected"] = _expected_payload(expected)
    must_have = last_closed_session(today)
    if expected.as_of < must_have.isoformat():
        return "halted", f"시세가 늦습니다: 엔진 마지막 종가 {expected.as_of}, 필요한 종가 {must_have}", session, expected

    positions = broker.positions()
    cash = broker.cash("USD")
    orders = broker.orders_today()
    record["actual"] = {
        "positions": [asdict(p) for p in positions],
        "cash": asdict(cash),
        "orders_today": [asdict(o) for o in orders],
    }
    result = reconcile(expected, positions, cash, orders, cash_tolerance=DEFAULT_CASH_TOLERANCE)
    record["checks"] = [asdict(check) for check in result.checks]
    if result.ok:
        return "ok", "장부와 실제 잔고가 맞습니다", session, expected
    failed = ", ".join(f"{c.label} {c.expected} vs {c.actual}" for c in result.failures)
    return "halted", f"장부와 실제 잔고가 다릅니다: {failed}", session, expected


def run_check(
    account_id: str,
    now: datetime | None = None,
    force: bool = False,
    broker_builder: BrokerBuilder = _default_broker,
    expected_builder: ExpectedBuilder = expected_state,
    load_account: Callable[[str], TraderAccount] = load_trader_account,
    store: RunStore | None = None,
) -> dict:
    now = now or ny_now()
    record = _new_record(account_id, "check", now, force)
    try:
        account = load_account(account_id)
        record.update(broker=account.broker, auto_trade=account.auto_trade)
        broker = broker_builder(account)
        status, message, _, _ = _check(record, account_id, now, force, broker, expected_builder)
        return _finish(record, store, status, message)
    except (BrokerError, TraderConfigError, UnknownAccount) as exc:
        record["error_code"] = getattr(exc, "code", None)
        return _finish(record, store, "error", str(exc))


def run_orders(
    account_id: str,
    approve: Approver | None = None,
    now: datetime | None = None,
    clock: Callable[[], datetime] = ny_now,
    broker_builder: BrokerBuilder = _default_broker,
    expected_builder: ExpectedBuilder = expected_state,
    load_account: Callable[[str], TraderAccount] = load_trader_account,
    store: RunStore | None = None,
) -> dict:
    """Check, show the sheet, send it once ``approve`` says yes. Nothing goes out on a day the check fails.

    Without ``approve`` the run is unattended: the account's ``auto_trade`` setting is the approval.
    """
    now = now or clock()
    record = _new_record(account_id, "orders", now, False)
    try:
        account = load_account(account_id)
        record.update(broker=account.broker, auto_trade=account.auto_trade)
        broker = broker_builder(account)
        status, message, session, expected = _check(record, account_id, now, False, broker, expected_builder)
        if status != "ok":
            return _finish(record, store, status, message)
        cutoff = session.close_order_cutoff
        if now >= cutoff:
            return _finish(record, store, "halted", f"주문 마감이 지났습니다 (뉴욕 {cutoff:%H:%M})")
        try:
            plan = plan_orders(expected)
        except PlanError as exc:
            return _finish(record, store, "halted", f"주문안을 만들 수 없습니다: {exc}")
        record["plan"] = [asdict(order) for order in plan]
        if not plan:
            return _finish(record, store, "ok", "보낼 주문이 없습니다")

        if approve is None:
            if not account.auto_trade:
                return _finish(record, store, "declined", "auto_trade가 꺼져 있어 보내지 않았습니다")
            record["approved_by"] = "auto_trade"
        elif approve(plan, expected):
            record["approved_by"] = "person"
        else:
            return _finish(record, store, "declined", "승인하지 않아 보내지 않았습니다")
        approved_at = clock()
        record["approved_at"] = approved_at.isoformat()
        # Approval can take a while: the cutoff and "nothing sent yet" must still hold now.
        if approved_at >= cutoff:
            return _finish(record, store, "halted", f"승인 사이에 주문 마감이 지났습니다 (뉴욕 {cutoff:%H:%M})")
        pending = [o for o in broker.orders_today() if o.symbol == expected.symbol and o.state in (OrderState.OPEN, OrderState.PARTIAL)]
        if pending:
            return _finish(record, store, "halted", f"승인 사이에 미체결 주문 {len(pending)}건이 생겨 보내지 않았습니다")

        sent: list[dict] = []
        record["sent"] = sent
        for request in plan:
            try:
                order = broker.submit(request)
            except BrokerError as exc:
                # Stop at the first miss: the rest of the sheet assumes the earlier rows are in.
                sent.append({"request": asdict(request), "error": str(exc), "error_code": exc.code})
                record["error_code"] = exc.code
                return _finish(record, store, "partial", f"{len(sent) - 1}/{len(plan)}건 접수 후 멈춤: {exc}")
            sent.append({"request": asdict(request), "order_id": order.order_id})
        return _finish(record, store, "sent", f"{len(plan)}건 모두 접수됐습니다")
    except (BrokerError, TraderConfigError, UnknownAccount) as exc:
        record["error_code"] = getattr(exc, "code", None)
        return _finish(record, store, "error", str(exc))


def _finish(record: dict, store: RunStore | None, status: str, message: str) -> dict:
    record["status"] = status
    record["message"] = message
    record["finished_at"] = ny_now().isoformat()
    try:
        record["path"] = (store or run_store()).save(record["account"], _jsonable(record))
    except Exception as exc:  # e.g. GCS unreachable: the run must still leave a record and not stop --all
        record["store_error"] = f"{type(exc).__name__}: {exc}"
        record["path"] = LocalRunStore().save(record["account"], _jsonable(record))
    return record


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value
