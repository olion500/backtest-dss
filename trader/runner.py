"""Run steps. Each step writes one run record, whatever happens, so the web can show it.

Steps (NY time):
  check  09:40  calendar, engine ledger vs broker account  (this stage: read only)
  orders 09:45  LOC sheet                                   (not built yet)
  stop   close−15m  stop-loss check → MOC                   (not built yet)

Record ``status``: ``ok`` (would trade), ``halted`` (a check failed, nothing sent),
``closed`` (no session today), ``error`` (a call failed).
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Callable

from trader.broker import make_broker
from trader.broker.base import Broker, BrokerError
from trader.calendar import Session, last_closed_session, ny_now, us_session
from trader.config import TraderAccount, TraderConfigError, load_trader_account
from trader.expected import Expected, UnknownAccount, expected_state
from trader.reconcile import DEFAULT_CASH_TOLERANCE, reconcile
from trader.runlog import LocalRunStore, RunStore

BrokerBuilder = Callable[[TraderAccount], Broker]
ExpectedBuilder = Callable[[str, date], Expected]


def _default_broker(account: TraderAccount) -> Broker:
    return make_broker(account.broker, account.credentials(), account.listings)


def _session_info(session: Session, broker_next: date | None) -> dict:
    return {
        "date": session.day.isoformat(),
        "is_open": session.is_open,
        "close": session.close.isoformat() if session.close else None,
        "early_close": session.early_close,
        "stop_check_at": session.stop_check_at.isoformat() if session.stop_check_at else None,
        "order_cutoff": session.close_order_cutoff.isoformat() if session.close_order_cutoff else None,
        "broker_next_session": broker_next.isoformat() if broker_next else None,
    }


def _expected_payload(expected: Expected) -> dict:
    data = asdict(expected)
    data["lots"] = [asdict(lot) for lot in expected.lots]
    data["orders"] = list(expected.orders)
    return data


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
    today = now.date()
    record: dict = {
        "account": account_id,
        "step": "check",
        "session": today.isoformat(),
        "started_at": now.isoformat(),
        "forced": force,
    }
    try:
        account = load_account(account_id)
        record.update(broker=account.broker, auto_trade=account.auto_trade)
        broker = broker_builder(account)

        session = us_session(today)
        broker_next = broker.next_session()
        record["calendar"] = _session_info(session, broker_next)
        broker_open = broker_next == today
        if session.is_open != broker_open:
            return _finish(record, store, "halted", f"달력이 다릅니다: 거래소 달력 {'영업일' if session.is_open else '휴장'}, 증권사 다음 영업일 {broker_next}")
        if not session.is_open and not force:
            return _finish(record, store, "closed", f"휴장 · 다음 영업일 {broker_next}")

        expected = expected_builder(account_id, today)
        record["expected"] = _expected_payload(expected)
        must_have = last_closed_session(today)
        if expected.as_of < must_have.isoformat():
            return _finish(record, store, "halted", f"시세가 늦습니다: 엔진 마지막 종가 {expected.as_of}, 필요한 종가 {must_have}")

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
            return _finish(record, store, "ok", "장부와 실제 잔고가 맞습니다")
        failed = ", ".join(f"{c.label} {c.expected} vs {c.actual}" for c in result.failures)
        return _finish(record, store, "halted", f"장부와 실제 잔고가 다릅니다: {failed}")
    except (BrokerError, TraderConfigError, UnknownAccount) as exc:
        record["error_code"] = getattr(exc, "code", None)
        return _finish(record, store, "error", str(exc))


def _finish(record: dict, store: RunStore | None, status: str, message: str) -> dict:
    record["status"] = status
    record["message"] = message
    record["finished_at"] = ny_now().isoformat()
    record["path"] = (store or LocalRunStore()).save(record["account"], _jsonable(record))
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
