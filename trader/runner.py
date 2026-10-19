"""Run steps. Each step writes one run record, whatever happens, so the web can show it.

Steps (NY time):
  check  09:40  calendar, engine ledger vs broker account  (this stage: read only)
  orders 09:45  LOC sheet                                   (not built yet)
  stop   close−15m  stop-loss check → MOC                   (not built yet)

Record ``status``: ``ok`` (would trade), ``halted`` (a check failed, nothing sent),
``closed`` (no regular session under way), ``error`` (a call failed).
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
from trader.runlog import LocalRunStore, RunStore, run_store

BrokerBuilder = Callable[[TraderAccount], Broker]
ExpectedBuilder = Callable[[str, date], Expected]


def _default_broker(account: TraderAccount) -> Broker:
    return make_broker(account.broker, account.credentials(), account.listings)


def _session_info(session: Session, broker_open: bool) -> dict:
    return {**session.to_dict(), "broker_open": broker_open}


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
        trading = session.trading_at(now)
        broker_open = broker.market_open()
        record["calendar"] = _session_info(session, broker_open)
        # Outside regular hours on a session day the broker's flag says nothing about today.
        if (trading or not session.is_open) and broker_open != trading:
            return _finish(record, store, "halted", f"달력이 다릅니다: 거래소 달력 {'장중' if trading else '휴장'}, 증권사 {'장중' if broker_open else '장 닫힘'}")
        if not trading and not force:
            if session.is_open:
                hours = f"{session.open:%H:%M}–{session.close:%H:%M}"
                return _finish(record, store, "closed", f"정규장 시간이 아닙니다 (뉴욕 {hours})")
            return _finish(record, store, "closed", "휴장")

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
