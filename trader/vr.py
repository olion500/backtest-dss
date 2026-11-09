"""VR (라오어 VR) cycle ladder → one set of broker reservations for the whole cycle.

The web VR tab's current cycle orders (built through the same service, so the trader can never
place a different ladder than the one on screen) go out once per cycle as 기간잔량 limit
reservations from the cycle's first session still ahead to its last session: the broker sends
what is still unfilled at each open. The ladder is fixed for the cycle, which is what makes
one reservation per step enough.

Sells only from what this broker account can sell: shares held elsewhere (an old account kept
for tax reasons) are never sold. The lowest sell steps go first, a step that does not fit is cut.

A VR account's trader settings live in ``config/trader/vr/<id>.json`` (``broker``,
``credentials_env``, ``auto_trade``), not next to the 동파법 accounts: ``orders --all`` must
never trade it. The broker keys come from ``<credentials_env>_APP_KEY`` / ``_APP_SECRET``.

``vr --auto`` (scheduled daily) acts only between the last close before a new cycle and that
cycle's first open: it saves the broker balance (shares, USD cash) for the cycle start once,
which the VR view adds to ``external`` to start the cycle from the real account, then places the
cycle's ladder when ``auto_trade`` is on. Any other time it records ``closed``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Callable

from trader.broker.base import BrokerError, OrderRequest, OrderType, Reservation, Side
from trader.broker.meritz import MeritzBroker
from trader.calendar import _nyse, last_closed_session, ny_now, us_session
from trader.config import DEFAULT_LISTINGS, TRADER_DIR, TraderAccount, TraderConfigError, load_trader_account
from trader.runner import _finish, _new_record
from trader.runlog import RunStore, run_store

VR_TRADER_DIR = TRADER_DIR / "vr"

Approver = Callable[["VRPlan", list[OrderRequest]], bool]


@dataclass(frozen=True)
class VRPlan:
    vr_id: str
    symbol: str
    cycle: int
    cycle_start: date
    cycle_end: date
    close_date: date          # the close the ladder was built from
    v: float
    pool: float
    first_session: date       # reservations run from here …
    last_session: date        # … to here
    orders: tuple[OrderRequest, ...]
    buy_cost: Decimal         # every buy step filled
    sellable: int
    sells_cut: int            # sell shares left out: more than the account can sell


def reservation_window(cycle_start: date, cycle_end: date, now: datetime) -> tuple[date, date] | None:
    """First and last session to reserve: from the cycle start, or the next session not yet opened."""
    calendar = _nyse()
    first = max(cycle_start, now.date())
    session = us_session(first)
    if session.is_open and now >= session.open:
        first += timedelta(days=1)
    first = calendar.date_to_session(first.isoformat(), direction="next").date()
    last = calendar.date_to_session(cycle_end.isoformat(), direction="previous").date()
    return (first, last) if first < last else None


def build_plan(vr_id: str, view: dict, sellable: int, window: tuple[date, date]) -> VRPlan:
    """Pure: the cycle's unfilled ladder steps as limit orders, sells cut to ``sellable``."""
    symbol = view["account"]["ticker"]
    current = view["current"]
    orders: list[OrderRequest] = []
    buy_cost = Decimal(0)
    left, cut = sellable, 0
    pending = [row for row in view["orders"] if not row.get("filled_on")]
    for row in sorted((r for r in pending if r["side"] == "buy"), key=lambda r: -r["price"]):
        price = Decimal(str(row["price"]))
        orders.append(OrderRequest(symbol, Side.BUY, OrderType.LIMIT, int(row["qty"]), price))
        buy_cost += price * int(row["qty"])
    for row in sorted((r for r in pending if r["side"] == "sell"), key=lambda r: r["price"]):
        qty = min(int(row["qty"]), left)
        cut += int(row["qty"]) - qty
        if qty:
            orders.append(OrderRequest(symbol, Side.SELL, OrderType.LIMIT, qty, Decimal(str(row["price"]))))
            left -= qty
    return VRPlan(
        vr_id=vr_id,
        symbol=symbol,
        cycle=int(current["index"]),
        cycle_start=date.fromisoformat(str(current["start"])),
        cycle_end=date.fromisoformat(str(current["end"])),
        close_date=date.fromisoformat(str(current["close_date"])),
        v=float(current["v"]),
        pool=float(current["pool_start"]),
        first_session=window[0],
        last_session=window[1],
        orders=tuple(orders),
        buy_cost=buy_cost,
        sellable=sellable,
        sells_cut=cut,
    )


def split_existing(plan: VRPlan, existing: list[Reservation]) -> tuple[list[OrderRequest], list[Reservation]]:
    """Pure: the plan's orders not reserved yet, and reservations that match no plan step.

    A step counts as reserved by a live reservation with the same side, quantity and price,
    so a run stopped halfway (or a one-order trial) picks up where it left off.
    """
    left = list(existing)
    todo: list[OrderRequest] = []
    for order in plan.orders:
        match = next((r for r in left if r.side is order.side and r.qty == order.qty and r.price == order.price), None)
        if match is None:
            todo.append(order)
        else:
            left.remove(match)
    return todo, left


def _vr_account(vr_id: str):
    from web.api.app.services.vr import load_vr_account_models

    for account in load_vr_account_models():
        if account.id == vr_id:
            return account
    raise BrokerError(f"config/vr/{vr_id}.json 이 없습니다", code="NO_VR_ACCOUNT")


def vr_view(vr_id: str, today: date, balances: dict | None = None) -> dict:
    from web.api.app.services.vr import run_vr_view

    return run_vr_view(_vr_account(vr_id), today=today, balances=balances)


def vr_trader_ids(trader_dir=VR_TRADER_DIR) -> list[str]:
    return sorted(path.stem for path in trader_dir.glob("*.json")) if trader_dir.is_dir() else []


def load_vr_trader(vr_id: str, trader_dir=VR_TRADER_DIR) -> TraderAccount:
    return load_trader_account(vr_id, trader_dir)


def first_unopened_session(now: datetime) -> date:
    """The next session whose open is still ahead of ``now``."""
    today = now.date()
    session = us_session(today)
    if session.is_open and now < session.open:
        return today
    return _nyse().date_to_session((today + timedelta(days=1)).isoformat(), direction="next").date()


def cycle_of(start_date: date, day: date) -> tuple[date, date]:
    """The cycle (start, end) of a VR account starting ``start_date`` that holds ``day``."""
    from engines.vr_engine import CYCLE_DAYS

    start = start_date + timedelta(days=(day - start_date).days // CYCLE_DAYS * CYCLE_DAYS)
    return start, start + timedelta(days=CYCLE_DAYS - 1)


def meritz_broker(credentials_env: str) -> MeritzBroker:
    prefix = credentials_env.upper().rstrip("_")
    credentials = {key: os.environ.get(f"{prefix}_{key}", "") for key in MeritzBroker.required_credentials}
    return MeritzBroker(credentials, DEFAULT_LISTINGS)


def run_vr_reserve(
    vr_id: str,
    credentials_env: str,
    approve: Approver,
    limit: int | None = None,
    now: datetime | None = None,
    broker=None,
    view_builder: Callable[[str, date], dict] = vr_view,
    store: RunStore | None = None,
) -> dict:
    """Show the cycle's reservations and place the missing ones once ``approve`` says yes.

    Steps already reserved are skipped; a live reservation that matches no step halts the run
    (the ladder changed, or someone placed it by hand). ``limit`` sends only the first N.
    """
    now = now or ny_now()
    record = _new_record(f"vr-{vr_id}", "vr-reserve", now, False)
    record["credentials_env"] = credentials_env
    try:
        broker = broker or meritz_broker(credentials_env)
        view = view_builder(vr_id, now.date())
        return _place(record, store, broker, view, vr_id, now, approve, limit)
    except BrokerError as exc:
        record["error_code"] = exc.code
        return _finish(record, store, "error", str(exc))


def run_vr_auto(
    vr_id: str,
    now: datetime | None = None,
    broker=None,
    settings: TraderAccount | None = None,
    view_builder: Callable[..., dict] = vr_view,
    store: RunStore | None = None,
) -> dict:
    """Scheduled: before a new cycle's first open, save the broker balance once and place the ladder.

    The balance (shares in the broker account, estimated USD cash) is the cycle start's
    actual state together with ``external``; it is kept for good (run store state) and never
    replaced, so the cycle's ladder stays the one placed.
    """
    from web.api.app.services.vr import ACTUALS, record_id

    now = now or ny_now()
    store = store or run_store()
    record = _new_record(record_id(vr_id), "vr-auto", now, False)
    try:
        settings = settings or load_vr_trader(vr_id)
        record["auto_trade"] = settings.auto_trade
        account = _vr_account(vr_id)
        first = first_unopened_session(now)
        close_day = last_closed_session(first)
        if now < us_session(close_day).close:
            return _finish(record, store, "closed", f"장중입니다 ({close_day} 마감 전)")
        start, end = cycle_of(account.start_date, first)
        cycle_first = _nyse().date_to_session(start.isoformat(), direction="next").date()
        record["cycle_window"] = {"start": start, "end": end, "first_session": first}
        if first != cycle_first:
            return _finish(record, store, "closed", f"사이클 중간입니다 ({start} ~ {end}): 새 사이클 첫 거래일 전에만 예약합니다")

        broker = broker or meritz_broker(settings.env_prefix)
        state = store.read_state(record_id(vr_id), ACTUALS)
        balances = dict(state.get("cycles", {}))
        key = start.isoformat()
        if key not in balances:
            position = next((p for p in broker.positions() if p.symbol == account.ticker), None)
            cash = broker.cash("USD")
            balances[key] = {
                "shares": int(position.qty) if position else 0,
                "cash": str(cash.estimated),
                "close_date": close_day.isoformat(),
                "taken_at": now.isoformat(),
            }
            store.write_state(record_id(vr_id), ACTUALS, {**state, "cycles": balances})
            record["balance_saved"] = True
        record["balance"] = balances[key]
        external = account.external_at(start)
        record["external"] = external.model_dump(mode="json", by_alias=True) if external else None

        view = view_builder(vr_id, first, balances)
        if str(view["current"]["start"]) != key:
            return _finish(record, store, "halted", f"웹 VR 계산의 사이클({view['current']['start']})이 {key}가 아닙니다")
        return _place(record, store, broker, view, vr_id, now, None if settings.auto_trade else _decline, None)
    except TraderConfigError as exc:
        return _finish(record, store, "error", str(exc))
    except BrokerError as exc:
        record["error_code"] = exc.code
        return _finish(record, store, "error", str(exc))


def _decline(plan: VRPlan, todo: list[OrderRequest]) -> bool:
    return False


def _place(record: dict, store, broker, view: dict, vr_id: str, now: datetime, approve: Approver | None, limit: int | None) -> dict:
    """The cycle's missing reservations, once ``approve`` says yes (``None``: no question, auto)."""
    current = view["current"]
    start, end = date.fromisoformat(str(current["start"])), date.fromisoformat(str(current["end"]))
    record["cycle"] = {key: current.get(key) for key in ("index", "start", "end", "v", "v_min", "v_max", "shares_start", "pool_start", "close_date", "actual")}
    wanted_close = last_closed_session(start)
    if date.fromisoformat(str(current["close_date"])) < wanted_close:
        return _finish(record, store, "halted", f"사이클 직전 종가({wanted_close})가 아직 없습니다 (마지막 {current['close_date']})")
    window = reservation_window(start, end, now)
    if window is None:
        return _finish(record, store, "halted", f"이 사이클({start} ~ {end})에 예약할 거래일이 2일 이상 남지 않았습니다")

    position = next((p for p in broker.positions() if p.symbol == view["account"]["ticker"]), None)
    sellable = int(position.sellable_qty) if position else 0
    plan = build_plan(vr_id, view, sellable, window)
    record["plan"] = {**asdict(plan), "orders": [asdict(order) for order in plan.orders]}
    cash = broker.cash("USD")
    record["cash"] = asdict(cash)
    if plan.buy_cost > cash.estimated:
        # Part of the pool may be held elsewhere: buys need it in this account by the time they fill.
        record["cash_short"] = str(plan.buy_cost - cash.estimated)

    existing = [r for r in broker.reservations(start - timedelta(days=14), now.date())
                if r.symbol == plan.symbol and not r.cancelled and r.end is not None and r.end >= plan.first_session]
    record["existing"] = [asdict(r) for r in existing]
    todo, strangers = split_existing(plan, existing)
    if strangers:
        listed = ", ".join(f"{r.reservation_id} {r.side.value} {r.qty.normalize():f}주 ${r.price}" for r in strangers)
        return _finish(record, store, "halted", f"주문표에 없는 예약이 {len(strangers)}건 있어 보내지 않았습니다: {listed}")
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        return _finish(record, store, "ok", f"보낼 예약이 없습니다 (이미 {len(existing)}건 예약됨)")
    if approve is not None and not approve(plan, todo):
        reason = "auto_trade가 꺼져 있어" if approve is _decline else "승인하지 않아"
        return _finish(record, store, "declined", f"{reason} 보내지 않았습니다 ({len(todo)}건)")
    record["approved_at"] = ny_now().isoformat()

    sent: list[dict] = []
    record["sent"] = sent
    for request in todo:
        try:
            number = broker.reserve(request, plan.first_session, plan.last_session)
        except BrokerError as exc:
            sent.append({"request": asdict(request), "error": str(exc), "error_code": exc.code})
            record["error_code"] = exc.code
            return _finish(record, store, "partial", f"{len(sent) - 1}/{len(todo)}건 예약 후 멈춤: {exc}")
        sent.append({"request": asdict(request), "reservation_id": number})
    return _finish(record, store, "sent", f"{len(sent)}건 모두 예약됐습니다 ({plan.first_session} ~ {plan.last_session})")

