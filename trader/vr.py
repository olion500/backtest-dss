"""VR (라오어 VR) cycle ladder → one set of broker reservations for the whole cycle.

The web VR tab's current cycle orders (built through the same service, so the trader can never
place a different ladder than the one on screen) go out once per cycle as 기간잔량 limit
reservations from the cycle's first session still ahead to its last session: the broker sends
what is still unfilled at each open. The ladder is fixed for the cycle, which is what makes
one reservation per step enough.

Sells only from what this broker account can sell: shares held elsewhere (an old account kept
for tax reasons) are never sold. The lowest sell steps go first, a step that does not fit is cut.

There is no ``config/trader`` file for a VR account on purpose: ``orders --all`` must never
trade it as a 동파법 account. The broker keys come from ``<credentials_env>_APP_KEY`` / ``_APP_SECRET``.
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
from trader.config import DEFAULT_LISTINGS
from trader.runner import _finish, _new_record
from trader.runlog import RunStore

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


def vr_view(vr_id: str, today: date) -> dict:
    from web.api.app.services.vr import load_vr_account_models, run_vr_view

    for account in load_vr_account_models():
        if account.id == vr_id:
            return run_vr_view(account, today=today)
    raise BrokerError(f"config/vr/{vr_id}.json 이 없습니다", code="NO_VR_ACCOUNT")


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
        current = view["current"]
        start, end = date.fromisoformat(str(current["start"])), date.fromisoformat(str(current["end"]))
        record["cycle"] = {key: current[key] for key in ("index", "start", "end", "v", "v_min", "v_max", "shares_start", "pool_start", "close_date")}
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
        if not approve(plan, todo):
            return _finish(record, store, "declined", "승인하지 않아 보내지 않았습니다")
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
    except BrokerError as exc:
        record["error_code"] = exc.code
        return _finish(record, store, "error", str(exc))

