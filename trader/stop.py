"""Same-day stop loss. Pure: picks the lots and the orders to cancel, no I/O.

The backtest sells a lot at the close when ``close <= SL``; no LOC sell can say that,
so the stop step looks at the price just before the close instead. A lot past its
stop gets its take-profit LOC cancelled (the broker counts shares under an open sell
as not sellable) and the shares go out as one MOC sell.

A stop lot's TP is its buy price plus the TP %, far above a price that is already at
its stop, so netting (which only moves sells below the day's buy limit) never touches
it: the TP order is found by exact price and quantity. Anything else stops the step.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from trader.broker.base import Order, OrderState, OrderType, OrderRequest, Side
from trader.expected import Expected, Lot

CENT = Decimal("0.01")


class StopError(ValueError):
    pass


@dataclass(frozen=True)
class StopPlan:
    lots: tuple[Lot, ...]       # lots at or below their stop
    cancel: tuple[Order, ...]   # their TP LOC sells
    sell: OrderRequest | None   # one MOC for all their shares (None: nothing to do)


def _cents(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def stop_lots(expected: Expected, last: Decimal) -> tuple[Lot, ...]:
    return tuple(lot for lot in expected.lots if lot.sl is not None and lot.sl > 0 and last <= lot.sl)


def plan_stop(expected: Expected, last: Decimal, orders_today: list[Order]) -> StopPlan:
    lots = stop_lots(expected, last)
    if not lots:
        return StopPlan((), (), None)
    open_sells = [
        o for o in orders_today
        if o.symbol == expected.symbol and o.side is Side.SELL and o.state is OrderState.OPEN
    ]
    # An expiring lot already sells at the close (MOC from the sheet): nothing to add.
    moc_left = Counter(int(o.qty) for o in open_sells if o.type is OrderType.MOC)
    loc_sells = [o for o in open_sells if o.type is OrderType.LOC]
    selling: list[Lot] = []
    cancel: list[Order] = []
    for lot in lots:
        if moc_left[lot.qty] > 0:
            moc_left[lot.qty] -= 1
            continue
        if lot.tp is None:
            raise StopError(f"{lot.buy_date} {lot.qty}주: 익절가가 없어 주문을 찾을 수 없습니다")
        tp = _cents(lot.tp)
        match = next((o for o in loc_sells if o.price is not None and _cents(o.price) == tp and int(o.qty) == lot.qty), None)
        if match is None:
            raise StopError(f"{lot.buy_date} {lot.qty}주: 익절 LOC ${tp} {lot.qty}주 주문이 미체결 목록에 없습니다")
        loc_sells.remove(match)
        cancel.append(match)
        selling.append(lot)
    qty = sum(lot.qty for lot in selling)
    sell = OrderRequest(expected.symbol, Side.SELL, OrderType.MOC, qty) if qty else None
    return StopPlan(lots, tuple(cancel), sell)
