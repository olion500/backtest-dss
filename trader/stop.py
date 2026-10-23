"""Same-day stop loss. Pure: picks the lots and the orders to cancel, no I/O.

The backtest sells a lot at the close when ``close <= SL``; no LOC sell can say that,
so the stop step looks at the price just before the close instead. A lot past its
stop gets its take-profit LOC cancelled (the broker counts shares under an open sell
as not sellable) and the shares go out as one MOC sell.

A stop lot's TP is its buy price plus the TP %, far above a price that is already at
its stop, so netting (which only moves sells below the day's buy limit) never touches
it: the TP order is found by exact price and quantity. Anything else stops the step.

The same step sends the spread buys below the sheet's floor (``Expected.extension``):
the backtest fills every step the close reaches, the sheet only lays them down to the
floor. Only steps within ``EXTENSION_REACH`` under the last price go out; a step
already open (same price and quantity) is not sent again.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from trader.broker.base import Order, OrderState, OrderType, OrderRequest, Side
from trader.expected import Expected, Lot
from trader.plan import fit_collar

CENT = Decimal("0.01")
# How far below the last price a below-floor step is still worth sending: the close
# is ten minutes away, a step under this would need a further 10% drop.
EXTENSION_REACH = Decimal("0.10")


class StopError(ValueError):
    pass


@dataclass(frozen=True)
class StopPlan:
    lots: tuple[Lot, ...]       # lots at or below their stop
    cancel: tuple[Order, ...]   # their TP LOC sells
    sell: OrderRequest | None   # one MOC for all their shares (None: nothing to do)
    buys: tuple[OrderRequest, ...] = ()  # spread steps below the sheet's floor, LOC


def _cents(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def stop_lots(expected: Expected, last: Decimal) -> tuple[Lot, ...]:
    return tuple(lot for lot in expected.lots if lot.sl is not None and lot.sl > 0 and last <= lot.sl)


def extension_buys(expected: Expected, last: Decimal, orders_today: list[Order]) -> tuple[OrderRequest, ...]:
    reach = last * (1 - EXTENSION_REACH)
    open_buys = Counter(
        (_cents(o.price), int(o.qty)) for o in orders_today
        if o.symbol == expected.symbol and o.side is Side.BUY and o.type is OrderType.LOC
        and o.state is OrderState.OPEN and o.price is not None
    )
    buys: list[OrderRequest] = []
    for row in expected.extension:
        price = _cents(Decimal(str(row["주문가"])))
        qty = int(row["수량"])
        if price < reach:
            break  # rows run high → low
        if open_buys[(price, qty)] > 0:
            open_buys[(price, qty)] -= 1
            continue
        buys.append(OrderRequest(expected.symbol, Side.BUY, OrderType.LOC, qty, price))
    return tuple(fit_collar(buys, last))


def plan_stop(expected: Expected, last: Decimal, orders_today: list[Order]) -> StopPlan:
    buys = extension_buys(expected, last, orders_today)
    lots = stop_lots(expected, last)
    if not lots:
        return StopPlan((), (), None, buys)
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
    return StopPlan(lots, tuple(cancel), sell, buys)
