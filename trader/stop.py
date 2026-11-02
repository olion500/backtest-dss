"""Same-day stop loss. Pure: picks the lots and the orders to cancel, no I/O.

The backtest sells a lot at the close when ``close <= SL``; no LOC sell can say that,
so the stop step looks at the price just before the close instead. A lot past its
stop gets its take-profit LOC cancelled (the broker counts shares under an open sell
as not sellable) and the shares go out as one MOC sell.

Netting can still split a stop lot's TP: yesterday's lot has a TP (buy + TP %) below
today's buy limit (close + buy %) whenever TP % < buy %, so ``apply_netting`` sends it
as the remainder at the TP and moves the offset shares into one LOC sell at the buy
limit + $0.01 (shared with every other netted lot). The stop cancels the remainder
and that moved sell, and puts the moved sell back for the shares that are not
stopping. The mirror buy at TP − $0.01 stays: it stands in for the netted buy and
differs from it only for a close between the TP and the buy limit, far above a price
already at the stop. Orders that do not add up to the lot stop the step.

The same step sends the spread buys below the sheet's floor (``Expected.extension``):
the backtest fills every step the close reaches, the sheet only lays them down to the
floor. Only steps within ``EXTENSION_REACH`` under the last price go out; a step
already open (same price and quantity) is not sent again.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from engines.order_book_engine import MOC_SELL
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
    resend: tuple[OrderRequest, ...] = ()  # netting's moved sell, back for the lots not stopping


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


def moved_sells(expected: Expected) -> tuple[Decimal | None, int]:
    """Price and shares of the sheet's LOC sells that sit at no lot's TP: netting's moved sell."""
    tps = {_cents(lot.tp) for lot in expected.lots if lot.tp is not None}
    rows = [
        (_cents(Decimal(str(row["주문가"]))), int(row["수량"])) for row in expected.orders
        if str(row.get("구분", "")).startswith("매도") and row.get("구분") != MOC_SELL
    ]
    moved = [(price, qty) for price, qty in rows if price not in tps]
    prices = {price for price, _ in moved}
    if len(prices) > 1:
        raise StopError(f"익절가가 아닌 매도가가 여러 개입니다: {', '.join(f'${p}' for p in sorted(prices))}")
    return (prices.pop() if prices else None), sum(qty for _, qty in moved)


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
    taken: list[tuple[Lot, Decimal, int]] = []  # (lot, TP, shares netting moved off its TP order)
    for lot in lots:
        if moc_left[lot.qty] > 0:
            moc_left[lot.qty] -= 1
            continue
        if lot.tp is None:
            raise StopError(f"{lot.buy_date} {lot.qty}주: 익절가가 없어 주문을 찾을 수 없습니다")
        tp = _cents(lot.tp)
        at_tp = [o for o in loc_sells if o.price is not None and _cents(o.price) == tp and int(o.qty) <= lot.qty]
        # The whole lot at its TP; else netting's remainder (none when netting took it all).
        match = next((o for o in at_tp if int(o.qty) == lot.qty), None) or max(at_tp, key=lambda o: o.qty, default=None)
        kept = int(match.qty) if match else 0
        if match:
            loc_sells.remove(match)
            cancel.append(match)
        if kept < lot.qty:
            taken.append((lot, tp, lot.qty - kept))
        selling.append(lot)

    resend: tuple[OrderRequest, ...] = ()
    moved_qty = sum(qty for _, _, qty in taken)
    if moved_qty:
        price, sheet_qty = moved_sells(expected)
        moved = [o for o in loc_sells if price is not None and o.price is not None and _cents(o.price) == price]
        open_qty = sum(int(o.qty) for o in moved)
        if price is None or sheet_qty < moved_qty or open_qty != sheet_qty:
            lot, tp, qty = taken[0]
            where = f"퉁치기로 옮긴 매도 ${price} {sheet_qty}주 중 미체결 {open_qty}주" if price else "퉁치기로 옮긴 매도 없음"
            raise StopError(f"{lot.buy_date} {lot.qty}주: 익절 LOC ${tp} {qty}주가 미체결 목록에 없습니다 ({where})")
        cancel.extend(moved)
        if sheet_qty > moved_qty:
            resend = (OrderRequest(expected.symbol, Side.SELL, OrderType.LOC, sheet_qty - moved_qty, price),)
    qty = sum(lot.qty for lot in selling)
    sell = OrderRequest(expected.symbol, Side.SELL, OrderType.MOC, qty) if qty else None
    return StopPlan(lots, tuple(cancel), sell, buys, resend)
