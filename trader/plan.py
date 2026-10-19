"""The order book's sheet as broker orders. Pure: the rows the web shows, nothing recomputed.

Every row becomes one order: LOC at its price, except the two MOC kinds (an expiring
lot and a lot the backtest stop-lost at the last close). An unknown label stops the
plan rather than guessing a side or type.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from engines.order_book_engine import MOC_SELL, SL_MOC_SELL
from trader.broker.base import OrderRequest, OrderType, Side
from trader.expected import Expected

CENT = Decimal("0.01")
MOC_KINDS = frozenset({MOC_SELL, SL_MOC_SELL})


class PlanError(ValueError):
    pass


def _side(label: str) -> Side:
    if label.startswith("매도"):
        return Side.SELL
    if label.startswith("매수"):
        return Side.BUY
    raise PlanError(f"주문표의 모르는 구분: {label!r}")


def _qty(row: dict) -> int:
    raw = Decimal(str(row.get("수량") or 0))
    if raw != raw.to_integral_value() or raw <= 0:
        raise PlanError(f"주문 수량이 양의 정수가 아닙니다: {row.get('구분')} {row.get('수량')}")
    return int(raw)


def stop_loss_rows(expected: Expected) -> list[dict]:
    """Lots the engine already sold at the last close that the account still holds."""
    return [row for row in expected.orders if row.get("구분") == SL_MOC_SELL]


def plan_orders(expected: Expected) -> list[OrderRequest]:
    orders: list[OrderRequest] = []
    for row in expected.orders:
        label = str(row.get("구분") or "")
        side = _side(label)
        qty = _qty(row)
        if label in MOC_KINDS:
            orders.append(OrderRequest(expected.symbol, side, OrderType.MOC, qty))
            continue
        price = Decimal(str(row.get("주문가") or 0)).quantize(CENT, rounding=ROUND_HALF_UP)
        if price <= 0:
            raise PlanError(f"주문가가 없습니다: {label}")
        orders.append(OrderRequest(expected.symbol, side, OrderType.LOC, qty, price))

    # The account holds the engine's lots plus the stop-lost ones still to sell.
    held = expected.qty + sum(_qty(row) for row in stop_loss_rows(expected))
    selling = sum(order.qty for order in orders if order.side is Side.SELL)
    if selling > held:
        raise PlanError(f"매도 수량 합 {selling}주가 보유 {held}주보다 많습니다")
    return orders


def describe(order: OrderRequest) -> str:
    side = "매도" if order.side is Side.SELL else "매수"
    price = f"${order.price}" if order.price is not None else "시장가"
    return f"{side} {order.type.value.upper()} {order.qty}주 {price}"
