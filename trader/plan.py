"""The order book's sheet as broker orders. Pure: the rows the web shows, nothing recomputed.

Every row becomes one order: LOC at its price, except an expiring lot's MOC sell.
An unknown label stops the plan rather than guessing a side or type.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal

from engines.order_book_engine import MOC_SELL
from trader.broker.base import OrderRequest, OrderType, Side
from trader.expected import Expected

CENT = Decimal("0.01")
# Meritz takes a limit more than 20% through the NBBO, then rejects it ("Broker Rejected: LimitPrice
# 107.68 > 24.59% Limit 20.00% thru nbbo", 2026-10-22, SOXL at $86.4). A LOC fills at the close
# whatever its limit, so pulling the limit to 19% of the last price changes the fill only if the close
# ends more than 19% beyond the price at send time.
COLLAR = Decimal("0.19")


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


def plan_orders(expected: Expected) -> list[OrderRequest]:
    orders: list[OrderRequest] = []
    for row in expected.orders:
        label = str(row.get("구분") or "")
        side = _side(label)
        qty = _qty(row)
        if label == MOC_SELL:
            orders.append(OrderRequest(expected.symbol, side, OrderType.MOC, qty))
            continue
        price = Decimal(str(row.get("주문가") or 0)).quantize(CENT, rounding=ROUND_HALF_UP)
        if price <= 0:
            raise PlanError(f"주문가가 없습니다: {label}")
        orders.append(OrderRequest(expected.symbol, side, OrderType.LOC, qty, price))

    selling = sum(order.qty for order in orders if order.side is Side.SELL)
    if selling > expected.qty:
        raise PlanError(f"매도 수량 합 {selling}주가 보유 {expected.qty}주보다 많습니다")
    return orders


def fit_collar(plan: list[OrderRequest], last: Decimal) -> list[OrderRequest]:
    """Buy limits at most ``last × 1.19``, sell limits at least ``last × 0.81``; MOC untouched."""
    high = (last * (1 + COLLAR)).quantize(CENT, rounding=ROUND_FLOOR)
    low = (last * (1 - COLLAR)).quantize(CENT, rounding=ROUND_CEILING)
    fitted = []
    for order in plan:
        if order.price is not None and order.side is Side.BUY and order.price > high:
            order = replace(order, price=high)
        elif order.price is not None and order.side is Side.SELL and order.price < low:
            order = replace(order, price=low)
        fitted.append(order)
    return fitted


def describe(order: OrderRequest) -> str:
    side = "매도" if order.side is Side.SELL else "매수"
    price = f"${order.price}" if order.price is not None else "시장가"
    return f"{side} {order.type.value.upper()} {order.qty}주 {price}"
