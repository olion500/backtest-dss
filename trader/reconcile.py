"""Engine ledger vs broker account. Pure: no I/O, so every rule is unit-tested.

A run sends orders only when every check with ``ok`` is True. Checks with
``ok=None`` are shown for context and never block.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trader.broker.base import Cash, Order, OrderState, Position
from trader.expected import Expected

DEFAULT_CASH_TOLERANCE = Decimal("5")
# The engine's last close (Yahoo, or the dataset) against the broker's previous close:
# a wrong bar mis-states the whole ledger, so it is named before the balances it breaks.
PREV_CLOSE_TOLERANCE = Decimal("0.01")


@dataclass(frozen=True)
class Check:
    key: str
    label: str
    expected: str
    actual: str
    diff: str
    ok: bool | None
    note: str = ""


@dataclass(frozen=True)
class Reconciliation:
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok is not False for check in self.checks)

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if check.ok is False)


def _usd(value: Decimal) -> str:
    sign = "−" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _signed_usd(value: Decimal) -> str:
    return ("+" if value > 0 else "") + _usd(value)


def reconcile(
    expected: Expected,
    positions: list[Position],
    cash: Cash,
    orders_today: list[Order],
    cash_tolerance: Decimal = DEFAULT_CASH_TOLERANCE,
    broker_prev_close: Decimal | None = None,
) -> Reconciliation:
    held = next((p for p in positions if p.symbol == expected.symbol), None)
    real_qty = held.qty if held else Decimal(0)
    qty_diff = real_qty - expected.qty
    cash_diff = cash.estimated - expected.cash
    lot_cost = sum((lot.buy_price * lot.qty for lot in expected.lots), Decimal(0))
    engine_avg = (lot_cost / expected.qty).quantize(Decimal("0.01")) if expected.qty else None
    pending = [o for o in orders_today if o.symbol == expected.symbol and o.state in (OrderState.OPEN, OrderState.PARTIAL)]
    others = sorted({p.symbol for p in positions if p.symbol != expected.symbol})

    checks = []
    if broker_prev_close is not None and expected.prev_close is not None:
        close_diff = expected.prev_close - broker_prev_close
        checks.append(Check(
            key="prev_close", label=f"{expected.symbol} 전일 종가 ({expected.as_of})",
            expected=_usd(expected.prev_close), actual=_usd(broker_prev_close),
            diff=_signed_usd(close_diff) if close_diff else "0",
            ok=abs(close_diff) <= PREV_CLOSE_TOLERANCE, note="시세 데이터 확인 (엔진 vs 증권사)",
        ))
    checks += [
        Check(
            key="qty", label=f"{expected.symbol} 수량",
            expected=f"{expected.qty}주", actual=f"{real_qty.normalize():f}주",
            diff=f"{qty_diff.normalize():+f}주" if qty_diff else "0",
            ok=qty_diff == 0, note="허용 0주",
        ),
        Check(
            key="cash", label=f"{cash.currency} 추정 예수금",
            expected=_usd(expected.cash), actual=_usd(cash.estimated),
            diff=_signed_usd(cash_diff) if cash_diff else "0",
            ok=abs(cash_diff) <= cash_tolerance, note=f"허용 ±{_usd(cash_tolerance)}",
        ),
        Check(
            key="open_orders", label="오늘 미체결 주문",
            expected="0건", actual=f"{len(pending)}건", diff=str(len(pending)) if pending else "0",
            ok=not pending, note="이미 낸 주문이 있으면 다시 내지 않음",
        ),
        Check(
            key="avg_price", label="평균 단가",
            expected=_usd(engine_avg) if engine_avg is not None else "—",
            actual=_usd(held.avg_price) if held else "—",
            diff=_signed_usd(held.avg_price - engine_avg) if held and engine_avg is not None else "",
            ok=None, note="참고 (증권사 평단은 매도 후에도 섞여 있음)",
        ),
        Check(
            key="settled_cash", label=f"{cash.currency} 예수금 (결제 완료)",
            expected="—", actual=_usd(cash.settled), diff="", ok=None, note="참고",
        ),
    ]
    if others:
        checks.append(Check(
            key="other_symbols", label="다른 종목", expected="—", actual=", ".join(others), diff="",
            ok=None, note="동파법 대상이 아니라 대조하지 않음",
        ))
    return Reconciliation(checks=tuple(checks))
