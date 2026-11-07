"""After the close: did today's orders fill as the close says they should? (pure)

Each order the trader sent today (orders + stop steps, minus the ones stop cancelled)
should have filled by a simple rule: a buy LOC when ``close <= limit``, a sell LOC when
``close >= limit``, an MOC always. That is compared with the broker's fills. The day's
outcome per lot comes from the morning ledger (the engine's view, before netting):
TP lots at or under the close, the lots the stop step sold, expiring lots; bought shares
follow from the net fills (netting keeps the net the same).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")
HUNDRED = Decimal(100)
EXPIRY_KINDS = ("매도 (MOC)", "매도 (만료)")


@dataclass(frozen=True)
class FillCheck:
    order_id: str
    request: dict
    expected_qty: int
    actual_qty: Decimal | None  # None: not on the broker's list
    fill_price: Decimal | None

    @property
    def ok(self) -> bool:
        return self.actual_qty is not None and self.actual_qty == self.expected_qty


@dataclass(frozen=True)
class LotOutcome:
    buy_date: str
    qty: int
    buy_price: Decimal
    kind: str  # 익절 | 손절 | 만기
    pnl: Decimal


def money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def expected_fill(request: dict, close: Decimal) -> int:
    qty = int(request["qty"])
    if request.get("type") in ("moc", "market") or request.get("price") is None:
        return qty
    limit = Decimal(str(request["price"]))
    if request.get("side") == "buy":
        return qty if close <= limit else 0
    return qty if close >= limit else 0


def sent_today(records: list[dict]) -> list[dict]:
    """``{order_id, request}`` for every order today's orders/stop runs placed and did not cancel."""
    cancelled = {str(order_id) for record in records for order_id in record.get("cancelled", [])}
    out = []
    for record in sorted(records, key=lambda r: r.get("started_at", "")):
        if record.get("step") not in ("orders", "stop"):
            continue
        for item in record.get("sent", []):
            if "order_id" in item and str(item["order_id"]) not in cancelled:
                out.append({"order_id": str(item["order_id"]), "request": item["request"]})
    return out


def check_fills(sent: list[dict], broker_orders: list[dict], close: Decimal) -> tuple[list[FillCheck], list[dict]]:
    """Each sent order vs the broker's fill, and the broker's orders the trader did not send."""
    by_id = {str(order["order_id"]): order for order in broker_orders}
    checks = []
    for item in sent:
        order = by_id.get(item["order_id"])
        checks.append(FillCheck(
            order_id=item["order_id"],
            request=item["request"],
            expected_qty=expected_fill(item["request"], close),
            actual_qty=None if order is None else Decimal(str(order.get("filled_qty") or 0)),
            fill_price=None if order is None or order.get("fill_price") is None else Decimal(str(order["fill_price"])),
        ))
    ours = {item["order_id"] for item in sent}
    stray = [order for order in broker_orders if str(order["order_id"]) not in ours and Decimal(str(order.get("filled_qty") or 0)) > 0]
    return checks, stray


def closing_price(sent: list[dict], broker_orders: list[dict], last: Decimal) -> tuple[Decimal, str]:
    """The close: what today's LOC/MOC fills got (they all fill in the closing auction), else ``last``."""
    ours = {item["order_id"] for item in sent}
    prices = [Decimal(str(order["fill_price"])) for order in broker_orders
              if str(order["order_id"]) in ours and order.get("fill_price") is not None
              and Decimal(str(order.get("filled_qty") or 0)) > 0]
    if prices:
        return max(set(prices), key=prices.count), "fill"
    return last, "quote"


def lot_outcomes(
    lots: list[dict],
    sheet: list[dict],
    stopped: list[dict],
    close: Decimal,
    fee_pct: Decimal = Decimal(0),
    sec_fee_pct: Decimal = Decimal(0),
) -> list[LotOutcome]:
    """The lots the engine closes today at ``close``, with realized P&L after costs."""
    expiring = {
        (_dec(row.get("매수가")), int(row.get("수량") or 0))
        for row in sheet if row.get("구분") in EXPIRY_KINDS and row.get("매수가") is not None
    }
    stop_keys = {(str(lot["buy_date"]), int(lot["qty"])) for lot in stopped}
    out = []
    for lot in lots:
        qty, buy = int(lot["qty"]), Decimal(str(lot["buy_price"]))
        tp = _dec(lot.get("tp"))
        if (str(lot["buy_date"]), qty) in stop_keys:
            kind = "손절"
        elif (buy, qty) in expiring:
            kind = "만기"
        elif tp is not None and close >= tp:
            kind = "익절"
        else:
            continue
        cost = money(buy * qty * fee_pct / HUNDRED) + money(close * qty * fee_pct / HUNDRED)
        if sec_fee_pct > 0:
            cost += max(CENT, money(close * qty * sec_fee_pct / HUNDRED))
        out.append(LotOutcome(str(lot["buy_date"]), qty, buy, kind, money((close - buy) * qty) - cost))
    return out


def stop_warnings(lots: list[dict], stopped: list[dict], close: Decimal) -> list[str]:
    """Lots where the 15:45 decision and the close disagree: the engine sells at ``close <= SL``."""
    stop_keys = {(str(lot["buy_date"]), int(lot["qty"])) for lot in stopped}
    out = []
    for lot in lots:
        sl = _dec(lot.get("sl"))
        sold = (str(lot["buy_date"]), int(lot["qty"])) in stop_keys
        if sl is None and not sold:
            continue
        if sl is not None and close <= sl and not sold:
            out.append(f"{lot['buy_date']} 매수분 {lot['qty']}주: 종가 ${close} ≤ 손절가 ${sl}인데 손절 안 함")
        elif sold and sl is not None and close > sl:
            out.append(f"{lot['buy_date']} 매수분 {lot['qty']}주: 손절했지만 종가 ${close} > 손절가 ${sl} (엔진은 보유)")
    return out


def bought_qty(checks: list[FillCheck], sold_qty: int) -> int:
    """Engine's gross buy today: the net shares the orders should move, plus what the lots sold."""
    net = sum(c.expected_qty if c.request.get("side") == "buy" else -c.expected_qty for c in checks)
    return net + sold_qty


def _dec(value) -> Decimal | None:
    return None if value is None else Decimal(str(value))
