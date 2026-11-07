"""Report step: today's sent orders vs the broker's fills at the close, lot outcomes. Fake broker."""
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

import pytest

from trader.broker.base import Cash, Order, OrderState, OrderType, Position, Quote, Side
from trader.calendar import NY
from trader.config import TraderAccount
from trader.report import bought_qty, check_fills, closing_price, expected_fill, lot_outcomes, sent_today, stop_warnings
from trader.runlog import LocalRunStore
from trader.runner import run_report

SESSION = "2026-11-06"
AFTER_CLOSE = datetime(2026, 11, 6, 18, 0, tzinfo=NY)  # 08:00 KST on Saturday
LOTS = [{"buy_date": "2026-11-03", "qty": 37, "buy_price": "73.12", "tp": "74.04", "sl": None},
        {"buy_date": "2026-11-05", "qty": 59, "buy_price": "40.87", "tp": "41.38", "sl": None}]


def req(side, qty, price, type_="loc"):
    return {"symbol": "SOXL", "side": side, "type": type_, "qty": qty, "price": price}


# The 11/6 sheet, netted: the 11/5 lot's TP moved above the buy limit, the buy moved under the TP.
SENT = [("1", req("sell", 37, "74.04")), ("2", req("sell", 59, "44.05")), ("3", req("buy", 59, "41.37")),
        ("4", req("buy", 1, "40.99"))]


def orders_record(**extra):
    return {"account": "main", "step": "orders", "session": SESSION, "started_at": "2026-11-06T09:45:21-05:00",
            "status": "sent", "sent": [{"request": r, "order_id": i} for i, r in SENT],
            "expected": {"symbol": "SOXL", "prev_close": "40.87", "lots": LOTS, "orders": []}, **extra}


@pytest.mark.parametrize("request_, close, qty", [
    (req("buy", 5, "41.37"), Decimal("41.37"), 5), (req("buy", 5, "41.37"), Decimal("41.38"), 0),
    (req("sell", 5, "44.05"), Decimal("44.05"), 5), (req("sell", 5, "44.05"), Decimal("44.04"), 0),
    (req("sell", 5, None, "moc"), Decimal("1"), 5),
])
def test_expected_fill_follows_the_loc_rule(request_, close, qty):
    assert expected_fill(request_, close) == qty


def test_sent_today_drops_what_stop_cancelled():
    stop = {"step": "stop", "started_at": "2026-11-06T15:45:00-05:00", "cancelled": ["2"],
            "sent": [{"request": req("sell", 59, None, "moc"), "order_id": "9"}, {"request": req("buy", 1, "30"), "error": "x"}]}
    assert [s["order_id"] for s in sent_today([stop, orders_record()])] == ["1", "3", "4", "9"]


def broker_order(order_id, request, filled, price=None):
    return {"order_id": order_id, "side": request["side"], "qty": request["qty"], "filled_qty": filled, "fill_price": price}


def test_fills_match_the_close_and_strays_are_listed():
    close = Decimal("45.97")
    broker = [broker_order(i, r, 59 if i == "2" else 0, "45.97" if i == "2" else None) for i, r in SENT]
    broker.append({"order_id": "77", "side": "buy", "qty": 1, "filled_qty": 1, "fill_price": "45.90"})
    checks, stray = check_fills([{"order_id": i, "request": r} for i, r in SENT], broker, close)
    assert all(c.ok for c in checks) and [s["order_id"] for s in stray] == ["77"]
    late = [broker_order(i, r, 0) for i, r in SENT]  # the TP did not fill
    assert [c.order_id for c in check_fills([{"order_id": i, "request": r} for i, r in SENT], late, close)[0] if not c.ok] == ["2"]


def test_lot_outcomes_and_the_gross_buy_inside_netting():
    close = Decimal("42.50")  # inside the netted zone: TP and buy both happen, nothing moves at the broker
    outcomes = lot_outcomes(LOTS, [], [], close, Decimal("0.07"), Decimal("0.00206"))
    [tp] = outcomes
    gross = (close - Decimal("40.87")) * 59
    cost = Decimal("1.69") + Decimal("1.76") + Decimal("0.05")  # SEC fee 2507.50 × 0.00206% = 0.05
    assert tp.kind == "익절" and tp.pnl == gross - cost
    checks, _ = check_fills([{"order_id": i, "request": r} for i, r in SENT], [], close)
    assert bought_qty(checks, 59) == 59  # net 0 at the broker, the engine bought 59 and sold 59


def test_stop_warnings_when_the_close_and_the_1545_price_disagree():
    lots = [dict(LOTS[1], sl="36.78")]
    assert "손절 안 함" in stop_warnings(lots, [], Decimal("36.50"))[0]
    sold = [{"buy_date": "2026-11-05", "qty": 59}]
    assert "엔진은 보유" in stop_warnings(lots, sold, Decimal("37.00"))[0]
    assert stop_warnings(lots, sold, Decimal("36.50")) == []
    assert lot_outcomes(lots, [], sold, Decimal("36.50"))[0].kind == "손절"


@dataclass
class ReportBroker:
    fills: dict
    last: Decimal = Decimal("45.97")
    auction: Decimal = Decimal("45.97")  # what the closing auction filled at
    name: str = "fake"
    asked: list = field(default_factory=list)

    def quote(self, symbol):
        return Quote(symbol, self.last, Decimal("40.87"), AFTER_CLOSE)

    def orders_today(self):
        return [Order(i, "SOXL", Side(r["side"]), OrderType(r["type"]), Decimal(r["qty"]),
                      Decimal(r["price"]) if r["price"] else None, Decimal(self.fills.get(i, 0)),
                      self.auction if self.fills.get(i) else None, OrderState.FILLED if self.fills.get(i) else OrderState.OPEN)
                for i, r in SENT]

    def positions(self):
        return [Position("SOXL", Decimal(72), Decimal(72), Decimal("63"), self.last)]

    def cash(self, currency="USD"):
        return Cash("USD", Decimal(0), Decimal("31567.20"))


def report(tmp_path, broker, now=AFTER_CLOSE):
    store = LocalRunStore(tmp_path)
    store.save("main", orders_record())
    return run_report("main", now=now, broker_builder=lambda _: broker,
                      load_account=lambda _: TraderAccount(id="main", broker="fake"),
                      fees=lambda _: (Decimal(0), Decimal(0)), store=store)


def test_report_ok_when_the_broker_filled_as_the_close_says(tmp_path):
    record = report(tmp_path, ReportBroker({"2": 59}, last=Decimal("46.40")))  # after hours: last is not the close
    assert record["status"] == "ok", record["message"]
    assert record["close"] == Decimal("45.97") and record["close_source"] == "fill"
    assert record["lots_closed"][0]["kind"] == "익절" and record["lots_closed"][0]["pnl"] == Decimal("300.90")
    assert record["bought_qty"] == 0 and record["position_qty"] == Decimal(72)


def test_report_mismatch_when_a_fill_is_missing(tmp_path):
    record = report(tmp_path, ReportBroker({}))
    assert record["close_source"] == "quote"
    assert record["status"] == "mismatch" and "1건이 종가 기준 예상과 다르게 체결" in record["message"]


def test_closing_price_is_the_auction_fill_else_the_last_price():
    sent = [{"order_id": "2", "request": {}}, {"order_id": "3", "request": {}}]
    fills = [{"order_id": "2", "filled_qty": "59", "fill_price": "45.97"}, {"order_id": "9", "filled_qty": "1", "fill_price": "46.50"}]
    assert closing_price(sent, fills, Decimal("46.40")) == (Decimal("45.97"), "fill")
    assert closing_price(sent, [], Decimal("46.40")) == (Decimal("46.40"), "quote")


@pytest.mark.parametrize("now", [datetime(2026, 11, 6, 15, 59, tzinfo=NY),   # before the close
                                 datetime(2026, 11, 7, 18, 0, tzinfo=NY)])   # Saturday: no session
def test_report_only_after_a_sessions_close(tmp_path, now):
    assert report(tmp_path, ReportBroker({"2": 59}), now=now)["status"] == "closed"


def test_report_after_an_early_close(tmp_path):
    early = datetime(2026, 11, 27, 18, 0, tzinfo=NY)  # the day after Thanksgiving closes at 13:00
    assert report(tmp_path, ReportBroker({"2": 59}), now=early)["status"] != "closed"
