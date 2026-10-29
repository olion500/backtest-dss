"""VR reservations: plan from the web VR view, window, resume. Fake broker, no network."""
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from trader.broker.base import BrokerError, Cash, OrderRequest, OrderType, Position, Reservation, Side
from trader.runlog import LocalRunStore
from trader.vr import build_plan, reservation_window, run_vr_reserve, split_existing

NY = ZoneInfo("America/New_York")
SAT = datetime(2026, 10, 31, 10, 0, tzinfo=NY)


def view(orders=None, close_date="2026-10-30"):
    return {
        "account": {"ticker": "TQQQ"},
        "current": {"index": 87, "start": "2026-10-31", "end": "2026-11-13", "v": 67049.55, "v_min": 56992.12,
                    "v_max": 77106.99, "shares_start": 855, "pool_start": 12519.72, "close_date": close_date},
        "orders": orders if orders is not None else [
            {"side": "buy", "price": 66.35, "qty": 4, "filled_on": None},
            {"side": "buy", "price": 65.40, "qty": 4, "filled_on": None},
            {"side": "buy", "price": 66.90, "qty": 4, "filled_on": "2026-11-02"},
            {"side": "sell", "price": 91.30, "qty": 4, "filled_on": None},
            {"side": "sell", "price": 90.41, "qty": 4, "filled_on": None},
        ],
    }


def test_window_starts_at_the_next_unopened_session_and_ends_at_the_last():
    assert reservation_window(date(2026, 10, 31), date(2026, 11, 13), SAT) == (date(2026, 11, 2), date(2026, 11, 13))
    tue_open = datetime(2026, 11, 3, 10, 0, tzinfo=NY)  # already trading: from tomorrow
    assert reservation_window(date(2026, 10, 31), date(2026, 11, 13), tue_open) == (date(2026, 11, 4), date(2026, 11, 13))
    tue_early = datetime(2026, 11, 3, 8, 0, tzinfo=NY)
    assert reservation_window(date(2026, 10, 31), date(2026, 11, 13), tue_early)[0] == date(2026, 11, 3)
    late = datetime(2026, 11, 13, 8, 0, tzinfo=NY)  # one session left: 기간잔량 needs two dates
    assert reservation_window(date(2026, 10, 31), date(2026, 11, 13), late) is None


def test_plan_skips_filled_steps_and_cuts_sells_to_what_the_account_holds():
    plan = build_plan("main", view(), sellable=6, window=(date(2026, 11, 2), date(2026, 11, 13)))
    assert [(o.side, o.qty, o.price) for o in plan.orders] == [
        (Side.BUY, 4, Decimal("66.35")), (Side.BUY, 4, Decimal("65.4")),
        (Side.SELL, 4, Decimal("90.41")), (Side.SELL, 2, Decimal("91.3")),
    ]
    assert all(o.type is OrderType.LIMIT for o in plan.orders)
    assert plan.sells_cut == 2 and plan.buy_cost == Decimal("527.00")


def reservation(order, number="1", cancelled=False):
    return Reservation(number, date(2026, 10, 31), order.symbol, order.side, order.type, Decimal(order.qty),
                       order.price, date(2026, 11, 2), date(2026, 11, 13), Decimal(0), cancelled)


def test_existing_reservations_are_matched_step_by_step():
    plan = build_plan("main", view(), sellable=8, window=(date(2026, 11, 2), date(2026, 11, 13)))
    stranger = reservation(OrderRequest("TQQQ", Side.BUY, OrderType.LIMIT, 8, Decimal("60")), "9")
    todo, strangers = split_existing(plan, [reservation(plan.orders[0]), stranger])
    assert todo == list(plan.orders[1:]) and strangers == [stranger]


class FakeBroker:
    def __init__(self, existing=(), fail_at=None):
        self.existing = list(existing)
        self.reserved = []
        self.fail_at = fail_at

    def positions(self):
        return [Position("TQQQ", Decimal(55), Decimal(55), Decimal("69.5658"))]

    def cash(self, currency="USD"):
        return Cash("USD", Decimal("1104.31"), Decimal("1104.31"))

    def reservations(self, since, until):
        return self.existing

    def reserve(self, order, start, end):
        if self.fail_at is not None and len(self.reserved) == self.fail_at:
            raise BrokerError("거절", code="IGW1")
        self.reserved.append((order, start, end))
        return str(100 + len(self.reserved))


def run(tmp_path, broker, approve=lambda plan, todo: True, the_view=None, **kwargs):
    return run_vr_reserve("main", "MERITZ_VR", approve, now=SAT, broker=broker,
                          view_builder=lambda vr_id, today: the_view or view(), store=LocalRunStore(tmp_path), **kwargs)


def test_sends_every_step_for_the_window_once_approved(tmp_path):
    broker = FakeBroker()
    record = run(tmp_path, broker)
    assert record["status"] == "sent" and len(broker.reserved) == 4
    assert {(start, end) for _, start, end in broker.reserved} == {(date(2026, 11, 2), date(2026, 11, 13))}
    assert record["path"].endswith("-vr-reserve.json") and "vr-main" in record["path"]


def test_limit_then_resume_sends_only_the_rest(tmp_path):
    broker = FakeBroker()
    assert run(tmp_path, broker, limit=1)["status"] == "sent" and len(broker.reserved) == 1
    again = FakeBroker(existing=[reservation(broker.reserved[0][0])])
    record = run(tmp_path, again)
    plan = build_plan("main", view(), 55, (date(2026, 11, 2), date(2026, 11, 13)))
    assert record["status"] == "sent" and [o for o, _, _ in again.reserved] == list(plan.orders[1:])


def test_halts_without_the_close_before_the_cycle_or_on_a_stranger(tmp_path):
    assert run(tmp_path, FakeBroker(), the_view=view(close_date="2026-10-29"))["status"] == "halted"
    stranger = reservation(OrderRequest("TQQQ", Side.SELL, OrderType.LIMIT, 1, Decimal("99")))
    broker = FakeBroker(existing=[stranger])
    assert run(tmp_path, broker)["status"] == "halted" and broker.reserved == []
    broker = FakeBroker(existing=[replace(stranger, cancelled=True)])
    assert run(tmp_path, broker)["status"] == "sent"


def test_declined_sends_nothing_and_a_rejection_stops(tmp_path):
    broker = FakeBroker()
    assert run(tmp_path, broker, approve=lambda plan, todo: False)["status"] == "declined" and broker.reserved == []
    broker = FakeBroker(fail_at=1)
    record = run(tmp_path, broker)
    assert record["status"] == "partial" and len(broker.reserved) == 1 and record["error_code"] == "IGW1"
