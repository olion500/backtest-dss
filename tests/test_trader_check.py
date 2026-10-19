"""Trader check step: calendar, reconcile rules, run records. Fake broker, no network."""
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

from trader.broker.base import BrokerError, Cash, Order, OrderState, OrderType, Position, Side
from trader.calendar import NY, last_closed_session, us_session
import pytest

from trader.config import DEFAULT_LISTINGS, TraderAccount, TraderConfigError, load_repo_env, load_trader_account
from trader.expected import Expected, Lot
from trader.reconcile import reconcile
from trader.runlog import LocalRunStore
from trader.plan import PlanError, plan_orders
from trader.runner import run_check, run_orders

LOTS = (
    Lot("2026-10-14", 40, Decimal("110.54"), Decimal("114.31"), Decimal("90.53")),
    Lot("2026-10-15", 30, Decimal("107.35"), Decimal("111.01"), Decimal("87.92")),
    Lot("2026-10-16", 50, Decimal("93.67"), Decimal("96.86"), Decimal("76.72")),
)


def expected(**overrides):
    values = dict(account_id="main", symbol="SOXL", as_of="2026-10-16", mode="offense", cash=Decimal("10000.00"),
                  qty=120, prev_close=Decimal("93.67"), tranche_budget=Decimal("7718"), lots=LOTS, orders=())
    values.update(overrides)
    return Expected(**values)


def soxl(qty="120", avg="102.71"):
    return Position("SOXL", Decimal(qty), Decimal(qty), Decimal(avg), Decimal("93.67"))


def usd(estimated="10000.00", settled="10000"):
    return Cash("USD", Decimal(settled), Decimal(estimated))


def order(state=OrderState.OPEN, symbol="SOXL"):
    return Order("1", symbol, Side.BUY, OrderType.LOC, Decimal(1), Decimal("97.7"), Decimal(0), None, state)


# --- calendar ---

def test_holiday_early_close_and_regular_day():
    assert not us_session(date(2026, 11, 26)).is_open           # Thanksgiving
    early = us_session(date(2026, 11, 27))
    assert early.is_open and early.early_close
    assert early.stop_check_at == datetime(2026, 11, 27, 12, 45, tzinfo=NY)
    assert early.close_order_cutoff == datetime(2026, 11, 27, 12, 50, tzinfo=NY)
    assert early.trading_at(datetime(2026, 11, 27, 12, 59, tzinfo=NY))
    assert not early.trading_at(datetime(2026, 11, 27, 13, 0, tzinfo=NY))
    regular = us_session(date(2026, 10, 19))
    assert regular.open == datetime(2026, 10, 19, 9, 30, tzinfo=NY)
    assert not regular.trading_at(datetime(2026, 10, 19, 9, 29, tzinfo=NY))
    assert not us_session(date(2026, 11, 26)).trading_at(datetime(2026, 11, 26, 12, 0, tzinfo=NY))
    assert not regular.early_close and regular.stop_check_at.hour == 15 and regular.stop_check_at.minute == 45


def test_last_closed_session_skips_weekends_and_holidays():
    assert last_closed_session(date(2026, 10, 19)) == date(2026, 10, 16)  # Monday → Friday
    assert last_closed_session(date(2026, 10, 17)) == date(2026, 10, 16)  # Saturday → Friday
    assert last_closed_session(date(2026, 11, 27)) == date(2026, 11, 25)  # after Thanksgiving


# --- reconcile ---

def test_matching_account_passes():
    result = reconcile(expected(), [soxl()], usd(), [])
    assert result.ok and not result.failures


def test_share_count_must_match_exactly():
    result = reconcile(expected(), [soxl("300", "131.00")], usd(), [])
    assert not result.ok and [c.key for c in result.failures] == ["qty"]
    assert result.failures[0].diff == "+180주"


def test_missing_position_counts_as_zero_shares():
    result = reconcile(expected(), [], usd(), [])
    assert result.failures[0].actual == "0주"


def test_cash_within_tolerance_passes_beyond_fails():
    assert reconcile(expected(), [soxl()], usd("9995.00"), []).ok
    failed = reconcile(expected(), [soxl()], usd("9994.07"), [])
    assert [c.key for c in failed.failures] == ["cash"] and failed.failures[0].diff == "−$5.93"


def test_open_orders_for_the_target_block_a_second_send():
    assert not reconcile(expected(), [soxl()], usd(), [order()]).ok
    assert reconcile(expected(), [soxl()], usd(), [order(OrderState.FILLED), order(symbol="AAPL")]).ok


STOP_ROW = {"구분": "매도 (손절)", "주문가": 93.67, "수량": 71, "변화율": "-"}


def test_stop_lost_lot_is_still_in_the_account_until_its_moc():
    # The engine sold 71 at the last close; the account holds them and has not got the money.
    ledger = expected(orders=(STOP_ROW,))
    result = reconcile(ledger, [soxl("191")], usd("3349.43"), [])
    assert result.ok
    qty = next(c for c in result.checks if c.key == "qty")
    assert qty.expected == "191주" and "손절 MOC 71주" in qty.note
    assert not reconcile(ledger, [soxl("120")], usd(), []).ok


def test_other_symbols_are_reported_but_never_block():
    other = Position("AAPL", Decimal(3), Decimal(3), Decimal(200))
    result = reconcile(expected(), [soxl(), other], usd(), [])
    assert result.ok and any(c.key == "other_symbols" and c.actual == "AAPL" for c in result.checks)


# --- config ---

def test_trader_config_reads_keys_by_account_prefix(tmp_path, monkeypatch):
    (tmp_path / "main.json").write_text('{"broker": "meritz", "auto_trade": "yes", "listings": {"XYZ": "NYSE"}}')
    monkeypatch.setenv("MERITZ_MAIN_APP_KEY", "abc")
    monkeypatch.setenv("MERITZ_MAIN_APP_SECRET", "def")
    monkeypatch.setenv("MERITZ_SUB_APP_KEY", "other account")
    account = load_trader_account("main", tmp_path)
    assert account.auto_trade is False  # only a real true turns it on
    assert account.listings == {**DEFAULT_LISTINGS, "XYZ": "NYSE"}
    assert account.credentials() == {"APP_KEY": "abc", "APP_SECRET": "def"}
    monkeypatch.delenv("MERITZ_MAIN_APP_KEY")
    monkeypatch.delenv("MERITZ_MAIN_APP_SECRET")
    with pytest.raises(TraderConfigError):
        account.credentials()


def test_trader_config_credentials_file_and_repo_env(tmp_path, monkeypatch):
    env = tmp_path / "m.env"
    env.write_text("# c\nAPP_KEY='abc'\nAPP_SECRET=def\n")
    (tmp_path / "main.json").write_text(f'{{"broker": "meritz", "credentials": "{env}"}}')
    assert load_trader_account("main", tmp_path).credentials() == {"APP_KEY": "abc", "APP_SECRET": "def"}
    monkeypatch.setenv("APP_SECRET", "already set")
    monkeypatch.delenv("APP_KEY", raising=False)
    load_repo_env(env)
    import os
    assert os.environ["APP_KEY"] == "abc" and os.environ["APP_SECRET"] == "already set"
    monkeypatch.delenv("APP_KEY")
    # the prefix is <BROKER>_<ID>; an id that cannot be a variable name must set credentials_env
    (tmp_path / "연향.json").write_text('{"broker": "meritz"}')
    with pytest.raises(TraderConfigError, match="credentials_env"):
        load_trader_account("연향", tmp_path).credentials()
    (tmp_path / "연향.json").write_text('{"broker": "meritz", "credentials_env": "meritz_yh"}')
    monkeypatch.setenv("MERITZ_YH_APP_KEY", "k")
    assert load_trader_account("연향", tmp_path).credentials() == {"APP_KEY": "k"}


# --- runner ---

@dataclass
class FakeBroker:
    open_now: bool
    held: list = field(default_factory=lambda: [soxl()])
    money: Cash = field(default_factory=usd)
    orders: list = field(default_factory=list)
    fail: bool = False
    name: str = "fake"
    submitted: list = field(default_factory=list)
    reject_at: int | None = None

    def market_open(self):
        if self.fail:
            raise BrokerError("연결 실패", code="UNREACHABLE")
        return self.open_now

    def positions(self):
        return self.held

    def cash(self, currency="USD"):
        return self.money

    def orders_today(self):
        return self.orders

    def submit(self, request):
        self.submitted.append(request)
        if len(self.submitted) == self.reject_at:
            raise BrokerError("주문가격 오류", code="IGW50002")
        return Order(str(len(self.submitted)), request.symbol, request.side, request.type, Decimal(request.qty),
                     request.price, Decimal(0), None, OrderState.OPEN)


def run(tmp_path, now, broker, expected_value=None, force=False):
    account = TraderAccount(id="main", broker="fake", credentials_path=None)
    return run_check(
        "main", now=now, force=force,
        broker_builder=lambda _: broker,
        expected_builder=lambda _id, _day: expected_value or expected(),
        load_account=lambda _id: account,
        store=LocalRunStore(tmp_path),
    )


MONDAY = datetime(2026, 10, 19, 9, 40, tzinfo=NY)
SATURDAY = datetime(2026, 10, 17, 9, 40, tzinfo=NY)


def test_ok_run_is_recorded(tmp_path):
    record = run(tmp_path, MONDAY, FakeBroker(True))
    assert record["status"] == "ok"
    [saved] = LocalRunStore(tmp_path).recent("main")
    assert saved["status"] == "ok" and saved["step"] == "check" and saved["session"] == "2026-10-19"
    assert saved["expected"]["cash"] == "10000.00" and saved["actual"]["positions"][0]["qty"] == "120"
    assert saved["calendar"]["stop_check_at"].startswith("2026-10-19T15:45")


def test_closed_day_does_nothing_unless_forced(tmp_path):
    assert run(tmp_path, SATURDAY, FakeBroker(False))["status"] == "closed"
    forced = run(tmp_path, SATURDAY, FakeBroker(False), force=True)
    assert forced["status"] == "ok" and forced["forced"] is True


def test_calendar_disagreement_halts(tmp_path):
    record = run(tmp_path, MONDAY, FakeBroker(False))
    assert record["status"] == "halted" and "달력" in record["message"]
    record = run(tmp_path, SATURDAY, FakeBroker(True))
    assert record["status"] == "halted" and "달력" in record["message"]


def test_session_day_outside_regular_hours_is_closed_without_asking_the_flag(tmp_path):
    # Before the open the broker reports the market closed; that is no disagreement.
    early = datetime(2026, 10, 19, 9, 0, tzinfo=NY)
    record = run(tmp_path, early, FakeBroker(False))
    assert record["status"] == "closed" and "09:30–16:00" in record["message"]
    assert record["calendar"]["broker_open"] is False
    assert run(tmp_path, early, FakeBroker(True), force=True)["status"] == "ok"


def test_stale_engine_close_halts(tmp_path):
    record = run(tmp_path, MONDAY, FakeBroker(True), expected(as_of="2026-10-15"))
    assert record["status"] == "halted" and "2026-10-16" in record["message"]


def test_mismatch_halts_and_names_the_check(tmp_path):
    record = run(tmp_path, MONDAY, FakeBroker(True, held=[soxl("300")]))
    assert record["status"] == "halted" and "SOXL 수량" in record["message"]


def test_broker_error_is_recorded_not_raised(tmp_path):
    record = run(tmp_path, MONDAY, FakeBroker(True, fail=True))
    assert record["status"] == "error" and record["error_code"] == "UNREACHABLE"
    assert LocalRunStore(tmp_path).recent("main")[0]["status"] == "error"


# --- order plan ---

SHEET = (
    {"구분": "매도 (익절)", "주문가": 114.31, "수량": 40},
    {"구분": "매도 (퉁치기)", "주문가": 97.71000000000001, "수량": 20},
    {"구분": "매수 (퉁치기)", "주문가": 96.85, "수량": 20},
    {"구분": "매도 (MOC)", "주문가": 93.67, "수량": 30},
    {"구분": "매수 (+3주)", "주문가": 93.1, "수량": 3},
)


def test_plan_maps_each_sheet_row_to_one_order():
    plan = plan_orders(expected(orders=SHEET))
    assert [(o.side, o.type, o.qty, o.price) for o in plan] == [
        (Side.SELL, OrderType.LOC, 40, Decimal("114.31")),
        (Side.SELL, OrderType.LOC, 20, Decimal("97.71")),
        (Side.BUY, OrderType.LOC, 20, Decimal("96.85")),
        (Side.SELL, OrderType.MOC, 30, None),
        (Side.BUY, OrderType.LOC, 3, Decimal("93.10")),
    ]
    stop = plan_orders(expected(orders=(STOP_ROW,)))
    assert [(o.side, o.type, o.qty) for o in stop] == [(Side.SELL, OrderType.MOC, 71)]


@pytest.mark.parametrize("row, why", [
    ({"구분": "정정", "주문가": 1, "수량": 1}, "구분"),
    ({"구분": "매수", "주문가": 1, "수량": 1.5}, "정수"),
    ({"구분": "매수", "주문가": 0, "수량": 1}, "주문가"),
    ({"구분": "매도 (익절)", "주문가": 120, "수량": 121}, "보유"),
])
def test_plan_refuses_rows_it_cannot_send_as_shown(row, why):
    with pytest.raises(PlanError, match=why):
        plan_orders(expected(orders=(row,)))


# --- orders step ---

def send(tmp_path, broker, answer=True, now=MONDAY, clock=None, sheet=SHEET):
    asked = []

    def approve(plan, ledger):
        asked.append(plan)
        return answer

    account = TraderAccount(id="main", broker="fake", credentials_path=None)
    record = run_orders(
        "main", approve, now=now, clock=clock or (lambda: now),
        broker_builder=lambda _: broker,
        expected_builder=lambda _id, _day: expected(orders=sheet),
        load_account=lambda _id: account,
        store=LocalRunStore(tmp_path),
    )
    return record, asked


def test_approved_sheet_is_sent_in_order(tmp_path):
    broker = FakeBroker(True)
    record, asked = send(tmp_path, broker)
    assert record["status"] == "sent" and len(asked) == 1
    assert [o.qty for o in broker.submitted] == [40, 20, 20, 30, 3]
    saved = LocalRunStore(tmp_path).recent("main")[0]
    assert saved["step"] == "orders" and [s["order_id"] for s in saved["sent"]] == ["1", "2", "3", "4", "5"]


def test_declined_sheet_sends_nothing(tmp_path):
    broker = FakeBroker(True)
    record, _ = send(tmp_path, broker, answer=False)
    assert record["status"] == "declined" and broker.submitted == [] and len(record["plan"]) == 5


def test_failed_check_never_asks(tmp_path):
    broker = FakeBroker(True, held=[soxl("300")])
    record, asked = send(tmp_path, broker)
    assert record["status"] == "halted" and asked == [] and broker.submitted == []


def test_orders_stop_at_the_close_cutoff(tmp_path):
    late = datetime(2026, 10, 19, 15, 50, tzinfo=NY)
    record, asked = send(tmp_path, FakeBroker(True), now=late)
    assert record["status"] == "halted" and "15:50" in record["message"] and asked == []
    # approved too late: still nothing goes out
    broker = FakeBroker(True)
    record, asked = send(tmp_path, broker, clock=lambda: late, now=MONDAY)
    assert record["status"] == "halted" and asked and broker.submitted == []


def test_orders_placed_meanwhile_block_the_send(tmp_path):
    broker = FakeBroker(True)

    def approve_and_place(plan, ledger):
        broker.orders.append(order())
        return True

    account = TraderAccount(id="main", broker="fake", credentials_path=None)
    record = run_orders("main", approve_and_place, now=MONDAY, clock=lambda: MONDAY, broker_builder=lambda _: broker,
                        expected_builder=lambda _id, _day: expected(orders=SHEET), load_account=lambda _id: account,
                        store=LocalRunStore(tmp_path))
    assert record["status"] == "halted" and "미체결" in record["message"] and broker.submitted == []


def test_first_rejection_stops_the_rest(tmp_path):
    broker = FakeBroker(True, reject_at=2)
    record, _ = send(tmp_path, broker)
    assert record["status"] == "partial" and len(broker.submitted) == 2
    assert record["sent"][0]["order_id"] == "1" and record["sent"][1]["error_code"] == "IGW50002"
    assert record["message"].startswith("1/5건")


def test_empty_sheet_asks_nothing(tmp_path):
    record, asked = send(tmp_path, FakeBroker(True), sheet=())
    assert record["status"] == "ok" and asked == []
