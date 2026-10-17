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
from trader.runner import run_check

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
    regular = us_session(date(2026, 10, 19))
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
    next_day: date
    held: list = field(default_factory=lambda: [soxl()])
    money: Cash = field(default_factory=usd)
    orders: list = field(default_factory=list)
    fail: bool = False
    name: str = "fake"

    def next_session(self):
        if self.fail:
            raise BrokerError("연결 실패", code="UNREACHABLE")
        return self.next_day

    def positions(self):
        return self.held

    def cash(self, currency="USD"):
        return self.money

    def orders_today(self):
        return self.orders


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
    record = run(tmp_path, MONDAY, FakeBroker(date(2026, 10, 19)))
    assert record["status"] == "ok"
    [saved] = LocalRunStore(tmp_path).recent("main")
    assert saved["status"] == "ok" and saved["step"] == "check" and saved["session"] == "2026-10-19"
    assert saved["expected"]["cash"] == "10000.00" and saved["actual"]["positions"][0]["qty"] == "120"
    assert saved["calendar"]["stop_check_at"].startswith("2026-10-19T15:45")


def test_closed_day_does_nothing_unless_forced(tmp_path):
    assert run(tmp_path, SATURDAY, FakeBroker(date(2026, 10, 19)))["status"] == "closed"
    forced = run(tmp_path, SATURDAY, FakeBroker(date(2026, 10, 19)), force=True)
    assert forced["status"] == "ok" and forced["forced"] is True


def test_calendar_disagreement_halts(tmp_path):
    record = run(tmp_path, MONDAY, FakeBroker(date(2026, 10, 20)))
    assert record["status"] == "halted" and "달력" in record["message"]


def test_stale_engine_close_halts(tmp_path):
    record = run(tmp_path, MONDAY, FakeBroker(date(2026, 10, 19)), expected(as_of="2026-10-15"))
    assert record["status"] == "halted" and "2026-10-16" in record["message"]


def test_mismatch_halts_and_names_the_check(tmp_path):
    record = run(tmp_path, MONDAY, FakeBroker(date(2026, 10, 19), held=[soxl("300")]))
    assert record["status"] == "halted" and "SOXL 수량" in record["message"]


def test_broker_error_is_recorded_not_raised(tmp_path):
    record = run(tmp_path, MONDAY, FakeBroker(date(2026, 10, 19), fail=True))
    assert record["status"] == "error" and record["error_code"] == "UNREACHABLE"
    assert LocalRunStore(tmp_path).recent("main")[0]["status"] == "error"
