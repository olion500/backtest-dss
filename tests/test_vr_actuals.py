"""VR cycle start from the broker's balance + external holdings (``trader vr --auto``)."""
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

import trader.vr as vr
from engines.vr_engine import VRConfig, run_vr
from tests.test_trader_vr import FakeBroker, view
from trader.config import TraderAccount
from trader.runlog import LocalRunStore
from web.api.app.services.vr import ACTUALS, VRAccount

NY = ZoneInfo("America/New_York")
FRI_NIGHT = datetime(2026, 11, 13, 19, 30, tzinfo=NY)  # 08:30 KST Sat: after the cycle's last close


def account(**kwargs) -> VRAccount:
    return VRAccount.model_validate({
        "id": "main", "name": "VR", "start_date": "2023-07-01", "start_shares": 34, "start_pool": 0,
        "external": [{"from": "2026-11-14", "shares": 800, "pool": 1985.75}], **kwargs,
    })


def test_balances_plus_external_make_the_cycle_start():
    actuals = account().actuals({"2026-11-14": {"shares": 109, "cash": "41.22"}, "2026-10-31": {"shares": 55, "cash": "10"}})
    assert actuals[date(2026, 11, 14)] == {"shares": 909, "pool": 2026.97}
    assert actuals[date(2026, 10, 31)] == {"shares": 55, "pool": 10.0}  # no external before its date


def _bars(start: str, closes: list[float]) -> pd.DataFrame:
    index = pd.bdate_range(start=start, periods=len(closes))
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes}, index=index)


def test_engine_starts_a_cycle_from_actuals_without_adding_the_deposit_again():
    bars = _bars("2024-01-01", [100.0] * 40)
    config = VRConfig(start_date=date(2024, 1, 6), start_shares=100, start_pool=1000, deposit=500,
                      actuals={date(2024, 1, 20): {"shares": 90, "pool": 300.0}})
    cycles = run_vr(bars, config, date(2024, 1, 25))
    # The real pool already holds what was paid in; V still grows by the deposit.
    assert (cycles[1].shares_start, cycles[1].pool_start, cycles[1].actual) == (90, 300.0, True)
    plain = run_vr(bars, VRConfig(start_date=date(2024, 1, 6), start_shares=100, start_pool=1000, deposit=500,
                                  adjustments=[{"date": date(2024, 1, 20), "shares": 90, "pool": 300.0}]), date(2024, 1, 25))
    assert plain[1].pool_start == 800.0 and plain[1].v == cycles[1].v
    assert cycles[0].actual is False


def test_state_documents_live_apart_from_the_records(tmp_path):
    store = LocalRunStore(tmp_path)
    assert store.read_state("vr-main", ACTUALS) == {}
    store.write_state("vr-main", ACTUALS, {"cycles": {"2026-11-14": {"shares": 1}}})
    assert store.read_state("vr-main", ACTUALS)["cycles"]["2026-11-14"] == {"shares": 1}
    assert store.recent("vr-main") == []


def _run(tmp_path, monkeypatch, now=FRI_NIGHT, auto_trade=True, broker=None):
    monkeypatch.setattr(vr, "_vr_account", lambda vr_id: account())
    seen = {}

    def builder(vr_id, today, balances):
        seen.update(today=today, balances=balances)
        current = view(close_date="2026-11-13")
        current["current"].update(start="2026-11-14", end="2026-11-27", actual=True)
        return current

    store = LocalRunStore(tmp_path)
    settings = TraderAccount(id="main", broker="meritz", auto_trade=auto_trade, credentials_env="MERITZ_VR")
    broker = broker or FakeBroker()
    record = vr.run_vr_auto("main", now=now, broker=broker, settings=settings, view_builder=builder, store=store)
    return record, broker, store, seen


def test_before_the_first_open_it_saves_the_balance_once_and_reserves(tmp_path, monkeypatch):
    record, broker, store, seen = _run(tmp_path, monkeypatch)
    assert record["status"] == "sent", record["message"]
    assert seen["today"] == date(2026, 11, 16)
    saved = store.read_state("vr-main", ACTUALS)["cycles"]["2026-11-14"]
    assert (saved["shares"], saved["cash"], saved["close_date"]) == (55, "1104.31", "2026-11-13")
    assert len(broker.reserved) == 4
    assert all(start == date(2026, 11, 16) and end == date(2026, 11, 27) for _, start, end in broker.reserved)

    # A rerun (Sunday) keeps the saved balance, even if the account moved since.
    class Moved(FakeBroker):
        def positions(self):
            return []

    sunday = datetime(2026, 11, 15, 19, 30, tzinfo=NY)
    again, _, store, _ = _run(tmp_path, monkeypatch, now=sunday, broker=Moved())
    assert store.read_state("vr-main", ACTUALS)["cycles"]["2026-11-14"]["shares"] == 55
    assert "balance_saved" not in again


def test_mid_cycle_and_during_a_session_it_does_nothing(tmp_path, monkeypatch):
    wednesday = datetime(2026, 11, 18, 19, 30, tzinfo=NY)
    record, broker, store, _ = _run(tmp_path, monkeypatch, now=wednesday)
    assert record["status"] == "closed" and broker.reserved == []
    friday_session = datetime(2026, 11, 13, 11, 0, tzinfo=NY)
    record, broker, store, _ = _run(tmp_path, monkeypatch, now=friday_session)
    assert record["status"] == "closed" and store.read_state("vr-main", ACTUALS) == {}


def test_auto_trade_off_saves_the_balance_but_sends_nothing(tmp_path, monkeypatch):
    record, broker, store, _ = _run(tmp_path, monkeypatch, auto_trade=False)
    assert record["status"] == "declined" and broker.reserved == []
    assert "2026-11-14" in store.read_state("vr-main", ACTUALS)["cycles"]


def test_buys_beyond_the_broker_cash_are_flagged(tmp_path, monkeypatch):
    class Poor(FakeBroker):
        def cash(self, currency="USD"):
            from decimal import Decimal
            from trader.broker.base import Cash
            return Cash("USD", Decimal("100"), Decimal("100"))

    record, *_ = _run(tmp_path, monkeypatch, broker=Poor())
    assert record["cash_short"] == "427.00"


def test_a_new_cycle_posts_one_discord_embed(tmp_path, monkeypatch):
    from trader.notify import build_embeds

    record, *_ = _run(tmp_path, monkeypatch)
    [embed] = build_embeds("vr", [record])
    assert embed["title"].startswith("🔁 [vr-main] 사이클 87")
    assert "외부 800주" in embed["description"]
    closed, *_ = _run(tmp_path, monkeypatch, now=datetime(2026, 11, 18, 19, 30, tzinfo=NY))
    assert build_embeds("vr", [closed]) == []
