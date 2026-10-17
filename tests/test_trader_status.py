"""Order-book auto-trading status: upcoming session and run records read by the web."""
from datetime import date, datetime

import pytest
from fastapi import HTTPException

from trader.calendar import NY, upcoming_session
from trader.runlog import LocalRunStore
from web.api.app import main as api_main
from web.api.app.services.accounts import Account
from web.api.app.services.trader import trader_status


def test_upcoming_session_is_today_until_the_close_then_the_next():
    assert upcoming_session(datetime(2026, 10, 19, 9, 0, tzinfo=NY)).day == date(2026, 10, 19)
    assert upcoming_session(datetime(2026, 10, 19, 16, 0, tzinfo=NY)).day == date(2026, 10, 20)
    assert upcoming_session(datetime(2026, 10, 17, 12, 0, tzinfo=NY)).day == date(2026, 10, 19)  # Saturday
    early = upcoming_session(datetime(2026, 11, 26, 10, 0, tzinfo=NY))                          # Thanksgiving
    assert early.day == date(2026, 11, 27) and early.early_close


def _save(store, session, started, status, checks=(), forced=False):
    store.save("main", {"account": "main", "step": "check", "session": session, "started_at": started,
                        "status": status, "message": status, "forced": forced, "auto_trade": False,
                        "broker": "meritz", "checks": list(checks)})


CHECK = {"key": "qty", "label": "SOXL 수량", "expected": "120주", "actual": "300주", "diff": "+180주", "ok": False, "note": ""}


def test_status_without_records_is_a_manual_account(tmp_path):
    status = trader_status("main", LocalRunStore(tmp_path), datetime(2026, 10, 19, 9, 0, tzinfo=NY))
    assert status["configured"] is False and status["check"] is None and status["recent"] == []
    assert status["session"]["date"] == "2026-10-19" and status["session"]["stop_check_at"].startswith("2026-10-19T15:45")


def test_status_prefers_this_sessions_check(tmp_path):
    store = LocalRunStore(tmp_path)
    _save(store, "2026-10-16", "2026-10-16T09:40:00-04:00", "halted", [CHECK])
    _save(store, "2026-10-19", "2026-10-19T09:40:00-04:00", "ok", [{**CHECK, "ok": True}])
    status = trader_status("main", store, datetime(2026, 10, 19, 10, 0, tzinfo=NY))
    assert status["check"]["status"] == "ok" and status["check"]["current"] is True
    assert [r["session"] for r in status["recent"]] == ["2026-10-19", "2026-10-16"]


def test_status_falls_back_to_the_latest_real_comparison_not_a_closed_day(tmp_path):
    store = LocalRunStore(tmp_path)
    _save(store, "2026-10-17", "2026-10-17T11:31:02-04:00", "halted", [CHECK], forced=True)
    _save(store, "2026-10-17", "2026-10-17T11:31:10-04:00", "closed")
    status = trader_status("main", store, datetime(2026, 10, 17, 12, 0, tzinfo=NY))
    check = status["check"]
    assert check["status"] == "halted" and check["current"] is False and check["forced"] is True
    assert check["checks"][0]["diff"] == "+180주"


def test_route_rejects_unknown_accounts(monkeypatch):
    monkeypatch.setattr(api_main, "load_accounts", lambda: [Account(id="main", name="메인", start_date=date(2026, 6, 1), initial_cash=1)])
    monkeypatch.setattr(api_main, "trader_status", lambda account: {"account": account})
    assert api_main.trader("main") == {"account": "main"}
    with pytest.raises(HTTPException) as err:
        api_main.trader("../etc")
    assert err.value.status_code == 404


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    def upload_from_string(self, text, content_type=None):
        self.bucket.objects[self.name] = text

    def download_as_text(self, encoding="utf-8"):
        return self.bucket.objects[self.name]


class FakeBucket:
    name = "b"

    def __init__(self):
        self.objects = {}
        self.downloads = 0

    def blob(self, name):
        bucket = self

        class Counting(FakeBlob):
            def download_as_text(self, encoding="utf-8"):
                bucket.downloads += 1
                return super().download_as_text(encoding)

        return Counting(self, name)


class FakeClient:
    def __init__(self):
        self._bucket = FakeBucket()

    def bucket(self, name):
        return self._bucket

    def list_blobs(self, bucket, prefix="", start_offset=""):
        return [FakeBlob(bucket, name) for name in sorted(bucket.objects) if name.startswith(prefix) and name >= start_offset]


def test_gcs_store_keeps_the_layout_and_reads_only_the_newest():
    from trader.runlog import GcsRunStore

    client = FakeClient()
    store = GcsRunStore("b", "runs/", client=client)
    for day in ("2026-10-15", "2026-10-16", "2026-10-19"):
        _save(store, day, f"{day}T09:40:00-04:00", "ok")
    _save(store.__class__("b", "runs", client=client), "2026-10-19", "2026-10-19T15:45:00-04:00", "ok")
    assert "runs/main/2026-10-19/094000-check.json" in client._bucket.objects
    recent = store.recent("main", limit=2, days=100_000)
    assert [(r["session"], r["started_at"][11:16]) for r in recent] == [("2026-10-19", "15:45"), ("2026-10-19", "09:40")]
    assert client._bucket.downloads == 2
    assert store.recent("other") == []
    # days bounds the listing by session date (names start with it), relative to today
    from datetime import timedelta, timezone
    since = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    assert all(r["session"] >= since for r in store.recent("main", days=1))
    assert len(store.recent("main", days=100_000)) == 4


def test_status_reads_broker_and_mode_from_a_record_that_has_them(tmp_path):
    store = LocalRunStore(tmp_path)
    _save(store, "2026-10-16", "2026-10-16T09:40:00-04:00", "ok")
    store.save("main", {"account": "main", "step": "check", "session": "2026-10-19", "started_at": "2026-10-19T09:40:00-04:00",
                        "status": "error", "message": "설정 없음"})  # failed before loading settings
    status = trader_status("main", store, datetime(2026, 10, 19, 10, 0, tzinfo=NY))
    assert status["broker"] == "meritz" and status["auto_trade"] is False


def test_a_failed_save_falls_back_to_a_local_record(tmp_path, monkeypatch):
    from trader import runlog, runner
    from trader.config import TraderAccount

    class Broken:
        def save(self, account_id, record):
            raise ConnectionError("gcs down")

    monkeypatch.setattr(runlog, "DEFAULT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "LocalRunStore", lambda: LocalRunStore(tmp_path))
    record = runner.run_check(
        "main", now=datetime(2026, 10, 17, 9, 40, tzinfo=NY), store=Broken(),
        load_account=lambda _id: TraderAccount(id="main", broker="x", credentials_path=None),
        broker_builder=lambda _a: type("B", (), {"next_session": lambda self: date(2026, 10, 19)})(),
    )
    assert record["status"] == "closed" and "gcs down" in record["store_error"]
    assert LocalRunStore(tmp_path).recent("main")[0]["store_error"].startswith("ConnectionError")


def test_run_store_picks_gcs_directory_or_default(tmp_path, monkeypatch):
    from trader import runlog

    monkeypatch.setattr(runlog, "GcsRunStore", lambda bucket, prefix: ("gcs", bucket, prefix))
    runlog._gcs_store.cache_clear()
    assert runlog.run_store("gs://bkt/trader/runs") == ("gcs", "bkt", "trader/runs")
    assert runlog.run_store(str(tmp_path)).root == tmp_path
    for name in (runlog.ENV_VAR, "GCP_PROJECT", "TRADER_BUCKET"):
        monkeypatch.delenv(name, raising=False)
    assert runlog.run_store().root == runlog.DEFAULT_ROOT
    monkeypatch.setenv("GCP_PROJECT", "proj")
    assert runlog.default_target() == "gs://proj-dongpa-trader/runs"
    monkeypatch.setenv("TRADER_BUCKET", "mine")
    assert runlog.default_target() == "gs://mine/runs"
    with pytest.raises(ValueError):
        runlog.run_store("gs://")
