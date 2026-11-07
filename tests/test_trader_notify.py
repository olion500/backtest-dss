"""Discord alerts: which runs speak, what they say, and that a dead webhook never breaks a run."""
import pytest
import requests

from trader import cli
from trader.notify import COLORS, run_embed, send


def _record(status, **extra):
    return {"account": "main", "step": "orders", "session": "2026-11-06", "status": status,
            "message": f"msg {status}", "path": "gs://b/runs/main/2026-11-06/094521-orders.json", **extra}


@pytest.mark.parametrize("status", ["ok", "closed", "declined"])
def test_quiet_statuses_say_nothing(status):
    assert run_embed("orders", _record(status)) is None


def test_halted_lists_the_failed_checks():
    checks = [{"label": "SOXL 수량", "expected": "131주", "actual": "130주", "diff": "-1주", "ok": False},
              {"label": "평균 단가", "expected": "$1", "actual": "$2", "diff": "", "ok": None}]
    embed = run_embed("orders", _record("halted", checks=checks))
    assert embed["color"] == COLORS["halted"] and "[main] 주문 halted" in embed["title"]
    assert "SOXL 수량: 엔진 131주 / 실제 130주 (-1주)" in embed["description"]
    assert "평균 단가" not in embed["description"]


def test_partial_names_the_rejected_and_unsent_orders():
    sent = [{"request": {"side": "buy", "type": "loc", "qty": 59, "price": "41.37"}, "order_id": "1"},
            {"request": {"side": "sell", "type": "loc", "qty": 37, "price": "74.04"}, "error": "거절"}]
    embed = run_embed("orders", _record("partial", sent=sent, rejected=["1"]))
    assert "거절 주문번호: 1" in embed["description"]
    assert "미접수: 매도 LOC 37주 $74.04 — 거절" in embed["description"]


def test_stop_sell_lists_its_orders_and_store_error_alerts_even_when_sent():
    stop = run_embed("stop", _record("sent", plan=[{"side": "sell", "type": "moc", "qty": 59, "price": None}]))
    assert stop["color"] == COLORS["sent"] and "매도 MOC 59주 시장가" in stop["description"]
    kept = run_embed("orders", _record("ok", store_error="ConnectionError: gcs"))
    assert kept is not None and "로컬에만 남음" in kept["description"]


class _Response:
    def __init__(self, code):
        self.status_code = code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("404 for url: https://discord.com/api/webhooks/1/SECRET", response=self)


def test_send_batches_by_ten_and_never_raises_or_leaks_the_url(capsys):
    posts = []

    def post(url, json, timeout):
        posts.append(len(json["embeds"]))
        return _Response(404 if len(posts) == 2 else 204)

    assert send([{"title": str(i)} for i in range(13)], webhook="https://hook/SECRET", post=post) is False
    assert posts == [10, 3]
    err = capsys.readouterr().err
    assert "HTTP 404" in err and "SECRET" not in err
    assert send([{"title": "x"}], webhook="", post=post) is False and posts == [10, 3]  # unset: no post


def _patch_runs(monkeypatch, record):
    monkeypatch.setattr(cli, "load_repo_env", lambda: None)
    monkeypatch.setattr(cli, "trader_account_ids", lambda: ["main"])
    monkeypatch.setattr(cli, "run_orders", lambda account_id, approve: dict(record, path="p"))
    sent = []
    monkeypatch.setattr(cli, "send", lambda embeds: sent.append(embeds))
    return sent


def test_only_scheduled_runs_alert(monkeypatch):
    sent = _patch_runs(monkeypatch, _record("halted"))
    assert cli.main(["orders", "--all", "--auto"]) == 2
    assert len(sent) == 1 and sent[0][0]["title"].endswith("주문 halted")
    cli.main(["orders", "--account", "main"])  # a person at the terminal: no alert
    assert len(sent) == 1


def test_a_crash_alerts_and_still_raises(monkeypatch):
    sent = _patch_runs(monkeypatch, _record("sent"))

    def boom(account_id, approve):
        raise RuntimeError("expected ledger unavailable")

    monkeypatch.setattr(cli, "run_orders", boom)
    with pytest.raises(RuntimeError):
        cli.main(["orders", "--all", "--auto"])
    assert "RuntimeError: expected ledger unavailable" in sent[0][-1]["description"]
