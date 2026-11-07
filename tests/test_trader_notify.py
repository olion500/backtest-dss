"""Discord alerts: which runs speak, what they say, and that a dead webhook never breaks a run."""
import pytest
import requests

from trader import cli
from trader.notify import COLORS, build_embeds, send


def _record(status, account="main", step="orders", **extra):
    return {"account": account, "step": step, "session": "2026-11-06", "status": status,
            "message": f"msg {status}", "finished_at": "2026-11-06T16:10:05-05:00", **extra}


def _day(status="ok", account="main", **extra):
    values = dict(symbol="SOXL", prev_close="40.87", close="45.97", close_source="fill",
                  lots_closed=[{"buy_date": "2026-11-05", "qty": 59, "buy_price": "40.87", "kind": "익절", "pnl": "300.90"}],
                  bought_qty=0, position_qty="72", cash="31567.20")
    values.update(extra)
    return _record(status, account, "report", **values)


@pytest.mark.parametrize("step, status", [("orders", "sent"), ("orders", "ok"), ("orders", "declined"),
                                          ("stop", "sent"), ("stop", "closed"), ("report", "closed")])
def test_a_normal_run_says_nothing_until_the_report(step, status):
    assert build_embeds(step, [_record(status, step=step)]) == []


def test_halted_shows_each_failed_check_as_a_field():
    checks = [{"label": "SOXL 수량", "expected": "131주", "actual": "130주", "diff": "-1주", "ok": False},
              {"label": "평균 단가", "expected": "$1", "actual": "$2", "diff": "", "ok": None}]
    [embed] = build_embeds("orders", [_record("halted", checks=checks)])
    assert embed["title"] == "⛔ [main] 주문 멈춤" and embed["color"] == COLORS["halted"]
    assert "주문을 보내지 않았습니다." in embed["description"]
    assert embed["fields"] == [{"name": "✗ SOXL 수량", "value": "엔진 131주\n실제 130주 (-1주)", "inline": True}]


def test_partial_names_the_rejected_and_unsent_orders():
    sent = [{"request": {"side": "buy", "type": "loc", "qty": 59, "price": "41.37"}, "order_id": "1"},
            {"request": {"side": "sell", "type": "loc", "qty": 37, "price": "74.04"}, "error": "거절"}]
    [embed] = build_embeds("orders", [_record("partial", sent=sent, rejected=["1"])])
    values = {f["name"]: f["value"] for f in embed["fields"]}
    assert values["거절된 주문번호"] == "1" and values["접수 안 된 주문"] == "매도 LOC 37주 $74.04 — 거절"


def test_report_is_one_embed_with_a_field_per_account():
    records = [_day(), _day(account="연향", lots_closed=[], bought_qty=6, position_qty="19", cash="3100")]
    [embed] = build_embeds("report", records)
    assert embed["title"] == "📊 11/6 (금) 마감 — SOXL $45.97 (+12.5%)  ✅ 예상대로"
    assert embed["color"] == COLORS["ok"]
    assert embed["fields"] == [
        {"name": "[main]", "value": "익절 59주 +$300.90\n보유 72주\n예수금 $31,567", "inline": True},
        {"name": "[연향]", "value": "매수 6주\n보유 19주\n예수금 $3,100", "inline": True},
    ]
    assert "description" not in embed
    [nothing] = build_embeds("report", [_day(lots_closed=[], bought_qty=0, close_source="quote")])
    assert nothing["fields"][0]["value"].startswith("체결 없음") and "시간외 가격일 수 있음" in nothing["description"]


def test_a_mismatch_gets_its_own_embed_and_is_marked_in_the_summary():
    fills = [{"request": {"side": "sell", "type": "loc", "qty": 59, "price": "44.05"}, "expected_qty": 59,
              "actual_qty": "0", "fill_price": None, "ok": False}]
    detail, summary = build_embeds("report", [_day("mismatch", fills=fills), _day(account="sub")])
    assert detail["title"] == "⚠️ [main] 마감 예상과 다름"
    assert detail["fields"][0] == {"name": "매도 LOC 59주 $44.05",
                                   "value": "예상: 59주 체결 (종가 $45.97 ≥ 지정가)\n실제: 미체결", "inline": False}
    assert summary["title"].endswith("⚠️ main 예상과 다름") and summary["fields"][0]["name"] == "⚠️ [main]"


def test_a_lost_record_alerts_even_when_the_run_went_through():
    [kept] = build_embeds("orders", [_record("sent", store_error="ConnectionError: gcs")])
    assert kept["color"] == COLORS["partial"] and "로컬에만 남음" in kept["fields"][0]["name"]


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
    assert len(sent) == 1 and sent[0][0]["title"] == "⛔ [main] 주문 멈춤"
    cli.main(["orders", "--account", "main"])  # a person at the terminal: no alert
    assert len(sent) == 1


def test_scheduled_report_sends_the_summary(monkeypatch):
    sent = _patch_runs(monkeypatch, _record("sent"))
    monkeypatch.setattr(cli, "run_report", lambda account_id: dict(_day(), path="p"))
    assert cli.main(["report", "--all", "--auto"]) == 0
    assert sent[0][0]["title"].startswith("📊 11/6 (금) 마감")


def test_a_crash_alerts_and_still_raises(monkeypatch):
    sent = _patch_runs(monkeypatch, _record("sent"))

    def boom(account_id, approve):
        raise RuntimeError("expected ledger unavailable")

    monkeypatch.setattr(cli, "run_orders", boom)
    with pytest.raises(RuntimeError):
        cli.main(["orders", "--all", "--auto"])
    assert "RuntimeError: expected ledger unavailable" in sent[0][-1]["description"]
