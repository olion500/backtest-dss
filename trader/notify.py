"""Discord alerts for the scheduled runs (``orders``/``stop``/``report --auto``, ``data --push``).

One webhook (``DISCORD_WEBHOOK_URL``, unset = no alerts). On a normal day the only
message is the report after the close: one embed, a field per account (lots closed
with P&L, shares bought, holdings, cash). Anything that needs a person comes at once
as an embed of its own: ``halted``, ``partial``, ``error``, a crash, a record kept only
locally, and a report ``mismatch`` (fills that differ from what the close says).
Sending never raises: a dead webhook must not stop or fail a trading run.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from decimal import Decimal

import requests

WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"
STEP_NAMES = {"orders": "주문", "stop": "손절", "data": "시세", "report": "마감", "vr": "VR 예약"}
STATUS_NAMES = {"error": "오류", "halted": "멈춤", "partial": "일부 실패", "mismatch": "예상과 다름"}
COLORS = {"error": 0xE74C3C, "halted": 0xE74C3C, "partial": 0xE67E22, "mismatch": 0xE67E22, "ok": 0x2ECC71}
MARKS = {"error": "🚨", "halted": "⛔", "partial": "⚠️", "mismatch": "⚠️"}
PROBLEMS = ("error", "halted", "partial", "mismatch")
WEEKDAYS = "월화수목금토일"
EMBEDS_PER_MESSAGE = 10  # Discord's limits
FIELDS_PER_EMBED = 25
FIELD_MAX = 1000
DESCRIPTION_MAX = 4000
TIMEOUT = 10


def build_embeds(step: str, records: list[dict]) -> list[dict]:
    """A step's alerts: one embed per run needing a person, then (report) the day's summary."""
    embeds = [_problem(step, record) for record in records if _needs_person(record)]
    if step == "vr":
        embeds += [_vr(record) for record in records if record.get("status") in ("sent", "declined") and record.get("cycle")]
    if step == "report":
        done = [record for record in records if record.get("status") in ("ok", "mismatch")]
        if done:
            embeds.append(_report(done))
    return embeds


def _needs_person(record: dict) -> bool:
    return record.get("status", "error") in PROBLEMS or bool(record.get("store_error"))


def _problem(step: str, record: dict) -> dict:
    status = record.get("status", "error")
    status = status if status in PROBLEMS else "partial"  # went through, but the record was not saved
    lines = [f"**{record.get('message', '')}**"]
    fields = []
    for check in record.get("checks", []):
        if check.get("ok") is False:
            fields.append(_field(f"✗ {check['label']}", f"엔진 {check['expected']}\n실제 {check['actual']} ({check['diff']})"))
    if record.get("rejected"):
        fields.append(_field("거절된 주문번호", ", ".join(map(str, record["rejected"])), inline=False))
    unsent = [f"{order_text(item['request'])} — {item['error']}" for item in record.get("sent", []) if "error" in item]
    if unsent:
        fields.append(_field("접수 안 된 주문", "\n".join(unsent), inline=False))
    close = record.get("close")
    for fill in record.get("fills", []):
        if not fill.get("ok"):
            fields.append(_field(order_text(fill["request"]), _fill_gap(fill, close), inline=False))
    for order in record.get("stray", []):
        fields.append(_field("자동매매가 안 낸 체결", f"주문번호 {order['order_id']} {_side(order)} {_qty(order['filled_qty'])}주", inline=False))
    if record.get("warnings"):
        fields.append(_field("손절 판단과 종가가 다름", "\n".join(record["warnings"]), inline=False))
    if record.get("store_error"):
        fields.append(_field("기록 저장 실패 (로컬에만 남음)", record["store_error"], inline=False))
    if status == "halted" and step in ("orders", "stop"):
        lines.append("주문을 보내지 않았습니다.")
    if status == "mismatch":
        lines.append("내일 09:45 대조에서 멈출 수 있습니다. 앱에서 체결 내역을 확인하세요.")
    else:
        lines.append("확인이 필요합니다.")
    return _embed(f"{MARKS[status]} {_who(record)} {STEP_NAMES.get(step, step)} {STATUS_NAMES[status]}",
                  "\n".join(lines), COLORS[status], fields, record)


def _fill_gap(fill: dict, close) -> str:
    request = fill["request"]
    want = fill["expected_qty"]
    if request.get("type") == "moc" or request.get("price") is None:
        why = "시장가는 항상 체결"
    elif request.get("side") == "buy":
        why = f"종가 ${close} {'≤' if want else '>'} 지정가"
    else:
        why = f"종가 ${close} {'≥' if want else '<'} 지정가"
    expected = f"{want}주 체결" if want else "미체결"
    if fill.get("actual_qty") is None:
        actual = "증권사 목록에 없음"
    else:
        got = Decimal(str(fill["actual_qty"]))
        actual = (f"{_qty(got)}주 체결" + (f" @ ${fill['fill_price']}" if fill.get("fill_price") else "")) if got else "미체결"
    return f"예상: {expected} ({why})\n실제: {actual}"


def _report(records: list[dict]) -> dict:
    first = records[0]
    close = Decimal(str(first["close"]))
    prev = first.get("prev_close")
    change = f" ({(close / Decimal(str(prev)) - 1) * 100:+.1f}%)" if prev else ""
    off = [record["account"] for record in records if record["status"] == "mismatch"]
    verdict = f"⚠️ {', '.join(off)} 예상과 다름" if off else "✅ 예상대로"
    fields = [_field(f"{'⚠️ ' if record['status'] == 'mismatch' else ''}{_who(record)}", _account_day(record)) for record in records]
    note = "종가는 증권사 현재가 기준 (오늘 체결 없음, 시간외 가격일 수 있음)" if first.get("close_source") == "quote" else ""
    return _embed(f"📊 {_day(first.get('session'))} 마감 — {first.get('symbol', '')} ${close}{change}  {verdict}",
                  note, COLORS["mismatch" if off else "ok"], fields, first)


def _vr(record: dict) -> dict:
    """A new VR cycle: where it starts (broker + external) and what was reserved."""
    cycle = record["cycle"]
    balance, external = record.get("balance") or {}, record.get("external") or {}
    lines = [f"**{record.get('message', '')}**",
             f"V ${float(cycle['v']):,.2f} · Pool ${float(cycle['pool_start']):,.2f} · 보유 {cycle['shares_start']}주"]
    if balance:
        lines.append(f"시작 잔고: 증권사 {balance['shares']}주 ${Decimal(str(balance['cash'])):,.2f}"
                     f" + 외부 {external.get('shares', 0)}주 ${float(external.get('pool', 0)):,.2f}")
    fields = []
    sent = [order_text(item["request"]) for item in record.get("sent", []) if "reservation_id" in item]
    if sent:
        fields.append(_field(f"예약 {len(sent)}건", "\n".join(sent), inline=False))
    if record.get("cash_short"):
        fields.append(_field("⚠️ 예수금 부족", f"매수가 전부 체결되면 이 계좌 예수금이 ${Decimal(record['cash_short']):,.2f} 모자랍니다. 체결 전에 채우세요.", inline=False))
    color = COLORS["ok"] if record["status"] != "declined" else COLORS["partial"]
    return _embed(f"🔁 {_who(record)} 사이클 {cycle['index']} ({cycle['start']} ~ {cycle['end']})", "\n".join(lines), color, fields, record)


def _account_day(record: dict) -> str:
    lines = []
    for kind in ("익절", "손절", "만기"):
        lots = [lot for lot in record.get("lots_closed", []) if lot["kind"] == kind]
        if lots:
            qty = sum(int(lot["qty"]) for lot in lots)
            pnl = sum(Decimal(str(lot["pnl"])) for lot in lots)
            lines.append(f"{kind} {qty}주 {_signed(pnl)}")
    if record.get("bought_qty"):
        lines.append(f"매수 {record['bought_qty']}주")
    if not lines:
        lines.append("체결 없음")
    lines.append(f"보유 {_qty(record.get('position_qty', 0))}주")
    if record.get("cash") is not None:
        lines.append(f"예수금 ${Decimal(str(record['cash'])):,.0f}")
    return "\n".join(lines)


def crash_embed(step: str, exc: BaseException) -> dict:
    return _embed(f"🚨 {STEP_NAMES.get(step, step)} 실행 중 예외", f"`{type(exc).__name__}: {exc}`\n확인이 필요합니다.",
                  COLORS["error"], [], {})


def order_text(order: dict) -> str:
    """A sent order as a record stores it (plain strings)."""
    price = f"${order['price']}" if order.get("price") is not None else "시장가"
    return f"{_side(order)} {str(order.get('type', '')).upper()} {order.get('qty')}주 {price}"


def _embed(title: str, description: str, color: int, fields: list[dict], record: dict) -> dict:
    embed = {"title": title[:256], "color": color}
    if description:
        embed["description"] = description[:DESCRIPTION_MAX]
    if fields:
        embed["fields"] = fields[:FIELDS_PER_EMBED]
    if record.get("finished_at") or record.get("started_at"):
        embed["timestamp"] = record.get("finished_at") or record.get("started_at")  # shown in the reader's time zone
    if record.get("session"):
        embed["footer"] = {"text": f"세션 {_day(record['session'])} (뉴욕)"}
    return embed


def _field(name: str, value: str, inline: bool = True) -> dict:
    return {"name": name[:256], "value": (value or "-")[:FIELD_MAX], "inline": inline}


def _who(record: dict) -> str:
    return f"[{record['account']}]" if record.get("account") else "[data]"


def _side(order: dict) -> str:
    return "매도" if str(order.get("side")) == "sell" else "매수"


def _qty(value) -> str:
    return f"{Decimal(str(value)).normalize():f}"


def _signed(value: Decimal) -> str:
    return f"{'+' if value >= 0 else '−'}${abs(value):,.2f}"


def _day(session: str | None) -> str:
    if not session:
        return ""
    day = date.fromisoformat(session)
    return f"{day.month}/{day.day} ({WEEKDAYS[day.weekday()]})"


def send(embeds: list[dict], webhook: str | None = None, post=requests.post) -> bool:
    """Post the embeds (in messages of up to ten). ``False`` when unset or any post failed."""
    embeds = [embed for embed in embeds if embed]
    url = (webhook if webhook is not None else os.environ.get(WEBHOOK_ENV, "")).strip()
    if not embeds or not url:
        return False
    ok = True
    for start in range(0, len(embeds), EMBEDS_PER_MESSAGE):
        try:
            response = post(url, json={"embeds": embeds[start:start + EMBEDS_PER_MESSAGE]}, timeout=TIMEOUT)
            response.raise_for_status()
        except Exception as exc:  # the alert is a side channel: report it, never break the run
            # Not str(exc): requests puts the URL in it, and the URL holds the webhook token.
            code = getattr(getattr(exc, "response", None), "status_code", None)
            print(f"! 디스코드 알림 실패: {type(exc).__name__}{f' HTTP {code}' if code else ''}", file=sys.stderr)
            ok = False
    return ok
