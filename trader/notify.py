"""Discord alerts for the scheduled runs (``orders --auto``, ``stop --auto``, ``data --push``).

One webhook (``DISCORD_WEBHOOK_URL``, unset = no alerts). A run that needs a person
(``halted``, ``partial``, ``error``, a record kept only locally, a crash) is red or orange;
orders sent and stop-loss sells are a short green line. ``closed``, ``declined`` and ``ok``
say nothing. Sending never raises: a dead webhook must not stop or fail a trading run.
"""

from __future__ import annotations

import os
import sys

import requests

WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"
STEP_NAMES = {"orders": "주문", "stop": "손절", "data": "시세"}
COLORS = {"error": 0xE74C3C, "halted": 0xE74C3C, "partial": 0xE67E22, "sent": 0x2ECC71}
TITLE_MARKS = {"error": "🚨", "halted": "⛔", "partial": "⚠️", "sent": "✅"}
EMBEDS_PER_MESSAGE = 10  # Discord's limit
DESCRIPTION_MAX = 4000   # Discord allows 4096
TIMEOUT = 10


def run_embed(step: str, record: dict) -> dict | None:
    """The embed for one run record, or ``None`` when the run needs no word."""
    status = record.get("status", "error")
    store_error = record.get("store_error")
    if status not in COLORS and not store_error:
        return None
    name = record.get("account") or "data"
    lines = [record.get("message", "")]
    for check in record.get("checks", []):
        if check.get("ok") is False:
            lines.append(f"✗ {check['label']}: 엔진 {check['expected']} / 실제 {check['actual']} ({check['diff']})")
    if record.get("rejected"):
        lines.append(f"거절 주문번호: {', '.join(map(str, record['rejected']))}")
    for item in record.get("sent", []):
        if "error" in item:
            lines.append(f"✗ 미접수: {_order(item['request'])} — {item['error']}")
    if store_error:
        lines.append(f"기록 저장 실패 (로컬에만 남음): {store_error}")
    if status == "sent" and step == "stop":
        lines.extend(_order(order) for order in record.get("plan", []))
    color = COLORS.get(status, COLORS["partial"])
    return {
        "title": f"{TITLE_MARKS.get(status, '⚠️')} [{name}] {STEP_NAMES.get(step, step)} {status}",
        "description": "\n".join(line for line in lines if line)[:DESCRIPTION_MAX],
        "color": color,
        "footer": {"text": f"{record.get('session', '')} {record.get('path', '')}".strip()},
    }


def crash_embed(step: str, exc: BaseException) -> dict:
    return {
        "title": f"🚨 {STEP_NAMES.get(step, step)} 실행 중 예외",
        "description": f"{type(exc).__name__}: {exc}"[:DESCRIPTION_MAX],
        "color": COLORS["error"],
    }


def _order(order: dict) -> str:
    side = "매도" if order.get("side") == "sell" else "매수"
    price = f"${order['price']}" if order.get("price") is not None else "시장가"
    return f"{side} {str(order.get('type', '')).upper()} {order.get('qty')}주 {price}"


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
