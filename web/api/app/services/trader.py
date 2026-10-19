"""Auto-trading status for the order book timeline, read from the trader's run records.

The web only reads: run records (``trader.runlog``) and the NYSE calendar. It never
loads broker settings or credentials, so an account without records is simply manual.
"""

from __future__ import annotations

from datetime import datetime

from trader.calendar import NY, Session, ny_now, upcoming_session
from trader.runlog import RunStore, run_store

RECENT = 10
LOOKBACK = 20  # records read per request (one download each on GCS)


def _session_payload(session: Session, now: datetime) -> dict:
    return {**session.to_dict(), "is_today": session.day == now.astimezone(NY).date()}


def _summary(record: dict) -> dict:
    return {key: record.get(key) for key in ("step", "session", "status", "message", "started_at", "forced")}


def trader_status(account_id: str, store: RunStore | None = None, now: datetime | None = None) -> dict:
    now = now or ny_now()
    session = upcoming_session(now)
    records = (store or run_store()).recent(account_id, limit=LOOKBACK)
    checks = [r for r in records if r.get("step") == "check"]
    # Today's check if it ran; otherwise the latest one that actually compared the ledger
    # (a "closed" run, on a weekend or outside the session's hours, says nothing about the account).
    check = (
        next((r for r in checks if r.get("session") == session.day.isoformat() and r.get("status") != "closed"), None)
        or next((r for r in checks if r.get("checks")), None)
        or (checks[0] if checks else None)
    )
    # A record that failed before loading the settings has no broker/auto_trade: skip it for those.
    described = next((r for r in records if "auto_trade" in r), None)
    return {
        "account": account_id,
        "now": now.isoformat(),
        "configured": bool(records),
        "auto_trade": described.get("auto_trade") if described else None,
        "broker": described.get("broker") if described else None,
        "session": _session_payload(session, now),
        "check": None if check is None else {
            **_summary(check),
            "current": check.get("session") == session.day.isoformat(),
            "checks": check.get("checks", []),
            "calendar": check.get("calendar"),
        },
        "recent": [_summary(r) for r in records[:RECENT]],
    }
