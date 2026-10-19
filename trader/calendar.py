"""US trading days and closes: holidays and early (13:00) closes from the NYSE calendar.

A run trades only during the regular session, and only when the exchange calendar
and the broker agree the market is trading right now; if they disagree the run stops
(one of them is wrong, and guessing could send orders on a closed day or skip a real one).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

NY = ZoneInfo("America/New_York")
REGULAR_CLOSE_HOUR = 16
# Meritz takes LOC/MOC orders, changes and cancels until 10 minutes before the
# close; the stop-loss check runs 5 minutes ahead of that.
CLOSE_ORDER_CUTOFF = timedelta(minutes=10)
STOP_CHECK_LEAD = timedelta(minutes=15)


@lru_cache(maxsize=1)
def _nyse():
    return xcals.get_calendar("XNYS")


@dataclass(frozen=True)
class Session:
    day: date
    is_open: bool
    close: datetime | None = None  # NY time
    open: datetime | None = None  # NY time

    @property
    def early_close(self) -> bool:
        return self.close is not None and self.close.hour < REGULAR_CLOSE_HOUR

    def trading_at(self, now: datetime) -> bool:
        """Whether the regular session is under way at ``now``."""
        return self.is_open and self.open <= now < self.close

    @property
    def close_order_cutoff(self) -> datetime | None:
        return self.close - CLOSE_ORDER_CUTOFF if self.close else None

    @property
    def stop_check_at(self) -> datetime | None:
        return self.close - STOP_CHECK_LEAD if self.close else None

    def to_dict(self) -> dict:
        def iso(at: datetime | None) -> str | None:
            return at.isoformat() if at else None

        return {
            "date": self.day.isoformat(),
            "is_open": self.is_open,
            "open": iso(self.open),
            "close": iso(self.close),
            "early_close": self.early_close,
            "stop_check_at": iso(self.stop_check_at),
            "order_cutoff": iso(self.close_order_cutoff),
        }


def us_session(day: date) -> Session:
    calendar = _nyse()
    stamp = day.isoformat()
    if not calendar.is_session(stamp):
        return Session(day=day, is_open=False)
    close = calendar.session_close(stamp).tz_convert(NY).to_pydatetime()
    open_ = calendar.session_open(stamp).tz_convert(NY).to_pydatetime()
    return Session(day=day, is_open=True, close=close, open=open_)


def last_closed_session(day: date) -> date:
    """The latest session that closed before ``day``'s trading (the close the order book is built from)."""
    calendar = _nyse()
    stamp = day.isoformat()
    if calendar.is_session(stamp):
        return calendar.previous_session(stamp).date()
    return calendar.date_to_session(stamp, direction="previous").date()


def upcoming_session(now: datetime) -> Session:
    """The session the order book trades next: today until its close, then the next one."""
    today = now.astimezone(NY).date()
    session = us_session(today)
    if session.is_open and now < session.close:
        return session
    calendar = _nyse()
    nxt = calendar.date_to_session(today.isoformat(), direction="next").date()
    if nxt == today:
        nxt = calendar.next_session(today.isoformat()).date()
    return us_session(nxt)


def ny_now() -> datetime:
    return datetime.now(NY)
