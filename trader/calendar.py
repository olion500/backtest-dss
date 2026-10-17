"""US trading days and closes: holidays and early (13:00) closes from the NYSE calendar.

A run trades only when the exchange calendar and the broker both say today is a
session; if they disagree the run stops (one of them is wrong, and guessing could
send orders on a closed day or skip a real one).
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

    @property
    def early_close(self) -> bool:
        return self.close is not None and self.close.hour < REGULAR_CLOSE_HOUR

    @property
    def close_order_cutoff(self) -> datetime | None:
        return self.close - CLOSE_ORDER_CUTOFF if self.close else None

    @property
    def stop_check_at(self) -> datetime | None:
        return self.close - STOP_CHECK_LEAD if self.close else None


def us_session(day: date) -> Session:
    calendar = _nyse()
    stamp = day.isoformat()
    if not calendar.is_session(stamp):
        return Session(day=day, is_open=False)
    close = calendar.session_close(stamp).tz_convert(NY).to_pydatetime()
    return Session(day=day, is_open=True, close=close)


def last_closed_session(day: date) -> date:
    """The latest session that closed before ``day``'s trading (the close the order book is built from)."""
    calendar = _nyse()
    stamp = day.isoformat()
    if calendar.is_session(stamp):
        return calendar.previous_session(stamp).date()
    return calendar.date_to_session(stamp, direction="previous").date()


def ny_now() -> datetime:
    return datetime.now(NY)
