"""Broker-neutral types: every broker adapter speaks these, the rest of ``trader`` never sees a broker's API.

An adapter turns its broker's codes (exchange ids, order type codes, response
codes) into these types and back. Adding a broker means one new module that
implements :class:`Broker`, plus an entry in ``trader.broker.BROKERS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Mapping, Protocol, runtime_checkable


class BrokerError(RuntimeError):
    """A broker call failed. ``code`` is the broker's own code when it sent one."""

    def __init__(self, message: str, code: str | None = None, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LOC = "loc"      # limit on close
    MOC = "moc"      # market on close
    LIMIT = "limit"
    MARKET = "market"


class OrderState(str, Enum):
    OPEN = "open"            # accepted, not (fully) filled yet
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: Decimal
    sellable_qty: Decimal
    avg_price: Decimal
    last_price: Decimal | None = None


@dataclass(frozen=True)
class Cash:
    currency: str
    settled: Decimal    # 예수금: cash already settled
    estimated: Decimal  # 추정 예수금: after pending settlements (what the engine's cash means)


@dataclass(frozen=True)
class Quote:
    symbol: str
    last: Decimal
    prev_close: Decimal | None
    at: datetime


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Side
    type: OrderType
    qty: int
    price: Decimal | None = None  # None for MOC / MARKET


@dataclass(frozen=True)
class Order:
    order_id: str
    symbol: str
    side: Side
    type: OrderType
    qty: Decimal
    price: Decimal | None
    filled_qty: Decimal
    fill_price: Decimal | None
    state: OrderState
    reason: str = ""  # rejection reason, when the broker gives one


@runtime_checkable
class Broker(Protocol):
    """One brokerage account. Quantities and prices are in the symbol's trading currency."""

    name: str

    def positions(self) -> list[Position]: ...

    def cash(self, currency: str = "USD") -> Cash: ...

    def quote(self, symbol: str) -> Quote: ...

    def next_session(self) -> date:
        """The broker's US trading day: today when the market trades today, else the next one."""
        ...

    def orders_today(self) -> list[Order]: ...

    def submit(self, order: OrderRequest) -> Order: ...

    def cancel(self, order_id: str) -> None: ...


Credentials = Mapping[str, str]
