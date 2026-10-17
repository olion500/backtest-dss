"""What the account should hold according to the engine: the same order book the web shows.

The order book is the source of truth for lots (each has its own TP/SL); the
broker's balance only confirms it. Built through the web's order-book service so
the trader can never compute a different sheet than the one on screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from web.api.app.services.accounts import load_accounts
from web.api.app.services.backtest import default_request, run_order_book_view
from web.api.app.services.market_data import market_today


@dataclass(frozen=True)
class Lot:
    buy_date: str
    qty: int
    buy_price: Decimal
    tp: Decimal | None
    sl: Decimal | None


@dataclass(frozen=True)
class Expected:
    account_id: str
    symbol: str
    as_of: str             # last session the engine has a close for
    mode: str
    cash: Decimal
    qty: int
    prev_close: Decimal | None
    tranche_budget: Decimal | None
    lots: tuple[Lot, ...]
    orders: tuple[dict, ...]  # the order book rows, as the web shows them


class UnknownAccount(LookupError):
    pass


def _dec(value) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def expected_state(account_id: str, today: date | None = None) -> Expected:
    account = next((item for item in load_accounts() if item.id == account_id), None)
    if account is None:
        raise UnknownAccount(f"config/accounts/{account_id}.json 계좌가 없습니다")
    end = today or market_today()
    request = default_request().model_copy(update={
        "start_date": account.start_date,
        "end_date": max(end, account.start_date),
        "initial_cash": account.initial_cash,
        "spread_buy_levels": account.spread_buy_levels,
        "spread_buy_step": account.spread_buy_step,
        "spread_floor_pct": account.spread_floor_pct,
        "fee_pct": account.fee_pct,
        "sec_fee_pct": account.sec_fee_pct,
    })
    view = run_order_book_view(request)
    book = view["order_book"]
    state = book["state"]
    lots = tuple(
        Lot(
            buy_date=str(row.get("매수일자") or ""),
            qty=int(row.get("매수수량") or 0),
            buy_price=Decimal(str(row.get("매수체결가") or 0)),
            tp=_dec(row.get("TP목표가")),
            sl=_dec(row.get("SL목표가")),
        )
        for row in view["trade_log"]["rows"]
        if row.get("상태") != "완료"
    )
    return Expected(
        account_id=account_id,
        symbol=request.strategy.target_ticker,
        as_of=str(state["last_date"]),
        mode=str(state["current_mode"]),
        cash=Decimal(str(state["current_cash"])),
        qty=int(state["current_position_qty"]),
        prev_close=_dec(state.get("prev_close")),
        tranche_budget=_dec(state.get("tranche_budget")),
        lots=lots,
        orders=tuple(book["orders"]["rows"]),
    )
