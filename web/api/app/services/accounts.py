"""Trading accounts for the order book (``config/accounts/<id>.json``).

Each account runs the shared ``config/strategy.json`` with its own name,
start date, starting cash and step orders. See :mod:`engines.config_store`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from engines.config_store import ACCOUNTS_DIR, load_accounts as read_accounts


class Account(BaseModel):
    id: str
    name: str = Field(min_length=1, max_length=40)
    order: int | None = None
    start_date: date
    initial_cash: float = Field(gt=0, le=100_000_000)
    spread_buy_levels: int = Field(default=5, ge=0, le=20)
    spread_buy_step: int = Field(default=1, ge=1, le=1000)
    spread_floor_pct: float = Field(default=0.0, ge=0, le=90)
    fee_pct: float = Field(default=0.0, ge=0, le=1)
    sec_fee_pct: float = Field(default=0.0, ge=0, le=1)


def load_accounts(accounts_dir: Path = ACCOUNTS_DIR) -> list[Account]:
    fallback_start = date(date.today().year - 4, 1, 1)
    accounts = []
    for raw in read_accounts(accounts_dir):
        try:
            accounts.append(Account(
                id=raw["id"],
                name=raw["name"],
                order=raw["order"],
                start_date=raw["start_date"] or fallback_start,
                initial_cash=raw["init_cash"],
                spread_buy_levels=raw["spread_buy_levels"],
                spread_buy_step=raw["spread_buy_step"],
                spread_floor_pct=raw["spread_floor_pct"],
                fee_pct=raw["fee_pct"],
                sec_fee_pct=raw["sec_fee_pct"],
            ))
        except ValidationError:
            continue  # a broken file should not hide the other accounts
    return accounts
