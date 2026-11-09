"""VR (value rebalancing) accounts from ``config/vr/<id>.json`` and their current cycle.

The whole state is rebuilt from the account file and daily prices on every request
(fills are estimated, see :mod:`engines.vr_engine`). A cycle the trader saw start
(``python -m trader vr --auto``) starts from the broker's balance instead: the broker
account's shares and USD cash read before its first session, plus ``external`` (shares
and cash held elsewhere, never traded), kept in the run store as ``_state/vr-<id>/actuals.json``.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, model_validator

from engines.config_store import VR_DIR, load_vr_accounts
from engines.vr_engine import CYCLE_DAYS, Cycle, VRConfig, deposit_for, g_for, next_v, run_vr
from web.api.app.serializers import json_value
from web.api.app.services.market_data import market_data_client, market_today

PriceProvider = Callable[[str, date, date], pd.DataFrame]
ACTUALS = "actuals.json"
log = logging.getLogger(__name__)


def record_id(vr_id: str) -> str:
    """The VR account's name in the run store (records and state)."""
    return f"vr-{vr_id}"


class DepositChange(BaseModel):
    from_: date = Field(alias="from")
    deposit: float


class Adjustment(BaseModel):
    date: date
    shares: int | None = Field(default=None, ge=0)
    pool: float | None = None
    v: float | None = Field(default=None, gt=0)


class External(BaseModel):
    """Shares and cash (USD) of this VR held outside the broker account, from ``from`` on."""

    from_: date = Field(alias="from")
    shares: int = Field(default=0, ge=0)
    pool: float = Field(default=0.0, ge=0)


class VRAccount(BaseModel):
    id: str
    name: str = Field(min_length=1, max_length=40)
    order: int | None = None
    ticker: str = "TQQQ"
    start_date: date
    origin_date: date | None = None  # VR 시작일 (시트 시작일): 주차와 G 증가를 여기서부터 셈
    start_shares: int = Field(ge=0)
    start_pool: float = Field(ge=0)
    start_v: float | None = Field(default=None, gt=0)
    principal: float | None = Field(default=None, ge=0)
    deposit: float = 0.0
    deposit_changes: list[DepositChange] = []
    g: float = Field(default=10, gt=0)
    g_step_cycles: int = Field(default=26, ge=0)
    g_step: float = 1.0
    advanced: bool = True
    band_pct: float = Field(default=15, gt=0, lt=100)
    pool_limit_pct: float = Field(default=75, gt=0, le=100)
    order_type: Literal["limit", "loc"] = "limit"
    buy_unit: int = Field(default=8, ge=1, le=1000)
    sell_unit: int = Field(default=1, ge=1, le=1000)
    sell_cap_pct: float = Field(default=40, gt=0, le=1000)
    adjustments: list[Adjustment] = []
    external: list[External] = []

    @model_validator(mode="before")
    @classmethod
    def origin_from_start_week(cls, data):
        """Older files gave the first cycle's week number instead of the VR start date."""
        if isinstance(data, dict) and data.get("start_week") and not data.get("origin_date"):
            data = dict(data)
            data["origin_date"] = date.fromisoformat(str(data["start_date"])) - timedelta(weeks=int(data["start_week"]))
        return data

    def external_at(self, day: date) -> External | None:
        """The ``external`` entry in force on ``day`` (the latest one from before it)."""
        return next((item for item in sorted(self.external, key=lambda e: e.from_, reverse=True) if item.from_ <= day), None)

    def actuals(self, balances: dict) -> dict:
        """Engine ``actuals`` from the broker balances by cycle start: broker + external."""
        out = {}
        for start, balance in (balances or {}).items():
            day = date.fromisoformat(start)
            ext = self.external_at(day)
            out[day] = {
                "shares": int(balance["shares"]) + (ext.shares if ext else 0),
                "pool": round(float(Decimal(str(balance["cash"]))) + (ext.pool if ext else 0.0), 2),
            }
        return out

    def engine_config(self, balances: dict | None = None) -> VRConfig:
        fields = self.model_dump(exclude={"id", "name", "order", "ticker", "deposit_changes", "adjustments", "external"})
        return VRConfig(
            **fields,
            deposit_changes=[{"from": item.from_, "deposit": item.deposit} for item in self.deposit_changes],
            adjustments=[item.model_dump() for item in self.adjustments],
            actuals=self.actuals(balances or {}),
        )


def load_balances(vr_id: str, store=None) -> dict:
    """Broker balances saved at each cycle start: {cycle start: {shares, cash, …}}. Empty when unreachable."""
    try:
        if store is None:
            from trader.runlog import run_store

            store = run_store()
        return store.read_state(record_id(vr_id), ACTUALS).get("cycles", {})
    except Exception as exc:  # no bucket access (local dev): estimates only, never a broken page
        log.warning("VR 실제 잔고를 못 읽음: %s", exc)
        return {}


def load_vr_account_models(vr_dir: Path = VR_DIR) -> list[VRAccount]:
    accounts = []
    for raw in load_vr_accounts(vr_dir):
        try:
            accounts.append(VRAccount.model_validate(raw))
        except ValidationError:
            continue  # a broken file should not hide the other accounts
    return accounts


def _order_rows(cycle: Cycle) -> list[dict]:
    rows = []
    for order in cycle.orders:
        rows.append({
            "side": order.side,
            "price": order.price,
            "qty": order.qty,
            "filled_on": order.filled_on,
            "fill_price": order.fill_price,
        })
    return rows


def _cycle_row(cycle: Cycle) -> dict:
    bought = sum(o.qty for o in cycle.orders if o.side == "buy" and o.filled_on)
    sold = sum(o.qty for o in cycle.orders if o.side == "sell" and o.filled_on)
    total = cycle.total
    return {
        "index": cycle.index,
        "week": cycle.week,
        "start": cycle.start,
        "end": cycle.end,
        "g": cycle.g,
        "v": cycle.v,
        "v_min": round(cycle.v_min, 2),
        "v_max": round(cycle.v_max, 2),
        "deposit": cycle.deposit,
        "shares_start": cycle.shares_start,
        "pool_start": round(cycle.pool_start, 2),
        "bought": bought,
        "sold": sold,
        "shares": cycle.shares,
        "close": cycle.close,
        "close_date": cycle.close_date,
        "actual": cycle.actual,
        "stock_value": round(cycle.stock_value, 2) if cycle.stock_value is not None else None,
        "pool": cycle.pool,
        "total": round(total, 2) if total is not None else None,
        "principal": round(cycle.principal, 2),
        "return_pct": round((total / cycle.principal - 1) * 100, 2) if total and cycle.principal else None,
    }


def _holding_cost(account: VRAccount, cycles: list[Cycle]) -> float:
    """Total cost of the shares held now: moving average, sells leave the average unchanged.

    The starting shares cost what was paid in apart from the starting Pool (the
    sheet's 체결금액), else their value at the first V. A cycle that starts with a
    different holding than the fills imply (an adjustment) keeps the average price.
    """
    first = cycles[0]
    if account.principal is not None:
        cost = max(account.principal - account.start_pool, 0.0)
    else:
        cost = first.v
    shares = first.shares_start
    for cycle in cycles:
        if cycle.shares_start != shares:
            cost = cost / shares * cycle.shares_start if shares else 0.0
            shares = cycle.shares_start
        for order in sorted((o for o in cycle.orders if o.filled_on), key=lambda o: o.filled_on):
            if order.side == "buy":
                cost += order.qty * order.fill_price
                shares += order.qty
            else:
                cost -= cost / shares * order.qty if shares else 0.0
                shares -= order.qty
    return cost


def _outlook(account: VRAccount, config: VRConfig, cycle: Cycle, as_of: date) -> dict:
    """Where the current cycle stands and what the next one would look like at the last close."""
    open_buys = [o.price for o in cycle.orders if o.side == "buy" and not o.filled_on]
    open_sells = [o.price for o in cycle.orders if o.side == "sell" and not o.filled_on]
    budget = cycle.pool_start * config.pool_limit_pct / 100
    spent = sum(o.qty * o.fill_price for o in cycle.orders if o.side == "buy" and o.filled_on)
    next_start = cycle.end + timedelta(days=1)
    deposit = deposit_for(config, next_start)
    stock = cycle.shares * cycle.close if cycle.close is not None else None
    return {
        "day": min(max((as_of - cycle.start).days + 1, 1), CYCLE_DAYS),
        "cycle_days": CYCLE_DAYS,
        "buy_line": max(open_buys) if open_buys else None,
        "sell_line": min(open_sells) if open_sells else None,
        "buy_budget": round(budget, 2),
        "buy_budget_left": round(budget - spent, 2),
        "next_start": next_start,
        "next_deposit": deposit,
        "next_g": g_for(config, cycle.index + 1),
        "next_v": round(next_v(cycle.v, cycle.pool, stock, g_for(config, cycle.index), config.advanced, deposit), 2)
        if stock is not None else None,
    }


def run_vr_view(
    account: VRAccount,
    provider: PriceProvider | None = None,
    today: date | None = None,
    balances: dict | None = None,
) -> dict:
    today = today or market_today()
    # An account starting later still shows its first cycle's orders.
    as_of = max(today, account.start_date)
    download = provider or market_data_client.download
    prices = download(account.ticker, account.start_date - timedelta(days=10), as_of + timedelta(days=1))
    config = account.engine_config(load_balances(account.id) if balances is None else balances)
    cycles = run_vr(prices, config, as_of)
    current = cycles[-1]
    cost = _holding_cost(account, cycles)
    return json_value({
        "account": account.model_dump(mode="json", by_alias=True),
        "today": today,
        "current": _cycle_row(current),
        "orders": _order_rows(current),
        "outlook": _outlook(account, config, current, today),
        "holding": {
            "cost_basis": round(cost, 2),
            "avg_cost": round(cost / current.shares, 4) if current.shares else None,
            "unrealized": round(current.stock_value - cost, 2) if current.stock_value is not None else None,
        },
        "daily": [
            {**day, "stock_value": round(day["shares"] * day["close"], 2), "cycle": cycle.index,
             "principal": round(cycle.principal, 2),
             "v": cycle.v, "v_min": round(cycle.v_min, 2), "v_max": round(cycle.v_max, 2)}
            for cycle in cycles for day in cycle.days
        ],
        "cycles": [_cycle_row(cycle) for cycle in cycles],
        "fills": [
            {"cycle": cycle.index, "date": order.filled_on, "side": order.side, "qty": order.qty, "price": order.fill_price}
            for cycle in cycles for order in cycle.orders if order.filled_on
        ],
    })
