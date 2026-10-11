"""Value Rebalancing (라오어 VR 5.0, 실력공식) — the rules of the user's VR Google Sheet.

See ``docs/vr_sheet.md``. Every 2 weeks (a *cycle*) the target value V moves:

    V' = V + P/G + (E − V) / (2√G) + deposit         (실력공식; without the last term: 기본공식)

where P is the pool (cash) and E the stock value at the previous close. The band is
V × (1 ± band). At the start of each cycle a buy ladder and a sell ladder are placed for the
whole two weeks (예약주문):

- buy: one share at V최소/Q, V최소/(Q+1), …, while the running cost stays within
  ``pool_limit`` of the pool at the cycle start; placed as orders of ``buy_unit`` shares.
- sell: one share at V최대/Q, V최대/(Q−1), …, up to ``sell_cap`` above the last close;
  placed as orders of ``sell_unit`` shares at the chunk's highest level.

Fills are estimated from daily bars: a limit order fills on the first day the low (buy) or
high (sell) reaches it, at the open if the price gaps through; a LOC order fills at the close.
Sale proceeds go to the pool but are not spent by this cycle's buy orders.

Everything is derived from the config and prices, so the state can be rebuilt any time.
``adjustments`` replace shares / pool / V at a cycle start when the real account drifted from
the estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from engines.dongpa_engine import normalize_ohlcv

CYCLE_DAYS = 14


@dataclass
class VRConfig:
    start_date: date              # first cycle's start; later cycles every 14 days
    start_shares: int
    start_pool: float             # pool at the first cycle's start, this cycle's deposit included
    start_v: float | None = None  # None → stock value at the close before start_date
    principal: float | None = None  # money put in up to start_date; None → stock value + pool
    deposit: float = 0.0          # per cycle from the second cycle on (+적립 / −인출)
    deposit_changes: list[dict] = field(default_factory=list)  # [{"from": date, "deposit": x}]
    g: float = 10.0
    g_step_cycles: int = 26       # G grows by g_step every this many cycles (0 = never)
    g_step: float = 1.0
    advanced: bool = True         # 실력공식
    band_pct: float = 15.0
    pool_limit_pct: float = 75.0
    order_type: str = "limit"     # "limit" | "loc"
    buy_unit: int = 8
    sell_unit: int = 1
    sell_cap_pct: float = 40.0
    adjustments: list[dict] = field(default_factory=list)  # [{"date": date, "shares"?, "pool"?, "v"?}]


@dataclass
class Order:
    side: str      # "buy" | "sell"
    price: float
    qty: int
    filled_on: date | None = None
    fill_price: float | None = None


@dataclass
class Cycle:
    index: int
    start: date
    end: date
    g: float
    v: float
    v_min: float
    v_max: float
    deposit: float
    principal: float
    shares_start: int
    pool_start: float
    orders: list[Order]
    shares: int = 0
    pool: float = 0.0
    close: float | None = None
    close_date: date | None = None
    days: list[dict] = field(default_factory=list)  # {date, close, shares, pool} after each session

    @property
    def stock_value(self) -> float | None:
        return None if self.close is None else self.shares * self.close

    @property
    def total(self) -> float | None:
        value = self.stock_value
        return None if value is None else value + self.pool


def _g(config: VRConfig, index: int) -> float:
    steps = index // config.g_step_cycles if config.g_step_cycles > 0 else 0
    return config.g + steps * config.g_step


def _deposit(config: VRConfig, start: date) -> float:
    amount = config.deposit
    for change in sorted(config.deposit_changes, key=lambda item: item["from"]):
        if change["from"] <= start:
            amount = change["deposit"]
    return amount


def _adjustment(config: VRConfig, start: date, end: date) -> dict:
    """Merged adjustments dated within [start, end] (applied at that cycle's start)."""
    merged: dict = {}
    for item in sorted(config.adjustments, key=lambda adj: adj["date"]):
        if start <= item["date"] <= end:
            merged.update({key: item[key] for key in ("shares", "pool", "v") if item.get(key) is not None})
    return merged


def deposit_for(config: VRConfig, start: date) -> float:
    """The deposit added at the cycle starting on ``start`` (not the first cycle)."""
    return _deposit(config, start)


def g_for(config: VRConfig, index: int) -> float:
    return _g(config, index)


def next_v(v: float, pool: float, stock_value: float, g: float, advanced: bool, deposit: float) -> float:
    growth = pool / g + ((stock_value - v) / (2 * math.sqrt(g)) if advanced else 0.0)
    return max(v + growth + deposit, 0.0)


def build_orders(shares: int, v: float, pool: float, last_close: float, config: VRConfig) -> list[Order]:
    """This cycle's buy and sell ladders (see module docstring)."""
    v_min = v * (1 - config.band_pct / 100)
    v_max = v * (1 + config.band_pct / 100)
    orders: list[Order] = []

    budget = pool * config.pool_limit_pct / 100
    cost, level, chunk = 0.0, 0, []
    while shares + level > 0:
        price = v_min / (shares + level)
        if cost + price > budget:
            break
        cost += price
        chunk.append(price)
        level += 1
        if len(chunk) == config.buy_unit:
            orders.append(Order("buy", round(chunk[-1], 2), len(chunk)))
            chunk = []
    if chunk:
        orders.append(Order("buy", round(chunk[-1], 2), len(chunk)))

    cap = last_close * (1 + config.sell_cap_pct / 100)
    level, chunk = 0, []
    while shares - level > 0:
        price = v_max / (shares - level)
        if price > cap:
            break
        chunk.append(price)
        level += 1
        if len(chunk) == config.sell_unit:
            orders.append(Order("sell", round(chunk[-1], 2), len(chunk)))
            chunk = []
    if chunk:
        orders.append(Order("sell", round(chunk[-1], 2), len(chunk)))
    return orders


def _fill(order: Order, bar, order_type: str) -> float | None:
    if order_type == "loc":
        close = float(bar["Close"])
        hit = close <= order.price if order.side == "buy" else close >= order.price
        return close if hit else None
    if order.side == "buy" and float(bar["Low"]) <= order.price:
        return min(order.price, float(bar["Open"]))
    if order.side == "sell" and float(bar["High"]) >= order.price:
        return max(order.price, float(bar["Open"]))
    return None


def run_vr(prices: pd.DataFrame, config: VRConfig, today: date) -> list[Cycle]:
    """Cycles from ``config.start_date`` through the one containing ``today``.

    ``prices`` holds daily Open/High/Low/Close bars, at least from a few sessions before
    ``start_date``. Bars after ``today`` are ignored.
    """
    prices = normalize_ohlcv(prices)
    bars = prices[(prices.index <= pd.Timestamp(today)) & prices["Close"].notna()]
    before = bars[bars.index < pd.Timestamp(config.start_date)]
    if before.empty:
        raise ValueError(f"{config.start_date} 이전 종가가 없습니다.")
    last_close = round(float(before["Close"].iloc[-1]), 2)
    last_date = before.index[-1].date()

    shares, pool = config.start_shares, config.start_pool
    v = config.start_v if config.start_v is not None else shares * last_close
    principal = config.principal if config.principal is not None else shares * last_close + pool
    cycles: list[Cycle] = []
    start, index = config.start_date, 0
    while start <= today:
        end = start + timedelta(days=CYCLE_DAYS - 1)
        deposit = 0.0
        g = _g(config, index)
        adjust = _adjustment(config, start, end)
        shares = int(adjust.get("shares", shares))
        pool = float(adjust.get("pool", pool))
        if cycles:
            prev = cycles[-1]
            deposit = _deposit(config, start)
            v = next_v(prev.v, pool, shares * last_close, _g(config, index - 1), config.advanced, deposit)
            pool += deposit
            principal += deposit
        v = float(adjust.get("v", v))
        cycle = Cycle(
            index=index, start=start, end=end, g=g, v=round(v, 2),
            v_min=v * (1 - config.band_pct / 100), v_max=v * (1 + config.band_pct / 100),
            deposit=deposit, principal=principal, shares_start=shares, pool_start=pool,
            orders=build_orders(shares, v, pool, last_close, config),
        )
        for day, bar in bars[(bars.index >= pd.Timestamp(start)) & (bars.index <= pd.Timestamp(end))].iterrows():
            for order in cycle.orders:
                if order.filled_on is not None:
                    continue
                price = _fill(order, bar, config.order_type)
                if price is None:
                    continue
                order.filled_on, order.fill_price = day.date(), round(price, 2)
                if order.side == "buy":
                    shares += order.qty
                    pool -= order.qty * order.fill_price
                else:
                    shares -= order.qty
                    pool += order.qty * order.fill_price
            last_close, last_date = round(float(bar["Close"]), 2), day.date()
            cycle.close, cycle.close_date = last_close, last_date
            cycle.days.append({"date": day.date(), "close": last_close, "shares": shares, "pool": round(pool, 2)})
        cycle.shares, cycle.pool = shares, round(pool, 2)
        if cycle.close is None:  # no session in this cycle yet: the latest close so far
            cycle.close, cycle.close_date = last_close, last_date
        cycles.append(cycle)
        start, index = end + timedelta(days=1), index + 1
    return cycles
