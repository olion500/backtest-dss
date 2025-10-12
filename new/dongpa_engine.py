
# -*- coding: utf-8 -*-
"""
Dongpa Backtest Engine (LOC-only, Daily N-Tranche Buy)
- Integer share enforcement: buy_qty is an integer number of shares.
- Journal columns are in Korean.
"""
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

# ---------------------- Indicators / Utils ----------------------

def wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def to_weekly_close(df_daily: pd.DataFrame) -> pd.Series:
    return df_daily['Close'].resample('W-FRI').last().dropna()

def cross_up(prev: float, curr: float, level: float) -> bool:
    return prev < level and curr >= level

def cross_down(prev: float, curr: float, level: float) -> bool:
    return prev > level and curr <= level

# Monetary rounding helpers (2 decimal places for trade calculations)
MONEY_QUANT = Decimal("0.01")
ONE = Decimal("1")
HUNDRED = Decimal("100")


def to_decimal(value) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if value is None:
        return Decimal("0")
    if isinstance(value, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        size = len(value)
        if size == 0:
            return Decimal("0")
        if size == 1:
            first = value.iloc[0] if hasattr(value, "iloc") else value[0]
            return to_decimal(first)
        raise ValueError(f"Expected scalar for money conversion, got container of size {size}")
    if isinstance(value, (int, np.integer)):
        return Decimal(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"Invalid numeric value for money: {value}")
        return Decimal(str(value))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return Decimal("0")
        lower = stripped.lower()
        if lower in {"nan", "inf", "-inf", "+inf"}:
            raise ValueError(f"Invalid numeric value for money: {value}")
        return Decimal(stripped)
    try:
        if pd.isna(value):
            raise ValueError("Encountered NaN while converting to Decimal.")
    except TypeError:
        pass
    return Decimal(str(value))


def money(value) -> Decimal:
    return to_decimal(value).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)


def money_to_float(value) -> float | None:
    if value is None:
        return None
    return round(float(value), 2)

# ---------------------- Config Models ----------------------

@dataclass
class ModeParams:
    buy_cond_pct: float     # Buy condition (%) relative to previous close
    tp_pct: float           # Take profit (%)
    max_hold_days: int      # Max holding days
    slices: int             # N tranches (cash split by N)

@dataclass
class CapitalParams:
    initial_cash: float
    refresh_cycle_days: int
    profit_compound_rate: float  # PCR, e.g., 0.8
    loss_compound_rate: float    # LCR, e.g., 0.3
    slippage_pct: float = 0.0

@dataclass
class StrategyParams:
    target_ticker: str
    momentum_ticker: str
    benchmark_ticker: str | None
    rsi_period: int = 14
    reset_on_mode_change: bool = True
    defense: ModeParams = None
    offense: ModeParams = None

# ---------------------- Engine ----------------------

class DongpaBacktester:
    def __init__(self, daily_target: pd.DataFrame, daily_momo: pd.DataFrame, params: StrategyParams, cap: CapitalParams):
        self.df = daily_target.copy()
        self.momo = daily_momo.copy()
        self.p = params
        self.c = cap
        self._prep()
        self.results = None

    def _prep(self):
        self.df = self.df.sort_index()
        self.momo = self.momo.sort_index()

        if 'Close' in self.df.columns:
            self.df = self.df[self.df['Close'].notna()].copy()
        if 'Close' in self.momo.columns:
            self.momo = self.momo[self.momo['Close'].notna()].copy()
        common_start = max(self.df.index.min(), self.momo.index.min())
        common_end   = min(self.df.index.max(), self.momo.index.max())
        self.df = self.df[(self.df.index>=common_start) & (self.df.index<=common_end)]
        self.momo = self.momo[(self.momo.index>=common_start) & (self.momo.index<=common_end)]

        w_close = to_weekly_close(self.momo)
        w_rsi = wilder_rsi(w_close, self.p.rsi_period)
        self.weekly_rsi = w_rsi
        self.weekly_rsi_delta = w_rsi.diff()

        self.daily_rsi = self.weekly_rsi.reindex(self.df.index, method='ffill')
        self.daily_rsi_delta = self.weekly_rsi_delta.reindex(self.df.index, method='ffill')
        self.daily_prev_week = self.weekly_rsi.shift(1).reindex(self.df.index, method='ffill')

    def _decide_mode(self, idx, prev_mode) -> str:
        rsi = float(self.daily_rsi.loc[idx])
        prev_w = float(self.daily_prev_week.loc[idx]) if not math.isnan(self.daily_prev_week.loc[idx]) else rsi
        delta = float(self.daily_rsi_delta.loc[idx]) if not math.isnan(self.daily_rsi_delta.loc[idx]) else 0.0

        if math.isnan(rsi) or math.isnan(prev_w):
            return prev_mode or "defense"

        is_down = delta < 0
        is_up   = delta > 0

        cond_def = (is_down and (rsi >= 65 or (40 < rsi < 50) or cross_down(prev_w, rsi, 50)))
        cond_off = (is_up   and (cross_up(prev_w, rsi, 50) or (50 < rsi < 60) or (rsi < 35)))

        if cond_off and not cond_def:
            return "offense"
        if cond_def and not cond_off:
            return "defense"
        return prev_mode or "defense"

    def run(self):
        dates = self.df.index
        mode = "defense"
        cash = money(self.c.initial_cash)
        initial_cash = cash
        lots = []   # [{'qty', 'fill', 'tp', 'days'}], qty is integer shares
        equity_curve = []
        eq_dates = []
        day_in_cycle = 0
        cycle_pnl = Decimal("0")
        profit_compound = to_decimal(self.c.profit_compound_rate)
        loss_compound = to_decimal(self.c.loss_compound_rate)

        def tranche_budget_for(slices: int) -> Decimal:
            return money(cash / Decimal(max(1, slices)))

        m = self.p.defense
        tranche_budget = tranche_budget_for(m.slices)

        journal = []
        prev_close = money(self.df['Close'].iloc[0])

        for i, d in enumerate(dates):
            close = money(self.df.loc[d, 'Close'])

            # Mode decision (weekly RSI based)
            new_mode = self._decide_mode(d, mode)
            if new_mode != mode and self.p.reset_on_mode_change:
                mode = new_mode
                m = self.p.offense if mode=="offense" else self.p.defense
                tranche_budget = tranche_budget_for(m.slices)
            else:
                mode = new_mode
                m = self.p.offense if mode=="offense" else self.p.defense

            # LOC Buy (once per day) with integer share enforcement
            buy_limit = money(prev_close * (ONE + to_decimal(m.buy_cond_pct)/HUNDRED)) if i>0 else None
            buy_qty = 0
            buy_amt = Decimal("0")
            buy_avg = None
            if buy_limit is not None and close <= buy_limit and tranche_budget > Decimal("0"):
                # integer shares by floor division
                shares = int(tranche_budget // close)
                if shares > 0:
                    trade_value = money(Decimal(shares) * close)
                    if trade_value <= cash:
                        cash = money(cash - trade_value)
                        tp = money(close * (ONE + to_decimal(m.tp_pct)/HUNDRED))
                        lots.append({'qty': shares, 'fill': close, 'tp': tp, 'days': 0})
                        buy_qty = shares
                        buy_amt = trade_value
                        buy_avg = close

            # LOC Sell (TP or timeout)
            sell_qty = 0
            sell_amt = Decimal("0")
            sell_avg = None
            realized_today = Decimal("0")
            remaining = []
            for lot in lots:
                sell_now = False
                if close >= lot['tp']:
                    sell_now = True
                elif lot['days'] + 1 >= m.max_hold_days:
                    sell_now = True

                if sell_now:
                    proceeds = money(Decimal(lot['qty']) * close)
                    cash = money(cash + proceeds)
                    cost_basis = money(Decimal(lot['qty']) * lot['fill'])
                    pnl = money(proceeds - cost_basis)
                    realized_today = money(realized_today + pnl)
                    sell_qty += lot['qty']
                    sell_amt = money(sell_amt + proceeds)
                else:
                    lot['days'] += 1
                    remaining.append(lot)
            lots = remaining
            if sell_qty > 0:
                sell_avg = money(sell_amt / Decimal(sell_qty))

            # Capital refresh
            day_in_cycle += 1
            cycle_pnl = money(cycle_pnl + realized_today)
            if day_in_cycle >= self.c.refresh_cycle_days:
                if cycle_pnl >= Decimal("0"):
                    cash = money(cash + money(cycle_pnl * profit_compound))
                else:
                    cash = money(cash + money(cycle_pnl * loss_compound))
                tranche_budget = tranche_budget_for(m.slices)
                cycle_pnl = Decimal("0")
                day_in_cycle = 0

            # Mark-to-close and journal
            position_qty = sum(l['qty'] for l in lots)  # integer
            position_val = money(Decimal(position_qty) * close)
            equity = money(cash + position_val)
            equity_curve.append(float(equity)); eq_dates.append(d)

            peak = max(equity_curve) if equity_curve else float(equity)
            dd = (float(equity) - peak)/peak if peak>0 else 0.0

            tp_target = None
            if position_qty>0:
                weighted_tp = sum(Decimal(l['qty']) * l['tp'] for l in lots)
                tp_target = money(weighted_tp / Decimal(position_qty))

            cumulative_return = 0.0
            if initial_cash != Decimal("0"):
                cumulative_return = round(float((equity / initial_cash - ONE) * Decimal("100")), 2)

            journal.append({
                "거래일자": d.strftime("%Y-%m-%d"),
                "종가": money_to_float(close),
                "모드": "공세" if mode=="offense" else "안전",
                "등락률(%)": round(((float(close)/float(prev_close))-1)*100,2) if i>0 else 0.0,
                "매수조건(%)": round(m.buy_cond_pct,2),
                "매수주문가": money_to_float(buy_limit) if buy_limit else None,
                "매수평균": money_to_float(buy_avg) if buy_qty>0 and buy_avg is not None else None,
                "매수수량": int(buy_qty),
                "매수금액": money_to_float(buy_amt),
                "TP목표가": money_to_float(tp_target) if tp_target else None,
                "매도평균": money_to_float(sell_avg) if sell_qty>0 and sell_avg is not None else None,
                "매도수량": int(sell_qty),
                "매도금액": money_to_float(sell_amt),
                "실현손익": money_to_float(realized_today),
                "누적손익": money_to_float(equity - initial_cash),
                "현금": money_to_float(cash),
                "보유수량": int(position_qty),
                "평가금액": money_to_float(position_val),
                "Equity": money_to_float(equity),
                "누적수익률(%)": cumulative_return,
                "낙폭(DD%)": round(dd*100,2),
                "일일트렌치예산": money_to_float(tranche_budget),
                "분할수(N)": m.slices,
            })

            prev_close = close

        equity_series = pd.Series(equity_curve, index=pd.DatetimeIndex(eq_dates, name='Date'), name='Equity')
        journal_df = pd.DataFrame(journal)
        self.results = {"equity": equity_series, "journal": journal_df, "cash_end": money_to_float(cash), "open_positions": len(lots)}
        return self.results

# ---------------------- Metrics ----------------------

def max_drawdown(series: pd.Series) -> float:
    cummax = series.cummax()
    dd = (series - cummax) / cummax
    return float(dd.min())

def CAGR(series: pd.Series) -> float:
    if series.empty: return 0.0
    start=float(series.iloc[0]); end=float(series.iloc[-1])
    years=(series.index[-1]-series.index[0]).days/365.25
    if start<=0 or years<=0: return 0.0
    return (end/start)**(1/years)-1

def summarize(equity: pd.Series) -> dict:
    ret = equity.pct_change().dropna()
    vol = float(ret.std() * (252 ** 0.5)) if not ret.empty else 0.0
    sharpe = float((ret.mean() / ret.std()) * (252 ** 0.5)) if ret.std() > 0 else 0.0
    return {
        "Final Equity": float(equity.iloc[-1]) if not equity.empty else 0.0,
        "CAGR": float(CAGR(equity)),
        "Volatility (ann)": vol,
        "Max Drawdown": float(max_drawdown(equity)) if not equity.empty else 0.0,
        "Sharpe (rf=0)": sharpe,
    }
