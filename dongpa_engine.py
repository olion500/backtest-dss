
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


def moving_average(series: pd.Series, period: int) -> pd.Series:
    """Calculate simple moving average."""
    return series.rolling(window=period, min_periods=1).mean()


def golden_cross(short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
    """Detect golden cross (short MA crosses above long MA)."""
    prev_short = short_ma.shift(1)
    prev_long = long_ma.shift(1)
    return (prev_short <= prev_long) & (short_ma > long_ma)


def death_cross(short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
    """Detect death cross (short MA crosses below long MA)."""
    prev_short = short_ma.shift(1)
    prev_long = long_ma.shift(1)
    return (prev_short >= prev_long) & (short_ma < long_ma)


def _scalar(value):
    if isinstance(value, pd.Series):
        if value.empty:
            return float("nan")
        return value.iloc[0]
    return value

# Monetary rounding helpers (2 decimal places for trade calculations)
MONEY_QUANT = Decimal("0.01")
SHARES_QUANT = Decimal("0.00000001")  # 8 decimal places for fractional shares (crypto)
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


def shares(value, allow_fractional: bool = False) -> Decimal:
    """Round share quantity - integer or fractional (8 decimal places)"""
    d = to_decimal(value)
    if allow_fractional:
        return d.quantize(SHARES_QUANT, rounding=ROUND_HALF_UP)
    else:
        return Decimal(int(d))


def shares_to_float(value, allow_fractional: bool = False) -> float | None:
    """Convert shares to float with appropriate precision"""
    if value is None:
        return None
    if allow_fractional:
        return round(float(value), 8)
    else:
        return float(int(value))


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
    stop_loss_pct: float | None = None

@dataclass
class CapitalParams:
    initial_cash: float

@dataclass
class StrategyParams:
    target_ticker: str
    momentum_ticker: str
    rsi_period: int = 14
    enable_netting: bool = True
    allow_fractional_shares: bool = False
    defense: ModeParams | None = None
    offense: ModeParams | None = None
    # Mode switching strategy: "rsi" or "ma_cross"
    mode_switch_strategy: str = "rsi"
    # RSI threshold parameters (for "rsi" strategy)
    rsi_high_threshold: float = 65.0  # Upper RSI threshold for defense mode
    rsi_mid_low: float = 40.0         # Lower bound of middle range
    rsi_mid_high: float = 60.0        # Upper bound of middle range
    rsi_low_threshold: float = 35.0   # Lower RSI threshold for offense mode
    rsi_neutral: float = 50.0         # Neutral line for crossover detection
    # Moving average parameters (for "ma_cross" strategy)
    ma_short_period: int = 5          # Short MA period (weeks for weekly data)
    ma_long_period: int = 20          # Long MA period (weeks for weekly data)
    # Buy execution: use min(tranche_budget, cash) instead of skipping when cash < tranche_budget
    cash_limited_buy: bool = False

    def __post_init__(self):
        if self.defense is None or self.offense is None:
            raise ValueError("StrategyParams requires both 'defense' and 'offense' ModeParams")

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

        # Calculate indicators FIRST using full momo data (for proper warm-up)
        w_close = to_weekly_close(self.momo)

        if self.p.mode_switch_strategy == "rsi":
            # RSI-based mode switching (calculate with full data)
            w_rsi = wilder_rsi(w_close, self.p.rsi_period)
            self.weekly_rsi = w_rsi
            self.weekly_rsi_delta = w_rsi.diff()
        elif self.p.mode_switch_strategy == "ma_cross":
            # MA cross-based mode switching (calculate with full data)
            self.weekly_ma_short = moving_average(w_close, self.p.ma_short_period)
            self.weekly_ma_long = moving_average(w_close, self.p.ma_long_period)
            self.weekly_golden = golden_cross(self.weekly_ma_short, self.weekly_ma_long)
            self.weekly_death = death_cross(self.weekly_ma_short, self.weekly_ma_long)
        else:
            raise ValueError(f"Unknown mode_switch_strategy: {self.p.mode_switch_strategy}")

        # THEN align df to common period (momo kept full for indicator lookback)
        common_start = max(self.df.index.min(), self.momo.index.min())
        common_end = min(self.df.index.max(), self.momo.index.max())
        self.df = self.df[(self.df.index >= common_start) & (self.df.index <= common_end)]

        # Reindex indicators to df's trading days
        if self.p.mode_switch_strategy == "rsi":
            self.daily_rsi = self.weekly_rsi.reindex(self.df.index, method='ffill')
            self.daily_rsi_delta = self.weekly_rsi_delta.reindex(self.df.index, method='ffill')
            self.daily_prev_week = self.weekly_rsi.shift(1).reindex(self.df.index, method='ffill')
        elif self.p.mode_switch_strategy == "ma_cross":
            self.daily_ma_short = self.weekly_ma_short.reindex(self.df.index, method='ffill')
            self.daily_ma_long = self.weekly_ma_long.reindex(self.df.index, method='ffill')
            self.daily_golden = self.weekly_golden.reindex(self.df.index, method='ffill', fill_value=False)
            self.daily_death = self.weekly_death.reindex(self.df.index, method='ffill', fill_value=False)

    def _decide_mode(self, idx, prev_mode) -> str:
        if self.p.mode_switch_strategy == "rsi":
            return self._decide_mode_rsi(idx, prev_mode)
        elif self.p.mode_switch_strategy == "ma_cross":
            return self._decide_mode_ma_cross(idx, prev_mode)
        else:
            raise ValueError(f"Unknown mode_switch_strategy: {self.p.mode_switch_strategy}")

    def _eval_rsi_conditions(self, rsi: float, prev_w: float, delta: float) -> str | None:
        """Evaluate RSI conditions and return mode change, or None to keep current."""
        is_down = delta < 0
        is_up = delta > 0

        rsi_high = self.p.rsi_high_threshold
        rsi_mid_low = self.p.rsi_mid_low
        rsi_mid_high = self.p.rsi_mid_high
        rsi_low = self.p.rsi_low_threshold
        rsi_neutral = self.p.rsi_neutral

        cond_def = (is_down and (rsi >= rsi_high or (rsi_mid_low < rsi < rsi_neutral) or cross_down(prev_w, rsi, rsi_neutral)))
        cond_off = (is_up and (cross_up(prev_w, rsi, rsi_neutral) or (rsi_neutral < rsi < rsi_mid_high) or (rsi < rsi_low)))

        if cond_off and not cond_def:
            return "offense"
        if cond_def and not cond_off:
            return "defense"
        return None

    def _decide_mode_rsi(self, idx, prev_mode) -> str:
        """RSI-based mode switching logic."""
        rsi_raw = _scalar(self.daily_rsi.loc[idx])
        if pd.isna(rsi_raw):
            return prev_mode or "defense"
        rsi = float(rsi_raw)

        prev_raw = _scalar(self.daily_prev_week.loc[idx])
        prev_w = float(prev_raw) if not pd.isna(prev_raw) else rsi

        delta_raw = _scalar(self.daily_rsi_delta.loc[idx])
        delta = float(delta_raw) if not pd.isna(delta_raw) else 0.0

        return self._eval_rsi_conditions(rsi, prev_w, delta) or prev_mode or "defense"

    def _decide_mode_ma_cross(self, idx, prev_mode) -> str:
        """MA cross-based mode switching logic."""
        # Check for golden cross (bullish signal -> offense)
        is_golden = _scalar(self.daily_golden.loc[idx])
        # Check for death cross (bearish signal -> defense)
        is_death = _scalar(self.daily_death.loc[idx])

        # Get current MA positions
        ma_short = _scalar(self.daily_ma_short.loc[idx])
        ma_long = _scalar(self.daily_ma_long.loc[idx])

        if pd.isna(ma_short) or pd.isna(ma_long):
            return prev_mode or "defense"

        # Golden cross detected -> switch to offense
        if is_golden:
            return "offense"
        # Death cross detected -> switch to defense
        if is_death:
            return "defense"

        # No cross detected, maintain current position-based mode
        # If short MA > long MA, stay in offense; otherwise stay in defense
        if ma_short > ma_long:
            return "offense"
        else:
            return "defense"

    def _determine_initial_mode(self) -> str:
        """Determine initial mode by replaying weekly indicator data before backtest start.

        Instead of always starting with "defense", this looks at the pre-backtest
        weekly data and replays mode switching logic to find what mode should be
        active at the start of the backtest period.
        """
        if len(self.df.index) == 0:
            return "defense"

        backtest_start = self.df.index[0]

        if self.p.mode_switch_strategy == "rsi":
            # Get weekly RSI data before backtest start
            pre_weeks = self.weekly_rsi[self.weekly_rsi.index < backtest_start]
            if pre_weeks.empty:
                return "defense"

            rsi_delta = self.weekly_rsi_delta
            prev_rsi = self.weekly_rsi.shift(1)

            mode = "defense"
            for week_date in pre_weeks.index:
                rsi_raw = _scalar(pre_weeks.loc[week_date])
                if pd.isna(rsi_raw):
                    continue
                rsi_val = float(rsi_raw)

                delta_raw = _scalar(rsi_delta.loc[week_date]) if week_date in rsi_delta.index else None
                delta = float(delta_raw) if delta_raw is not None and not pd.isna(delta_raw) else 0.0

                prev_raw = _scalar(prev_rsi.loc[week_date]) if week_date in prev_rsi.index else None
                prev_w = float(prev_raw) if prev_raw is not None and not pd.isna(prev_raw) else rsi_val

                result = self._eval_rsi_conditions(rsi_val, prev_w, delta)
                if result is not None:
                    mode = result

            return mode

        elif self.p.mode_switch_strategy == "ma_cross":
            # Get weekly MA data before backtest start
            pre_short = self.weekly_ma_short[self.weekly_ma_short.index < backtest_start]
            pre_long = self.weekly_ma_long[self.weekly_ma_long.index < backtest_start]
            if pre_short.empty or pre_long.empty:
                return "defense"

            # Use the last available MA values before backtest start
            last_short = float(_scalar(pre_short.iloc[-1]))
            last_long = float(_scalar(pre_long.iloc[-1]))

            if pd.isna(last_short) or pd.isna(last_long):
                return "defense"

            return "offense" if last_short > last_long else "defense"

        return "defense"

    def run(self):
        dates = self.df.index
        cash = money(self.c.initial_cash)
        initial_cash = cash

        if len(dates) == 0:
            equity_series = pd.Series(dtype=float, name='Equity')
            journal_df = pd.DataFrame()
            self.results = {"equity": equity_series, "journal": journal_df, "cash_end": money_to_float(cash), "open_positions": 0}
            return self.results

        mode = self._determine_initial_mode()
        lots = []   # [{'qty', 'fill', 'tp', 'days', 'buy_date', 'trade_idx'}]
        trades = []
        daily_rows = []
        equity_curve = []
        eq_dates = []
        realized_cumulative = Decimal("0")

        def tranche_budget_for(slices: int, base_cash: Decimal) -> Decimal:
            return money(base_cash / Decimal(max(1, slices)))

        m = self.p.offense if mode == "offense" else self.p.defense
        tranche_base_cash = cash
        tranche_budget = tranche_budget_for(m.slices, tranche_base_cash)
        prev_close = money(self.df['Close'].iloc[0])

        peak_equity = float(cash)

        for i, d in enumerate(dates):
            close = money(self.df.loc[d, 'Close'])

            # Mode decision (weekly RSI based)
            mode = self._decide_mode(d, mode)
            m = self.p.offense if mode == "offense" else self.p.defense

            tranche_budget = tranche_budget_for(m.slices, tranche_base_cash)

            # LOC Buy (once per day) with share quantity enforcement
            buy_limit = money(prev_close * (ONE + to_decimal(m.buy_cond_pct) / HUNDRED)) if i > 0 else None
            buy_trade_id = None
            buy_qty_executed = Decimal("0")
            buy_amt_value = Decimal("0")
            planned_buy_qty = Decimal("0")
            planned_buy_order_value = Decimal("0")
            if buy_limit is not None and tranche_budget > Decimal("0") and buy_limit > Decimal("0"):
                # Calculate share quantity based on fractional trading setting
                raw_qty = tranche_budget / buy_limit
                planned_buy_qty = shares(raw_qty, self.p.allow_fractional_shares)
                if planned_buy_qty > Decimal("0"):
                    planned_buy_order_value = money(planned_buy_qty * buy_limit)

            if (
                buy_limit is not None
                and close <= buy_limit
                and tranche_budget > Decimal("0")
                and close > Decimal("0")
            ):
                # 실제 체결 수량은 종가 기준으로 계산 (LOC는 종가에 체결)
                exec_budget = min(tranche_budget, cash) if self.p.cash_limited_buy else tranche_budget
                raw_exec_qty = exec_budget / close
                share_qty = shares(raw_exec_qty, self.p.allow_fractional_shares)
                trade_value = money(share_qty * close)
                if share_qty > Decimal("0") and trade_value <= cash:
                        cash = money(cash - trade_value)
                        tp = money(close * (ONE + to_decimal(m.tp_pct) / HUNDRED))
                        sl = None
                        if m.stop_loss_pct is not None and m.stop_loss_pct > 0:
                            stop_loss_factor = ONE - to_decimal(m.stop_loss_pct) / HUNDRED
                            if stop_loss_factor > Decimal("0"):
                                sl = money(close * stop_loss_factor)
                        mode_label = "공세" if mode == "offense" else "안전"
                        trade_entry = {
                            "거래ID": len(trades) + 1,
                            "매수일자": d.strftime("%Y-%m-%d"),
                            "매수모드": mode_label,
                            "매수조건(%)": round(m.buy_cond_pct, 2),
                            "매수주문가": money_to_float(buy_limit) if buy_limit is not None else None,
                            "매수체결가": money_to_float(close),
                            "매수수량": shares_to_float(share_qty, self.p.allow_fractional_shares),
                            "매수금액": money_to_float(trade_value),
                            "TP목표가": money_to_float(tp),
                            "SL목표가": money_to_float(sl) if sl is not None else None,
                            "최대보유일": int(m.max_hold_days),
                            "수익률(%)": None,
                            "매도일자": None,
                            "매도평균": None,
                            "매도수량": 0,
                            "매도금액": None,
                            "보유기간(일)": None,
                            "실현손익": None,
                            "청산사유": None,
                            "상태": "보유중",
                            "_buy_timestamp": d,
                        }
                        trades.append(trade_entry)
                        lots.append({
                            'qty': share_qty,
                            'fill': close,
                            'tp': tp,
                            'sl': sl,
                            'days': 0,
                            'buy_date': d,
                            'trade_idx': len(trades) - 1,
                            'max_hold': int(m.max_hold_days),
                            'buy_idx': i,
                        })
                        buy_trade_id = trade_entry["거래ID"]
                        buy_qty_executed = share_qty
                        buy_amt_value = trade_value

            # LOC Sell (TP or timeout)
            realized_today = Decimal("0")
            remaining = []
            sell_qty_total = Decimal("0")
            sell_amt_total = Decimal("0")
            sell_trade_ids: list[int] = []
            for lot in lots:
                sell_now = False
                sell_reason = None
                if lot.get('sl') is not None and close <= lot['sl']:
                    sell_now = True
                    sell_reason = "SL"
                elif close >= lot['tp']:
                    sell_now = True
                    sell_reason = "TP"
                elif lot['days'] + 1 >= lot['max_hold']:
                    sell_now = True
                    sell_reason = "MOC"

                if sell_now:
                    proceeds = money(Decimal(lot['qty']) * close)
                    cash = money(cash + proceeds)
                    cost_basis = money(Decimal(lot['qty']) * lot['fill'])
                    pnl = money(proceeds - cost_basis)
                    realized_today = money(realized_today + pnl)
                    pnl_pct = None
                    if cost_basis > Decimal("0"):
                        pnl_pct = round(float((pnl / cost_basis) * Decimal("100")), 2)

                    trade_entry = trades[lot['trade_idx']]
                    hold_days = int((i - lot['buy_idx']) + 1)
                    trade_entry.update({
                        "매도일자": d.strftime("%Y-%m-%d"),
                        "매도평균": money_to_float(close),
                        "매도수량": shares_to_float(lot['qty'], self.p.allow_fractional_shares),
                        "매도금액": money_to_float(proceeds),
                        "보유기간(일)": hold_days,
                        "실현손익": money_to_float(pnl),
                        "수익률(%)": pnl_pct,
                        "청산사유": sell_reason,
                        "상태": "완료",
                    })
                    sell_qty_total = sell_qty_total + lot['qty']
                    sell_amt_total = money(sell_amt_total + proceeds)
                    sell_trade_ids.append(trade_entry["거래ID"])
                else:
                    lot['days'] += 1
                    remaining.append(lot)
            lots = remaining

            realized_cumulative = money(realized_cumulative + realized_today)

            # After sells, reset tranche base to current cash so budget
            # always reflects available capital (prevents budget deadlock
            # where tranche_base stays at a historical peak while cash shrinks).
            if sell_qty_total > Decimal("0"):
                tranche_base_cash = cash

            # Mark-to-close (for equity curve)
            position_qty = sum(l['qty'] for l in lots)
            position_val = money(Decimal(position_qty) * close)
            equity = money(cash + position_val)
            tp_avg_open = None
            if position_qty > 0:
                weighted_tp = sum(Decimal(l['qty']) * l['tp'] for l in lots)
                tp_avg_open = money(weighted_tp / Decimal(position_qty))
            equity_curve.append(float(equity))
            eq_dates.append(d)

            buy_summary = "매수 없음"
            if planned_buy_qty > Decimal("0") and buy_limit is not None:
                buy_price = money_to_float(buy_limit)
                buy_budget = money_to_float(planned_buy_order_value)
                qty_display = shares_to_float(planned_buy_qty, self.p.allow_fractional_shares)
                buy_summary = f"매수 {qty_display}주 @ {buy_price:.2f} (예산 ${buy_budget:,.2f})"

            sell_summary = "TP대기 없음"
            if lots:
                tp_groups: dict[Decimal, Decimal] = {}
                for lot in lots:
                    tp_price = money(lot['tp'])
                    tp_groups[tp_price] = tp_groups.get(tp_price, Decimal("0")) + lot['qty']
                sell_entries = [
                    f"{shares_to_float(qty, self.p.allow_fractional_shares)}주 @ {money_to_float(tp):.2f}"
                    for tp, qty in sorted(tp_groups.items(), key=lambda item: float(item[0]))
                ]
                sell_summary = "TP대기 " + ", ".join(sell_entries)

            order_summary = f"{buy_summary} | {sell_summary}" if buy_summary or sell_summary else "예약 없음"

            if float(equity) > peak_equity:
                peak_equity = float(equity)
            drawdown_pct = 0.0
            if peak_equity > 0:
                drawdown_pct = round(((float(equity) / peak_equity) - 1) * 100, 2)

            cumulative_return = 0.0
            if initial_cash != Decimal("0"):
                cumulative_return = round(float((realized_cumulative / initial_cash) * Decimal("100")), 2)

            raw_buy_qty = buy_qty_executed
            raw_buy_amt = buy_amt_value
            raw_sell_qty = sell_qty_total
            raw_sell_amt = sell_amt_total

            # Netting is display-only: offsets buy/sell quantities in the daily
            # journal for reporting purposes. Actual cash, lots, and positions
            # are already settled independently above.
            net_buy_qty = raw_buy_qty
            net_sell_qty = raw_sell_qty
            net_buy_amt = raw_buy_amt
            net_sell_amt = raw_sell_amt
            netting_applied = False
            netting_detail = None

            if self.p.enable_netting and raw_buy_qty > Decimal("0") and raw_sell_qty > Decimal("0"):
                offset_qty = min(raw_buy_qty, raw_sell_qty)
                if offset_qty > Decimal("0"):
                    offset_amt = money(offset_qty * close)
                    net_buy_qty = raw_buy_qty - offset_qty
                    net_sell_qty = raw_sell_qty - offset_qty
                    net_buy_amt = money(max(Decimal("0"), net_buy_amt - offset_amt))
                    net_sell_amt = money(max(Decimal("0"), net_sell_amt - offset_amt))
                    netting_applied = True
                    raw_buy_display = shares_to_float(raw_buy_qty, self.p.allow_fractional_shares)
                    raw_sell_display = shares_to_float(raw_sell_qty, self.p.allow_fractional_shares)
                    if net_buy_qty > Decimal("0"):
                        net_buy_display = shares_to_float(net_buy_qty, self.p.allow_fractional_shares)
                        netting_detail = f"매수 {raw_buy_display}주, 매도 {raw_sell_display}주 → 순매수 {net_buy_display}주"
                    elif net_sell_qty > Decimal("0"):
                        net_sell_display = shares_to_float(net_sell_qty, self.p.allow_fractional_shares)
                        netting_detail = f"매수 {raw_buy_display}주, 매도 {raw_sell_display}주 → 순매도 {net_sell_display}주"
                    else:
                        netting_detail = f"매수 {raw_buy_display}주, 매도 {raw_sell_display}주 → 상쇄"

            net_sell_avg = None
            if net_sell_qty > Decimal("0"):
                net_sell_avg = money_to_float(money(net_sell_amt / net_sell_qty))

            raw_sell_avg = None
            if raw_sell_qty > Decimal("0"):
                raw_sell_avg = money_to_float(money(raw_sell_amt / raw_sell_qty))

            daily_row = {
                "거래일자": d.strftime("%Y-%m-%d"),
                "모드": "공세" if mode == "offense" else "안전",
                "종가": money_to_float(close),
                "등락률(%)": round(((float(close) / float(prev_close)) - 1) * 100, 2) if i > 0 else 0.0,
                "매수조건(%)": round(m.buy_cond_pct, 2),
                "매수주문가": money_to_float(buy_limit) if buy_limit is not None else None,
                "매수체결가": money_to_float(close) if buy_trade_id else None,
                "매수수량": shares_to_float(net_buy_qty, self.p.allow_fractional_shares),
                "매수금액": money_to_float(net_buy_amt) if net_buy_qty > Decimal("0") else 0.0,
                "매수거래ID": buy_trade_id,
                "매도평균": net_sell_avg,
                "매도수량": shares_to_float(net_sell_qty, self.p.allow_fractional_shares),
                "매도금액": money_to_float(net_sell_amt) if net_sell_qty > Decimal("0") else 0.0,
                "매도거래ID목록": ",".join(str(tid) for tid in sell_trade_ids) if sell_trade_ids else None,
                "실현손익": money_to_float(realized_today),
                "현금": money_to_float(cash),
                "보유수량": shares_to_float(position_qty, self.p.allow_fractional_shares),
                "평가금액": money_to_float(position_val),
                "Equity": money_to_float(equity),
                "누적손익": money_to_float(realized_cumulative),
                "누적수익률(%)": cumulative_return,
                "낙폭(DD%)": drawdown_pct,
                "일일트렌치예산": money_to_float(tranche_budget),
                "TP평균(보유)": money_to_float(tp_avg_open) if tp_avg_open is not None else None,
                "원매수수량": shares_to_float(raw_buy_qty, self.p.allow_fractional_shares),
                "원매수금액": money_to_float(raw_buy_amt) if raw_buy_qty > Decimal("0") else 0.0,
                "원매도수량": shares_to_float(raw_sell_qty, self.p.allow_fractional_shares),
                "원매도금액": money_to_float(raw_sell_amt) if raw_sell_qty > Decimal("0") else 0.0,
                "원매도평균": raw_sell_avg,
                "퉁치기적용": netting_applied,
                "퉁치기상세": netting_detail,
                "예약요약": order_summary,
            }
            daily_rows.append(daily_row)

            prev_close = close

        # Update open trades with current holding period
        last_index = len(dates) - 1
        for lot in lots:
            trade_entry = trades[lot['trade_idx']]
            trade_entry["보유기간(일)"] = int((last_index - lot['buy_idx']) + 1)

        equity_series = pd.Series(equity_curve, index=pd.DatetimeIndex(eq_dates, name='Date'), name='Equity')
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty and "_buy_timestamp" in trades_df.columns:
            trades_df = trades_df.drop(columns=["_buy_timestamp"])
        daily_df = pd.DataFrame(daily_rows)

        self.results = {
            "equity": equity_series,
            "journal": daily_df,
            "trade_log": trades_df,
            "cash_end": money_to_float(cash),
            "open_positions": len(lots),
        }
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
    cagr = float(CAGR(equity))
    mdd = float(max_drawdown(equity)) if not equity.empty else 0.0
    calmar = cagr / abs(mdd) if mdd != 0.0 else 0.0
    return {
        "Final Equity": float(equity.iloc[-1]) if not equity.empty else 0.0,
        "CAGR": cagr,
        "Volatility (ann)": vol,
        "Max Drawdown": mdd,
        "Sharpe (rf=0)": sharpe,
        "Calmar Ratio": calmar,
    }
