
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

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns (e.g. ('Close', 'SOXL')) to a
    single level ('Close', ...).

    Recent yfinance versions return MultiIndex columns even for a single
    ticker. When that happens, ``df['Close']`` is a DataFrame rather than a
    Series, so a mask like ``df[df['Close'].notna()]`` aligns on columns
    instead of filtering rows — silently letting NaN-close rows (e.g. the
    current, not-yet-closed trading day) slip through. Collapsing the ticker
    level keeps all downstream ``df['Close']`` / ``df.loc[d, 'Close']`` access
    scalar and correct.
    """
    if df is None:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

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


def weekly_roc(series: pd.Series, period: int = 4) -> pd.Series:
    """N주 변화율 (Rate of Change). 양수=상승, 음수=하락."""
    shifted = series.shift(period)
    return (series - shifted) / shifted.replace(0, np.nan)


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

@dataclass(frozen=True)
class ModeParams:
    buy_cond_pct: float     # Buy condition (%) relative to previous close
    tp_pct: float           # Take profit (%)
    max_hold_days: int      # Max holding days
    slices: int             # N tranches (cash split by N)
    stop_loss_pct: float | None = None

@dataclass(frozen=True)
class CapitalParams:
    initial_cash: float

@dataclass(frozen=True)
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
    # ROC parameters (for "roc" strategy)
    roc_period: int = 4               # N-week Rate of Change period
    # BTC overnight parameters (for "btc_overnight" strategy)
    btc_lookback_days: int = 1        # N calendar days for BTC return
    btc_threshold_pct: float = 0.0    # Min % return to trigger mode (0 = any positive)
    # Buy execution: use min(tranche_budget, cash) instead of skipping when cash < tranche_budget
    cash_limited_buy: bool = False

    def __post_init__(self):
        if self.defense is None or self.offense is None:
            raise ValueError("StrategyParams requires both 'defense' and 'offense' ModeParams")

# ---------------------- Indicator Data ----------------------

@dataclass(frozen=True)
class Indicators:
    """Pre-computed indicator series from compute_indicators()."""
    strategy: str  # "rsi" | "ma_cross" | "roc" | "btc_overnight"
    weekly_rsi: pd.Series | None = None
    weekly_rsi_delta: pd.Series | None = None
    daily_rsi: pd.Series | None = None
    daily_rsi_delta: pd.Series | None = None
    daily_prev_week: pd.Series | None = None
    weekly_ma_short: pd.Series | None = None
    weekly_ma_long: pd.Series | None = None
    daily_golden: pd.Series | None = None
    daily_death: pd.Series | None = None
    daily_ma_short: pd.Series | None = None
    daily_ma_long: pd.Series | None = None
    daily_roc: pd.Series | None = None
    daily_btc_signal: pd.Series | None = None

# ---------------------- Backtest Result ----------------------

@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    journal: pd.DataFrame
    trade_log: pd.DataFrame
    cash_end: float
    open_positions: int

# ---------------------- Pure Functions: Indicators ----------------------

def compute_indicators(
    daily_target: pd.DataFrame,
    daily_momo: pd.DataFrame,
    params: StrategyParams,
    btc_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, Indicators]:
    """Pure: compute all indicator series from raw data.
    Returns (aligned_target_df, indicators)."""
    df = normalize_ohlcv(daily_target).copy().sort_index()
    momo = normalize_ohlcv(daily_momo).copy().sort_index()
    btc_data = normalize_ohlcv(btc_data)

    if 'Close' in df.columns:
        df = df[df['Close'].notna()].copy()
    if 'Close' in momo.columns:
        momo = momo[momo['Close'].notna()].copy()

    # Calculate indicators FIRST using full momo data (for proper warm-up)
    w_close = to_weekly_close(momo)

    ind_kwargs: dict = {"strategy": params.mode_switch_strategy}
    _btc_close = None  # for btc_overnight, used after alignment

    if params.mode_switch_strategy == "rsi":
        w_rsi = wilder_rsi(w_close, params.rsi_period)
        ind_kwargs["weekly_rsi"] = w_rsi
        ind_kwargs["weekly_rsi_delta"] = w_rsi.diff()
    elif params.mode_switch_strategy == "ma_cross":
        w_ma_short = moving_average(w_close, params.ma_short_period)
        w_ma_long = moving_average(w_close, params.ma_long_period)
        ind_kwargs["weekly_ma_short"] = w_ma_short
        ind_kwargs["weekly_ma_long"] = w_ma_long
    elif params.mode_switch_strategy == "roc":
        ind_kwargs["daily_roc"] = None  # placeholder, set after alignment
    elif params.mode_switch_strategy == "btc_overnight":
        if btc_data is None:
            raise ValueError("btc_data required for btc_overnight strategy")
        btc_close = btc_data['Close']
        if isinstance(btc_close, pd.DataFrame):
            btc_close = btc_close.iloc[:, 0]
        _btc_close = btc_close.sort_index()
    else:
        raise ValueError(f"Unknown mode_switch_strategy: {params.mode_switch_strategy}")

    # THEN align df to common period (momo kept full for indicator lookback)
    common_start = max(df.index.min(), momo.index.min())
    common_end = min(df.index.max(), momo.index.max())
    df = df[(df.index >= common_start) & (df.index <= common_end)]

    # Reindex indicators to df's trading days
    if params.mode_switch_strategy == "rsi":
        ind_kwargs["daily_rsi"] = ind_kwargs["weekly_rsi"].reindex(df.index, method='ffill')
        ind_kwargs["daily_rsi_delta"] = ind_kwargs["weekly_rsi_delta"].reindex(df.index, method='ffill')
        ind_kwargs["daily_prev_week"] = ind_kwargs["weekly_rsi"].shift(1).reindex(df.index, method='ffill')
    elif params.mode_switch_strategy == "ma_cross":
        w_golden = golden_cross(ind_kwargs["weekly_ma_short"], ind_kwargs["weekly_ma_long"])
        w_death = death_cross(ind_kwargs["weekly_ma_short"], ind_kwargs["weekly_ma_long"])
        ind_kwargs["daily_ma_short"] = ind_kwargs["weekly_ma_short"].reindex(df.index, method='ffill')
        ind_kwargs["daily_ma_long"] = ind_kwargs["weekly_ma_long"].reindex(df.index, method='ffill')
        ind_kwargs["daily_golden"] = w_golden.reindex(df.index, method='ffill', fill_value=False)
        ind_kwargs["daily_death"] = w_death.reindex(df.index, method='ffill', fill_value=False)
    elif params.mode_switch_strategy == "roc":
        w_roc = weekly_roc(w_close, params.roc_period)
        ind_kwargs["daily_roc"] = w_roc.reindex(df.index, method='ffill')
    elif params.mode_switch_strategy == "btc_overnight":
        target_dates = df.index
        lookback = params.btc_lookback_days
        d1_dates = target_dates - pd.Timedelta(days=1)
        btc_d1 = _btc_close.reindex(d1_dates, method='ffill')
        dn_dates = target_dates - pd.Timedelta(days=1 + lookback)
        btc_dn = _btc_close.reindex(dn_dates, method='ffill')
        ind_kwargs["daily_btc_signal"] = pd.Series(
            (btc_d1.values / btc_dn.values) - 1,
            index=target_dates
        )

    return df, Indicators(**ind_kwargs)

# ---------------------- Pure Functions: Mode Decisions ----------------------

def eval_rsi_conditions(rsi: float, prev_w: float, delta: float, params: StrategyParams) -> str | None:
    """Evaluate RSI conditions and return mode change, or None to keep current."""
    is_down = delta < 0
    is_up = delta > 0

    rsi_high = params.rsi_high_threshold
    rsi_mid_low = params.rsi_mid_low
    rsi_mid_high = params.rsi_mid_high
    rsi_low = params.rsi_low_threshold
    rsi_neutral = params.rsi_neutral

    cond_def = (is_down and (rsi >= rsi_high or (rsi_mid_low < rsi < rsi_neutral) or cross_down(prev_w, rsi, rsi_neutral)))
    cond_off = (is_up and (cross_up(prev_w, rsi, rsi_neutral) or (rsi_neutral < rsi < rsi_mid_high) or (rsi < rsi_low)))

    if cond_off and not cond_def:
        return "offense"
    if cond_def and not cond_off:
        return "defense"
    return None


def _decide_mode_rsi(idx, prev_mode: str, indicators: Indicators, params: StrategyParams) -> str:
    rsi_raw = _scalar(indicators.daily_rsi.loc[idx])
    if pd.isna(rsi_raw):
        return prev_mode or "defense"
    rsi = float(rsi_raw)

    prev_raw = _scalar(indicators.daily_prev_week.loc[idx])
    prev_w = float(prev_raw) if not pd.isna(prev_raw) else rsi

    delta_raw = _scalar(indicators.daily_rsi_delta.loc[idx])
    delta = float(delta_raw) if not pd.isna(delta_raw) else 0.0

    return eval_rsi_conditions(rsi, prev_w, delta, params) or prev_mode or "defense"


def _decide_mode_ma_cross(idx, prev_mode: str, indicators: Indicators) -> str:
    is_golden = _scalar(indicators.daily_golden.loc[idx])
    is_death = _scalar(indicators.daily_death.loc[idx])
    ma_short = _scalar(indicators.daily_ma_short.loc[idx])
    ma_long = _scalar(indicators.daily_ma_long.loc[idx])

    if pd.isna(ma_short) or pd.isna(ma_long):
        return prev_mode or "defense"

    if is_golden:
        return "offense"
    if is_death:
        return "defense"

    if ma_short > ma_long:
        return "offense"
    else:
        return "defense"


def _decide_mode_roc(idx, prev_mode: str, indicators: Indicators) -> str:
    roc_raw = _scalar(indicators.daily_roc.loc[idx])
    if pd.isna(roc_raw):
        return prev_mode or "defense"
    roc_val = float(roc_raw)
    if roc_val > 0:
        return "offense"
    elif roc_val < 0:
        return "defense"
    return prev_mode or "defense"


def _decide_mode_btc(idx, prev_mode: str, indicators: Indicators, params: StrategyParams) -> str:
    sig = _scalar(indicators.daily_btc_signal.loc[idx])
    if pd.isna(sig):
        return prev_mode or "defense"
    threshold = params.btc_threshold_pct / 100.0
    sig_val = float(sig)
    if sig_val > threshold:
        return "offense"
    elif sig_val < -threshold:
        return "defense"
    return prev_mode or "defense"


def decide_mode(idx, prev_mode: str, indicators: Indicators, params: StrategyParams) -> str:
    """Decide trading mode for a given day. Pure function."""
    strategy = indicators.strategy
    if strategy == "rsi":
        return _decide_mode_rsi(idx, prev_mode, indicators, params)
    elif strategy == "ma_cross":
        return _decide_mode_ma_cross(idx, prev_mode, indicators)
    elif strategy == "roc":
        return _decide_mode_roc(idx, prev_mode, indicators)
    elif strategy == "btc_overnight":
        return _decide_mode_btc(idx, prev_mode, indicators, params)
    else:
        raise ValueError(f"Unknown mode_switch_strategy: {strategy}")


def _check(ok: bool) -> str:
    return "O" if ok else "X"


def explain_mode(idx, indicators: Indicators, params: StrategyParams) -> str:
    """Return a human-readable explanation of why the current mode was chosen."""
    strategy = indicators.strategy

    if strategy == "rsi":
        rsi_raw = _scalar(indicators.daily_rsi.loc[idx])
        if pd.isna(rsi_raw):
            return "RSI 데이터 없음 → 이전 모드 유지"
        rsi = float(rsi_raw)
        prev_raw = _scalar(indicators.daily_prev_week.loc[idx])
        prev_w = float(prev_raw) if not pd.isna(prev_raw) else rsi
        delta_raw = _scalar(indicators.daily_rsi_delta.loc[idx])
        delta = float(delta_raw) if not pd.isna(delta_raw) else 0.0

        is_down = delta < 0
        is_up = delta > 0

        rsi_high = params.rsi_high_threshold
        rsi_mid_low = params.rsi_mid_low
        rsi_mid_high = params.rsi_mid_high
        rsi_low = params.rsi_low_threshold
        rsi_neutral = params.rsi_neutral

        # Evaluate individual conditions
        c_high_down = is_down and rsi >= rsi_high
        c_mid_low_down = is_down and (rsi_mid_low < rsi < rsi_neutral)
        c_cross_down = is_down and cross_down(prev_w, rsi, rsi_neutral)
        c_cross_up = is_up and cross_up(prev_w, rsi, rsi_neutral)
        c_mid_high_up = is_up and (rsi_neutral < rsi < rsi_mid_high)
        c_low_up = is_up and (rsi < rsi_low)

        cond_def = c_high_down or c_mid_low_down or c_cross_down
        cond_off = c_cross_up or c_mid_high_up or c_low_up

        if cond_off and not cond_def:
            result = "공세 모드"
        elif cond_def and not cond_off:
            result = "안전 모드"
        else:
            result = "이전 모드 유지"

        lines = [
            f"[ 지표 현황 ]",
            f"  주봉 RSI    : {rsi:.2f}",
            f"  전주 RSI    : {prev_w:.2f}",
            f"  ΔRSI        : {delta:+.2f} ({'상승' if is_up else '하락' if is_down else '변동없음'})",
            f"",
            f"[ 임계값 설정 ]",
            f"  H={rsi_high} MH={rsi_mid_high} N={rsi_neutral} ML={rsi_mid_low} L={rsi_low}",
            f"",
            f"[ 안전 조건 ] (ΔRSI<0 필수)",
            f"  [{_check(is_down)}] ΔRSI < 0 (전제조건)",
            f"  [{_check(c_high_down)}] RSI ≥ H : {rsi:.1f} ≥ {rsi_high}",
            f"  [{_check(c_mid_low_down)}] ML < RSI < N : {rsi_mid_low} < {rsi:.1f} < {rsi_neutral}",
            f"  [{_check(c_cross_down)}] N선 하향돌파 : {prev_w:.1f} → {rsi:.1f}",
            f"",
            f"[ 공세 조건 ] (ΔRSI>0 필수)",
            f"  [{_check(is_up)}] ΔRSI > 0 (전제조건)",
            f"  [{_check(c_cross_up)}] N선 상향돌파 : {prev_w:.1f} → {rsi:.1f}",
            f"  [{_check(c_mid_high_up)}] N < RSI < MH : {rsi_neutral} < {rsi:.1f} < {rsi_mid_high}",
            f"  [{_check(c_low_up)}] RSI < L : {rsi:.1f} < {rsi_low}",
            f"",
            f"→ 판정: **{result}**",
        ]
        return "\n".join(lines)

    elif strategy == "ma_cross":
        ma_short = _scalar(indicators.daily_ma_short.loc[idx])
        ma_long = _scalar(indicators.daily_ma_long.loc[idx])
        if pd.isna(ma_short) or pd.isna(ma_long):
            return "MA 데이터 없음 → 이전 모드 유지"

        ma_s = float(ma_short)
        ma_l = float(ma_long)
        is_golden = bool(_scalar(indicators.daily_golden.loc[idx]))
        is_death = bool(_scalar(indicators.daily_death.loc[idx]))
        above = ma_s > ma_l

        if is_golden:
            result = "공세 모드 (골든크로스)"
        elif is_death:
            result = "안전 모드 (데스크로스)"
        elif above:
            result = "공세 모드"
        else:
            result = "안전 모드"

        lines = [
            f"[ 지표 현황 ]",
            f"  Short MA ({params.ma_short_period}주) : {ma_s:.2f}",
            f"  Long  MA ({params.ma_long_period}주) : {ma_l:.2f}",
            f"  차이 (S-L)       : {ma_s - ma_l:+.2f}",
            f"",
            f"[ 조건 체크 ]",
            f"  [{_check(is_golden)}] 골든크로스 (Short가 Long 상향돌파)",
            f"  [{_check(is_death)}] 데스크로스 (Short가 Long 하향돌파)",
            f"  [{_check(above)}] Short > Long (상승 추세)",
            f"",
            f"→ 판정: **{result}**",
        ]
        return "\n".join(lines)

    elif strategy == "roc":
        roc_raw = _scalar(indicators.daily_roc.loc[idx])
        if pd.isna(roc_raw):
            return "ROC 데이터 없음 → 이전 모드 유지"
        roc_val = float(roc_raw)

        positive = roc_val > 0
        negative = roc_val < 0
        if positive:
            result = "공세 모드"
        elif negative:
            result = "안전 모드"
        else:
            result = "이전 모드 유지"

        lines = [
            f"[ 지표 현황 ]",
            f"  {params.roc_period}주 ROC : {roc_val:+.6f} ({roc_val * 100:+.2f}%)",
            f"",
            f"[ 판정 규칙 ]",
            f"  [{_check(positive)}] ROC > 0 → 공세",
            f"  [{_check(negative)}] ROC < 0 → 안전",
            f"  [{_check(not positive and not negative)}] ROC = 0 → 이전 모드 유지",
            f"",
            f"→ 판정: **{result}**",
        ]
        return "\n".join(lines)

    elif strategy == "btc_overnight":
        sig = _scalar(indicators.daily_btc_signal.loc[idx])
        if pd.isna(sig):
            return "BTC 시그널 없음 → 이전 모드 유지"
        sig_val = float(sig)
        threshold = params.btc_threshold_pct / 100.0

        above = sig_val > threshold
        below = sig_val < -threshold
        if above:
            result = "공세 모드"
        elif below:
            result = "안전 모드"
        else:
            result = "이전 모드 유지"

        lines = [
            f"[ 지표 현황 ]",
            f"  BTC 야간수익률  : {sig_val:+.6f} ({sig_val * 100:+.2f}%)",
            f"  Lookback        : {params.btc_lookback_days}일",
            f"  임계값          : ±{params.btc_threshold_pct:.1f}% (±{threshold:.4f})",
            f"",
            f"[ 판정 규칙 ]",
            f"  [{_check(above)}] 수익률 > +임계값 ({sig_val:+.4f} > {threshold:+.4f}) → 공세",
            f"  [{_check(below)}] 수익률 < -임계값 ({sig_val:+.4f} < {-threshold:+.4f}) → 안전",
            f"  [{_check(not above and not below)}] 범위 내 → 이전 모드 유지",
            f"",
            f"→ 판정: **{result}**",
        ]
        return "\n".join(lines)

    return "알 수 없는 전략"


def determine_initial_mode(backtest_start, indicators: Indicators, params: StrategyParams) -> str:
    """Determine initial mode by replaying weekly indicator data before backtest start."""
    if params.mode_switch_strategy == "rsi":
        w_rsi = indicators.weekly_rsi
        if w_rsi is None:
            return "defense"
        pre_weeks = w_rsi[w_rsi.index < backtest_start]
        if pre_weeks.empty:
            return "defense"

        rsi_delta = indicators.weekly_rsi_delta
        prev_rsi = w_rsi.shift(1)

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

            result = eval_rsi_conditions(rsi_val, prev_w, delta, params)
            if result is not None:
                mode = result

        return mode

    elif params.mode_switch_strategy == "ma_cross":
        w_short = indicators.weekly_ma_short
        w_long = indicators.weekly_ma_long
        if w_short is None or w_long is None:
            return "defense"
        pre_short = w_short[w_short.index < backtest_start]
        pre_long = w_long[w_long.index < backtest_start]
        if pre_short.empty or pre_long.empty:
            return "defense"

        last_short = float(_scalar(pre_short.iloc[-1]))
        last_long = float(_scalar(pre_long.iloc[-1]))

        if pd.isna(last_short) or pd.isna(last_long):
            return "defense"

        return "offense" if last_short > last_long else "defense"

    elif params.mode_switch_strategy == "roc":
        d_roc = indicators.daily_roc
        if d_roc is None:
            return "defense"
        # For ROC we use the daily_roc which is already weekly_roc reindexed;
        # look at the underlying weekly_roc before backtest start.
        # Since we don't store weekly_roc_series separately, use daily_roc's
        # pre-backtest values (which are ffill'd from weekly).
        pre_roc = d_roc[d_roc.index < backtest_start]
        if pre_roc.empty:
            return "defense"

        last_roc = _scalar(pre_roc.iloc[-1])
        if pd.isna(last_roc):
            return "defense"

        return "offense" if float(last_roc) > 0 else "defense"

    elif params.mode_switch_strategy == "btc_overnight":
        btc_sig = indicators.daily_btc_signal
        if btc_sig is not None and not btc_sig.empty:
            first_sig = _scalar(btc_sig.iloc[0])
            if not pd.isna(first_sig):
                threshold = params.btc_threshold_pct / 100.0
                sig_val = float(first_sig)
                if sig_val > threshold:
                    return "offense"
                elif sig_val < -threshold:
                    return "defense"
        return "defense"

    return "defense"

# ---------------------- Pure Function: run_backtest ----------------------

def run_backtest(
    daily_target: pd.DataFrame,
    daily_momo: pd.DataFrame,
    params: StrategyParams,
    cap: CapitalParams,
    btc_data: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run full backtest. Pure function — same inputs, same outputs."""
    df, indicators = compute_indicators(daily_target, daily_momo, params, btc_data)

    dates = df.index
    cash = money(cap.initial_cash)
    initial_cash = cash

    if len(dates) == 0:
        return BacktestResult(
            equity=pd.Series(dtype=float, name='Equity'),
            journal=pd.DataFrame(),
            trade_log=pd.DataFrame(),
            cash_end=money_to_float(cash),
            open_positions=0,
        )

    mode = determine_initial_mode(dates[0], indicators, params)
    lots = []
    trades = []
    daily_rows = []
    equity_curve = []
    eq_dates = []
    realized_cumulative = Decimal("0")

    def tranche_budget_for(slices: int, base_cash: Decimal) -> Decimal:
        return money(base_cash / Decimal(max(1, slices)))

    m = params.offense if mode == "offense" else params.defense
    tranche_base_cash = cash
    tranche_budget = tranche_budget_for(m.slices, tranche_base_cash)
    prev_close = money(df['Close'].iloc[0])

    peak_equity = float(cash)

    for i, d in enumerate(dates):
        close = money(df.loc[d, 'Close'])

        mode = decide_mode(d, mode, indicators, params)
        m = params.offense if mode == "offense" else params.defense

        tranche_budget = tranche_budget_for(m.slices, tranche_base_cash)

        # LOC Buy (once per day) with share quantity enforcement
        buy_limit = money(prev_close * (ONE + to_decimal(m.buy_cond_pct) / HUNDRED)) if i > 0 else None
        buy_trade_id = None
        buy_qty_executed = Decimal("0")
        buy_amt_value = Decimal("0")
        planned_buy_qty = Decimal("0")
        planned_buy_order_value = Decimal("0")
        if buy_limit is not None and tranche_budget > Decimal("0") and buy_limit > Decimal("0"):
            raw_qty = tranche_budget / buy_limit
            planned_buy_qty = shares(raw_qty, params.allow_fractional_shares)
            if planned_buy_qty > Decimal("0"):
                planned_buy_order_value = money(planned_buy_qty * buy_limit)

        if (
            buy_limit is not None
            and close <= buy_limit
            and tranche_budget > Decimal("0")
            and close > Decimal("0")
        ):
            exec_budget = min(tranche_budget, cash) if params.cash_limited_buy else tranche_budget
            raw_exec_qty = exec_budget / close
            share_qty = shares(raw_exec_qty, params.allow_fractional_shares)
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
                        "매수수량": shares_to_float(share_qty, params.allow_fractional_shares),
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
                    "매도수량": shares_to_float(lot['qty'], params.allow_fractional_shares),
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
            qty_display = shares_to_float(planned_buy_qty, params.allow_fractional_shares)
            buy_summary = f"매수 {qty_display}주 @ {buy_price:.2f} (예산 ${buy_budget:,.2f})"

        sell_summary = "TP대기 없음"
        if lots:
            tp_groups: dict[Decimal, Decimal] = {}
            for lot in lots:
                tp_price = money(lot['tp'])
                tp_groups[tp_price] = tp_groups.get(tp_price, Decimal("0")) + lot['qty']
            sell_entries = [
                f"{shares_to_float(qty, params.allow_fractional_shares)}주 @ {money_to_float(tp):.2f}"
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

        # Netting is display-only
        net_buy_qty = raw_buy_qty
        net_sell_qty = raw_sell_qty
        net_buy_amt = raw_buy_amt
        net_sell_amt = raw_sell_amt
        netting_applied = False
        netting_detail = None

        if params.enable_netting and raw_buy_qty > Decimal("0") and raw_sell_qty > Decimal("0"):
            offset_qty = min(raw_buy_qty, raw_sell_qty)
            if offset_qty > Decimal("0"):
                offset_amt = money(offset_qty * close)
                net_buy_qty = raw_buy_qty - offset_qty
                net_sell_qty = raw_sell_qty - offset_qty
                net_buy_amt = money(max(Decimal("0"), net_buy_amt - offset_amt))
                net_sell_amt = money(max(Decimal("0"), net_sell_amt - offset_amt))
                netting_applied = True
                raw_buy_display = shares_to_float(raw_buy_qty, params.allow_fractional_shares)
                raw_sell_display = shares_to_float(raw_sell_qty, params.allow_fractional_shares)
                if net_buy_qty > Decimal("0"):
                    net_buy_display = shares_to_float(net_buy_qty, params.allow_fractional_shares)
                    netting_detail = f"매수 {raw_buy_display}주, 매도 {raw_sell_display}주 → 순매수 {net_buy_display}주"
                elif net_sell_qty > Decimal("0"):
                    net_sell_display = shares_to_float(net_sell_qty, params.allow_fractional_shares)
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
            "매수수량": shares_to_float(net_buy_qty, params.allow_fractional_shares),
            "매수금액": money_to_float(net_buy_amt) if net_buy_qty > Decimal("0") else 0.0,
            "매수거래ID": buy_trade_id,
            "매도평균": net_sell_avg,
            "매도수량": shares_to_float(net_sell_qty, params.allow_fractional_shares),
            "매도금액": money_to_float(net_sell_amt) if net_sell_qty > Decimal("0") else 0.0,
            "매도거래ID목록": ",".join(str(tid) for tid in sell_trade_ids) if sell_trade_ids else None,
            "실현손익": money_to_float(realized_today),
            "현금": money_to_float(cash),
            "보유수량": shares_to_float(position_qty, params.allow_fractional_shares),
            "평가금액": money_to_float(position_val),
            "Equity": money_to_float(equity),
            "누적손익": money_to_float(realized_cumulative),
            "누적수익률(%)": cumulative_return,
            "낙폭(DD%)": drawdown_pct,
            "일일트렌치예산": money_to_float(tranche_budget),
            "트렌치기반현금": money_to_float(tranche_base_cash),
            "TP평균(보유)": money_to_float(tp_avg_open) if tp_avg_open is not None else None,
            "원매수수량": shares_to_float(raw_buy_qty, params.allow_fractional_shares),
            "원매수금액": money_to_float(raw_buy_amt) if raw_buy_qty > Decimal("0") else 0.0,
            "원매도수량": shares_to_float(raw_sell_qty, params.allow_fractional_shares),
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

    return BacktestResult(
        equity=equity_series,
        journal=daily_df,
        trade_log=trades_df,
        cash_end=money_to_float(cash),
        open_positions=len(lots),
    )

# ---------------------- Metrics ----------------------

def compute_buy_and_hold_return(df: pd.DataFrame) -> float | None:
    """Compute buy-and-hold return percentage from a price DataFrame."""
    if df.empty or "Close" not in df.columns:
        return None
    closes = df["Close"]
    if isinstance(closes, pd.DataFrame):
        closes = closes.squeeze("columns")
    closes = closes.dropna()
    if closes.empty:
        return None
    first_val = closes.iloc[0]
    last_val = closes.iloc[-1]
    try:
        first_val = float(first_val)
        last_val = float(last_val)
    except (TypeError, ValueError):
        return None
    if first_val == 0:
        return None
    return ((last_val / first_val) - 1) * 100.0


def compute_equity_return(series: pd.Series) -> float | None:
    """Compute equity return percentage from an equity series."""
    if series.empty:
        return None
    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    if start == 0:
        return None
    return ((end / start) - 1) * 100.0


def compute_trade_metrics(
    trade_log: pd.DataFrame | None,
    initial_cash: float,
) -> dict[str, float | int | None] | None:
    """Compute realized trade metrics from trade log."""
    if trade_log is None or trade_log.empty:
        return None

    closed = trade_log[trade_log["상태"] == "완료"].copy()
    empty_result = {
        "trade_count": 0,
        "moc_count": 0,
        "net_profit": 0.0,
        "avg_hold_days": None,
        "avg_return_pct": None,
        "avg_gain_pct": None,
        "avg_loss_pct": None,
        "avg_gain": None,
        "avg_loss": None,
        "period_return_pct": None,
    }
    if closed.empty:
        return empty_result

    for col in ("실현손익", "보유기간(일)", "수익률(%)"):
        if col in closed.columns:
            closed[col] = pd.to_numeric(closed[col], errors="coerce")

    closed = closed.dropna(subset=["실현손익"])
    if closed.empty:
        return empty_result

    net_profit = float(closed["실현손익"].sum())
    trade_count = int(len(closed))
    moc_count = int((closed["청산사유"] == "MOC").sum()) if "청산사유" in closed.columns else 0
    avg_hold = float(closed["보유기간(일)"].mean()) if "보유기간(일)" in closed.columns else None
    avg_return_pct = None
    if "수익률(%)" in closed.columns and closed["수익률(%)"].notna().any():
        avg_return_pct = float(closed["수익률(%)"].dropna().mean())

    gain_series = closed.loc[closed["실현손익"] > 0, "실현손익"]
    loss_series = closed.loc[closed["실현손익"] < 0, "실현손익"]
    gain_pct_series = pd.Series(dtype=float)
    loss_pct_series = pd.Series(dtype=float)
    if "수익률(%)" in closed.columns:
        pct_series = pd.to_numeric(closed["수익률(%)"], errors="coerce")
        gain_pct_series = pct_series[pct_series > 0]
        loss_pct_series = pct_series[pct_series < 0]

    avg_gain = float(gain_series.mean()) if not gain_series.empty else None
    avg_loss = float(loss_series.mean()) if not loss_series.empty else None
    avg_gain_pct = float(gain_pct_series.mean()) if not gain_pct_series.empty else None
    avg_loss_pct = float(loss_pct_series.mean()) if not loss_pct_series.empty else None

    period_return_pct = None
    if initial_cash > 0:
        period_return_pct = (net_profit / initial_cash) * 100.0

    return {
        "trade_count": trade_count,
        "moc_count": moc_count,
        "net_profit": net_profit,
        "avg_hold_days": avg_hold,
        "avg_return_pct": avg_return_pct,
        "avg_gain_pct": avg_gain_pct,
        "avg_loss_pct": avg_loss_pct,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "period_return_pct": period_return_pct,
    }


def compute_mode_bands(journal: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive mode periods for chart background coloring."""
    if journal.empty or "모드" not in journal.columns:
        return pd.DataFrame()
    mj = journal[["거래일자", "모드"]].copy()
    mj["거래일자"] = pd.to_datetime(mj["거래일자"])
    mj["grp"] = (mj["모드"] != mj["모드"].shift(1)).cumsum()
    mode_bg = mj.groupby("grp").agg(
        start=("거래일자", "first"),
        end=("거래일자", "last"),
        mode=("모드", "first"),
    ).reset_index(drop=True)
    mode_bg["end"] = mode_bg["end"] + pd.Timedelta(days=1)
    return mode_bg


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
