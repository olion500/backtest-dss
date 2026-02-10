"""LOC 주문 스케줄 페이지."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yfinance as yf

from dongpa_engine import (
    CapitalParams,
    DongpaBacktester,
    ModeParams,
    StrategyParams,
    _scalar,
    summarize,
)
from chart_utils import (
    EquityPriceChartConfig,
    prepare_equity_price_frames,
    build_equity_price_chart,
)
from ui_common import (
    LOOKBACK_DAYS,
    compute_trade_metrics,
    load_settings,
    render_navigation,
    save_settings,
)


def _safe_int(value: object) -> int:
    try:
        if value is None:
            return 0
        if isinstance(value, float) and math.isnan(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


MODE_ALIASES = {
    "안전": "defense",
    "defense": "defense",
    "공세": "offense",
    "offense": "offense",
}


def _normalize_mode(value: object) -> str:
    """Map journal mode labels (Korean or English) to canonical keys."""
    if isinstance(value, str):
        trimmed = value.strip()
        lowered = trimmed.lower()
        if lowered in MODE_ALIASES:
            return MODE_ALIASES[lowered]
        if trimmed in MODE_ALIASES:
            return MODE_ALIASES[trimmed]
    return "defense"


def _is_market_closed_today() -> bool:
    """Check if US market has closed today.

    US market hours (EST): 9:30 AM - 4:00 PM
    Returns True if current time is after market close (4:00 PM EST)
    """
    try:
        now_utc = datetime.now(ZoneInfo("UTC"))
        now_est = now_utc.astimezone(ZoneInfo("America/New_York"))

        # Market closes at 4:00 PM EST
        market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)

        # If current time is past market close, today's data should be available
        return now_est >= market_close
    except Exception:
        # If timezone conversion fails, assume market hasn't closed (conservative approach)
        return False


def _prepare_defaults(saved: dict) -> dict:
    return {
        "target": saved.get("target", "SOXL"),
        "momentum": saved.get("momentum", "QQQ"),
        "bench": saved.get("bench", "SOXX"),
        "log_scale": saved.get("log_scale", True),
        "allow_fractional": saved.get("allow_fractional", False),
        "enable_netting": saved.get("enable_netting", True),
        "init_cash": float(saved.get("init_cash", 10000)),
        "defense_slices": int(saved.get("defense_slices", 7)),
        "defense_buy": float(saved.get("defense_buy", 3.0)),
        "defense_tp": float(saved.get("defense_tp", 0.2)),
        "defense_sl": float(saved.get("defense_sl", 0.0)),
        "defense_hold": int(saved.get("defense_hold", 30)),
        "offense_slices": int(saved.get("offense_slices", 7)),
        "offense_buy": float(saved.get("offense_buy", 5.0)),
        "offense_tp": float(saved.get("offense_tp", 2.5)),
        "offense_sl": float(saved.get("offense_sl", 0.0)),
        "offense_hold": int(saved.get("offense_hold", 7)),
        "cash_limited_buy": saved.get("cash_limited_buy", False),
        "spread_buy_levels": int(saved.get("spread_buy_levels", 5)),
        "spread_buy_step": int(saved.get("spread_buy_step", 1)),
        "rsi_high_threshold": float(saved.get("rsi_high_threshold", 65.0)),
        "rsi_mid_high": float(saved.get("rsi_mid_high", 60.0)),
        "rsi_neutral": float(saved.get("rsi_neutral", 50.0)),
        "rsi_mid_low": float(saved.get("rsi_mid_low", 40.0)),
        "rsi_low_threshold": float(saved.get("rsi_low_threshold", 35.0)),
    }


def _collect_params(ui_values: dict) -> tuple[StrategyParams, CapitalParams]:
    defense = ModeParams(
        buy_cond_pct=ui_values["defense_buy"],
        tp_pct=ui_values["defense_tp"],
        max_hold_days=int(ui_values["defense_hold"]),
        slices=int(ui_values["defense_slices"]),
        stop_loss_pct=float(ui_values["defense_sl"]) if ui_values["defense_sl"] > 0 else None,
    )
    offense = ModeParams(
        buy_cond_pct=ui_values["offense_buy"],
        tp_pct=ui_values["offense_tp"],
        max_hold_days=int(ui_values["offense_hold"]),
        slices=int(ui_values["offense_slices"]),
        stop_loss_pct=float(ui_values["offense_sl"]) if ui_values["offense_sl"] > 0 else None,
    )

    # Build strategy params based on mode switching strategy
    strategy_dict = {
        "target_ticker": ui_values["target"],
        "momentum_ticker": ui_values["momentum"],
        "benchmark_ticker": ui_values["bench"] if ui_values["bench"].strip() else None,
        "enable_netting": True,
        "allow_fractional_shares": ui_values["allow_fractional"],
        "cash_limited_buy": ui_values.get("cash_limited_buy", False),
        "defense": defense,
        "offense": offense,
    }

    # Add mode switch strategy parameters
    if ui_values.get("mode_switch_strategy") == "Golden Cross":
        strategy_dict.update({
            "mode_switch_strategy": "ma_cross",
            "ma_short_period": int(ui_values["ma_short"]),
            "ma_long_period": int(ui_values["ma_long"]),
        })
    else:
        strategy_dict.update({
            "mode_switch_strategy": "rsi",
            "rsi_period": 14,
            "rsi_high_threshold": float(ui_values.get("rsi_high_threshold", 65.0)),
            "rsi_mid_high": float(ui_values.get("rsi_mid_high", 60.0)),
            "rsi_neutral": float(ui_values.get("rsi_neutral", 50.0)),
            "rsi_mid_low": float(ui_values.get("rsi_mid_low", 40.0)),
            "rsi_low_threshold": float(ui_values.get("rsi_low_threshold", 35.0)),
        })

    strategy = StrategyParams(**strategy_dict)

    capital = CapitalParams(
        initial_cash=float(ui_values["init_cash"]),
        slippage_pct=0.0,
    )
    return strategy, capital




st.set_page_config(page_title="orderBook", layout="wide")

render_navigation()


today = date.today()
saved_values = load_settings()
defaults = _prepare_defaults(saved_values)

st.title("orderBook")
st.caption("동파 LOC 주문 스케줄러. 오늘 기준 LOC 예약 주문과 누적 실적을 확인합니다.")

with st.sidebar:
    log_scale_enabled = st.toggle(
        "Equity 로그 스케일",
        value=defaults.get("log_scale", True),
        key="orderbook_equity_scale_toggle",
    )
    st.header("기본 설정")

    # Start date selection
    default_start = saved_values.get("start_date")
    if default_start:
        try:
            default_start = date.fromisoformat(default_start)
        except (ValueError, TypeError):
            default_start = today - timedelta(days=180)
    else:
        default_start = today - timedelta(days=180)

    start_date = st.date_input(
        "백테스트 시작일",
        value=default_start,
        max_value=today,
        help="이 날짜부터 오늘까지 백테스트를 실행합니다. RSI 계산을 위해 충분히 이전 날짜부터 시작하세요."
    )

    col_a, col_b = st.columns(2)
    target = col_a.text_input("투자 종목 코드", value=defaults["target"])
    momentum = col_b.text_input("모멘텀 종목(주봉 RSI 계산)", value=defaults["momentum"])
    bench = st.text_input("벤치마크(선택)", value=defaults["bench"])

    st.divider()
    st.subheader("📊 모드 전환 전략")
    mode_switch_strategy = st.radio(
        "모드 전환 방식",
        options=["RSI", "Golden Cross"],
        index=saved_values.get("mode_switch_strategy_index", 0),
        help="RSI: 기존 RSI 기반 모드 전환 | Golden Cross: 이동평균 교차 기반 모드 전환"
    )

    rsi_high_threshold = defaults["rsi_high_threshold"]
    rsi_mid_high = defaults["rsi_mid_high"]
    rsi_neutral = defaults["rsi_neutral"]
    rsi_mid_low = defaults["rsi_mid_low"]
    rsi_low_threshold = defaults["rsi_low_threshold"]
    if mode_switch_strategy == "RSI":
        with st.expander("RSI 임계값 설정", expanded=False):
            rsi_high_threshold = st.number_input(
                "상한 (High)", value=float(defaults["rsi_high_threshold"]),
                step=1.0, format="%.1f", help="RSI가 이 값 이상이고 하락 중이면 안전 모드",
                key="ob_rsi_high",
            )
            rsi_mid_high = st.number_input(
                "중상 (Mid-High)", value=float(defaults["rsi_mid_high"]),
                step=1.0, format="%.1f", help="RSI가 neutral~이 값 사이이고 상승 중이면 공세 모드",
                key="ob_rsi_mid_high",
            )
            rsi_neutral = st.number_input(
                "중립선 (Neutral)", value=float(defaults["rsi_neutral"]),
                step=1.0, format="%.1f", help="RSI 교차 감지 기준선",
                key="ob_rsi_neutral",
            )
            rsi_mid_low = st.number_input(
                "중하 (Mid-Low)", value=float(defaults["rsi_mid_low"]),
                step=1.0, format="%.1f", help="RSI가 이 값~neutral 사이이고 하락 중이면 안전 모드",
                key="ob_rsi_mid_low",
            )
            rsi_low_threshold = st.number_input(
                "하한 (Low)", value=float(defaults["rsi_low_threshold"]),
                step=1.0, format="%.1f", help="RSI가 이 값 미만이고 상승 중이면 공세 모드",
                key="ob_rsi_low",
            )

    # Show MA period inputs only if Golden Cross is selected
    ma_short = None
    ma_long = None
    if mode_switch_strategy == "Golden Cross":
        col_ma1, col_ma2 = st.columns(2)
        ma_short = col_ma1.number_input(
            "Short MA (주)",
            min_value=1,
            max_value=50,
            value=saved_values.get("ma_short", 3),
            step=1,
            help="짧은 이동평균 기간 (주 단위)"
        )
        ma_long = col_ma2.number_input(
            "Long MA (주)",
            min_value=2,
            max_value=50,
            value=saved_values.get("ma_long", 7),
            step=1,
            help="긴 이동평균 기간 (주 단위)"
        )

        if ma_short >= ma_long:
            st.warning("⚠️ Short MA는 Long MA보다 작아야 합니다!")

    st.divider()

    st.header("거래 옵션")
    allow_fractional = st.checkbox(
        "소수점 거래 허용",
        value=defaults.get("allow_fractional", False),
        help="BTC와 같은 자산의 소수점 매수를 허용합니다 (예: 0.00123 BTC). 기본적으로는 정수 주식만 거래합니다.",
    )
    enable_netting = st.checkbox(
        "퉁치기 적용",
        value=defaults.get("enable_netting", True),
        help="매수/매도가 동시에 있을 때 겹치는 수량을 상쇄하여 순매수/순매도만 표시합니다.",
    )
    cash_limited_buy = st.checkbox(
        "현금 한도 매수",
        value=defaults.get("cash_limited_buy", False),
        help="트렌치 예산 > 잔여 현금일 때, 현금 한도 내에서 매수합니다. OFF면 예산 부족 시 매수를 건너뜁니다.",
    )
    col_spread1, col_spread2 = st.columns(2)
    spread_buy_levels = col_spread1.number_input(
        "스프레드 레벨",
        min_value=0,
        max_value=20,
        value=int(defaults.get("spread_buy_levels", 5)),
        step=1,
        help="폭락 대비 추가 매수 주문 개수. 공식: 매수가 = 투자금 ÷ (기본수량 + N × 레벨당 수량)",
    )
    spread_buy_step = col_spread2.number_input(
        "레벨당 수량",
        min_value=1,
        max_value=10,
        value=int(defaults.get("spread_buy_step", 1)),
        step=1,
        help="각 레벨에서 매수할 주식 수. 예: 2면 +2주, +4주, +6주...",
    )

    st.header("초기 자금")
    init_cash = st.number_input(
        "초기 가용현금",
        value=float(defaults["init_cash"]),
        step=1000.0,
    )

    st.header("안전 모드")
    def_slice = st.number_input("분할수(N) - 안전", value=int(defaults["defense_slices"]), step=1)
    def_buy = st.number_input("매수조건(%) - 안전", value=float(defaults["defense_buy"]), step=0.1, format="%.2f")
    def_tp = st.number_input("익절(%) - 안전", value=float(defaults["defense_tp"]), step=0.1, format="%.2f")
    def_sl = st.number_input("손절(%) - 안전", value=float(defaults["defense_sl"]), step=0.1, format="%.2f")
    def_hold = st.number_input("최대 보유일(거래일) - 안전", value=int(defaults["defense_hold"]), step=1)

    st.header("공세 모드")
    off_slice = st.number_input("분할수(N) - 공세", value=int(defaults["offense_slices"]), step=1)
    off_buy = st.number_input("매수조건(%) - 공세", value=float(defaults["offense_buy"]), step=0.1, format="%.2f")
    off_tp = st.number_input("익절(%) - 공세", value=float(defaults["offense_tp"]), step=0.1, format="%.2f")
    off_sl = st.number_input("손절(%) - 공세", value=float(defaults["offense_sl"]), step=0.1, format="%.2f")
    off_hold = st.number_input("최대 보유일(거래일) - 공세", value=int(defaults["offense_hold"]), step=1)

    if st.button("설정 저장"):
        settings_payload = {
            "start_date": start_date.isoformat(),
            "target": target,
            "momentum": momentum,
            "bench": bench,
            "log_scale": log_scale_enabled,
            "allow_fractional": allow_fractional,
            "enable_netting": enable_netting,
            "cash_limited_buy": cash_limited_buy,
            "init_cash": init_cash,
            "defense_slices": def_slice,
            "defense_buy": def_buy,
            "defense_tp": def_tp,
            "defense_sl": def_sl,
            "defense_hold": def_hold,
            "offense_slices": off_slice,
            "offense_buy": off_buy,
            "offense_tp": off_tp,
            "offense_sl": off_sl,
            "offense_hold": off_hold,
            "spread_buy_levels": spread_buy_levels,
            "spread_buy_step": spread_buy_step,
            "mode_switch_strategy_index": 0 if mode_switch_strategy == "RSI" else 1,
            "rsi_high_threshold": float(rsi_high_threshold),
            "rsi_mid_high": float(rsi_mid_high),
            "rsi_neutral": float(rsi_neutral),
            "rsi_mid_low": float(rsi_mid_low),
            "rsi_low_threshold": float(rsi_low_threshold),
        }
        if mode_switch_strategy == "Golden Cross":
            settings_payload["ma_short"] = ma_short
            settings_payload["ma_long"] = ma_long
        save_settings(settings_payload)
        st.success("설정을 저장했습니다.")


ui_values = {
    "start_date": start_date,
    "target": target.strip().upper(),
    "momentum": momentum.strip().upper(),
    "bench": bench.strip().upper(),
    "allow_fractional": allow_fractional,
    "cash_limited_buy": cash_limited_buy,
    "init_cash": init_cash,
    "defense_slices": def_slice,
    "defense_buy": def_buy,
    "defense_tp": def_tp,
    "defense_sl": def_sl,
    "defense_hold": def_hold,
    "offense_slices": off_slice,
    "offense_buy": off_buy,
    "offense_tp": off_tp,
    "offense_sl": off_sl,
    "offense_hold": off_hold,
    "spread_buy_levels": spread_buy_levels,
    "spread_buy_step": spread_buy_step,
    "mode_switch_strategy": mode_switch_strategy,
    "rsi_high_threshold": rsi_high_threshold,
    "rsi_mid_high": rsi_mid_high,
    "rsi_neutral": rsi_neutral,
    "rsi_mid_low": rsi_mid_low,
    "rsi_low_threshold": rsi_low_threshold,
}

# Add MA parameters if Golden Cross mode
if mode_switch_strategy == "Golden Cross":
    if ma_short >= ma_long:
        st.error("❌ Short MA는 Long MA보다 작아야 합니다!")
        st.stop()
    ui_values["ma_short"] = ma_short
    ui_values["ma_long"] = ma_long


# Calculate data fetch range
# We need extra data before start_date for RSI calculation (at least 100 days for weekly RSI with 14 period)
data_fetch_start = start_date - timedelta(days=LOOKBACK_DAYS)

# Check if market has closed today using timezone
market_closed_today = _is_market_closed_today()

if market_closed_today:
    # Market has closed, so today's data should be available
    backtest_end_date = today
    end_fetch = today + timedelta(days=1)
    market_started = False
else:
    # Market hasn't closed yet, use yesterday's data
    backtest_end_date = today - timedelta(days=1)
    end_fetch = today
    market_started = True

with st.spinner(f"{start_date}부터 {backtest_end_date}까지 백테스트 실행 중..."):
    df_target = yf.download(
        ui_values["target"],
        start=data_fetch_start,
        end=end_fetch,
        progress=False,
        auto_adjust=False,
    )
    df_momo = yf.download(
        ui_values["momentum"],
        start=data_fetch_start,
        end=end_fetch,
        progress=False,
        auto_adjust=False,
    )

if df_target.empty or df_momo.empty:
    st.error("데이터가 비어 있습니다. 티커를 확인하거나 거래 가능일을 기다려 주세요.")
    st.stop()

# Filter data to start from start_date and end at backtest_end_date
df_target_filtered = df_target[
    (df_target.index >= pd.Timestamp(start_date)) &
    (df_target.index <= pd.Timestamp(backtest_end_date))
]
df_momo_filtered = df_momo[
    (df_momo.index >= pd.Timestamp(start_date)) &
    (df_momo.index <= pd.Timestamp(backtest_end_date))
]

if df_target_filtered.empty:
    st.error(f"{start_date}부터 {backtest_end_date}까지 데이터가 없습니다. 시작일을 확인해주세요.")
    st.stop()

strategy, capital = _collect_params(ui_values)
# Pass full df_momo for proper RSI/MA warm-up; df_target_filtered defines backtest period
backtester = DongpaBacktester(df_target_filtered, df_momo, strategy, capital)
result = backtester.run()
journal = result.get("journal", pd.DataFrame())
trade_log = result.get("trade_log", pd.DataFrame())

if journal.empty:
    st.warning("거래 기록이 없습니다.")
    st.stop()

# Get last trading day state
journal["거래일자"] = pd.to_datetime(journal["거래일자"], errors="coerce")
last_row = journal.iloc[-1].copy()
last_date = last_row["거래일자"].date()
last_timestamp = pd.Timestamp(last_date)

# Extract current state from last row
mode_raw_value = last_row.get("모드", "안전")
current_mode = _normalize_mode(mode_raw_value)
current_cash = _safe_float(last_row.get("현금")) or float(ui_values["init_cash"])
current_position_qty = _safe_int(last_row.get("보유수량"))
prev_close = _safe_float(last_row.get("종가"))
tranche_budget = _safe_float(last_row.get("일일트렌치예산"))

# Get RSI value
rsi_value = None
if hasattr(backtester, "daily_rsi") and last_timestamp in backtester.daily_rsi.index:
    rsi_raw = _scalar(backtester.daily_rsi.loc[last_timestamp])
    if rsi_raw is not None and not pd.isna(rsi_raw):
        rsi_value = float(rsi_raw)

# Get open positions from trade_log
open_trades = trade_log[trade_log.get("상태") != "완료"].copy() if not trade_log.empty else pd.DataFrame()

# Show header
st.subheader(f"백테스트 결과 ({start_date} ~ {last_date})")
if market_started:
    st.info(f"⏰ 오늘({today}) 장이 진행 중입니다. {last_date}까지의 보유 포지션을 표시하고, 오늘 마감 시 실행될 LOC 주문을 아래에서 확인하세요.")

mode_label = "공세" if current_mode == "offense" else "안전"
mode_line = f"현재 모드: **{mode_label}**"

# Show mode indicator based on strategy
if ui_values.get("mode_switch_strategy") == "Golden Cross":
    mode_line += f" (Golden Cross 전략: {ui_values['ma_short']}주 × {ui_values['ma_long']}주 MA)"
elif rsi_value is not None:
    mode_line += f" (주봉 RSI {rsi_value:.2f})"

st.markdown(mode_line)
if prev_close is not None:
    st.markdown(f"최근 종가 ({last_date}): **${prev_close:,.2f}**")
st.markdown(f"잔여 현금: **${current_cash:,.2f}**, 보유 수량: **{current_position_qty}주**")
if tranche_budget:
    st.markdown(f"현재 트렌치 예산: **${tranche_budget:,.2f}**")

st.markdown("---")

# Show current holdings
st.subheader("보유 포지션 현황")
if not open_trades.empty and prev_close:
    holdings = []
    for _, trade in open_trades.iterrows():
        buy_date = trade.get("매수일자", "")
        buy_price = _safe_float(trade.get("매수체결가"))
        buy_qty = _safe_int(trade.get("매수수량", 0))
        tp_price = _safe_float(trade.get("TP목표가"))
        sl_price = _safe_float(trade.get("SL목표가"))
        max_hold = _safe_int(trade.get("최대보유일", 0))

        if buy_qty > 0:
            # Calculate current P&L
            current_value = prev_close * buy_qty if prev_close else 0
            cost_basis = buy_price * buy_qty if buy_price else 0
            unrealized_pnl = current_value - cost_basis
            unrealized_pct = ((prev_close / buy_price) - 1) * 100 if buy_price and prev_close else None

            # Check expiration (trading days from engine's 보유기간)
            hold_period = _safe_int(trade.get("보유기간(일)", 0))
            days_left = None
            if max_hold > 0 and hold_period > 0:
                days_left = max_hold - hold_period

            # Determine status
            status = []
            if tp_price and prev_close and prev_close >= tp_price:
                status.append("TP도달")
            if sl_price and prev_close and prev_close <= sl_price:
                status.append("SL도달")
            if days_left is not None and days_left <= 0:
                status.append("만료")

            holdings.append({
                "매수일자": buy_date,
                "매수가": f"${buy_price:.2f}" if buy_price else "-",
                "수량": buy_qty,
                "현재가": f"${prev_close:.2f}" if prev_close else "-",
                "평가손익": f"${unrealized_pnl:.2f}" if unrealized_pnl else "$0.00",
                "수익률": f"{unrealized_pct:.1f}%" if unrealized_pct is not None else "-",
                "TP": f"${tp_price:.2f}" if tp_price else "-",
                "SL": f"${sl_price:.2f}" if sl_price else "-",
                "잔여일": days_left if days_left is not None else "-",
                "상태": ", ".join(status) if status else "보유중",
            })

    if holdings:
        holdings_df = pd.DataFrame(holdings)
        st.dataframe(holdings_df, width="stretch", hide_index=True)

        # Summary
        total_qty = sum(h["수량"] for h in holdings)
        st.caption(f"총 보유 수량: {total_qty}주 | 보유 종목: {ui_values['target']}")
    else:
        st.write("보유 중인 포지션이 없습니다.")
else:
    st.write("보유 중인 포지션이 없습니다.")

st.markdown("---")

# LOC orders are for next trading day's market close
if market_started:
    st.subheader(f"오늘({today}) 마감 시 실행될 LOC 주문 시트")
    st.caption("아래 주문들은 오늘 장 마감(4:00 PM EST)에 실행됩니다.")
else:
    st.subheader(f"다음 거래일 LOC 주문 시트")

# Build unified order sheet (always use last_row data for LOC orders)
order_sheet = []
sl_order_sheet = []  # displayed in a collapsible panel

# Add sell orders (TP and SL for each open position)
if not open_trades.empty and prev_close:
    for idx, trade in open_trades.iterrows():
        buy_date = trade.get("매수일자", "")
        buy_price = _safe_float(trade.get("매수체결가"))
        buy_qty = _safe_int(trade.get("매수수량", 0))
        tp_price = _safe_float(trade.get("TP목표가"))
        sl_price = _safe_float(trade.get("SL목표가"))
        max_hold = _safe_int(trade.get("최대보유일", 0))

        if buy_qty > 0:
            # Calculate days left (trading days from engine's 보유기간)
            hold_period = _safe_int(trade.get("보유기간(일)", 0))
            days_left = None
            is_expiring = False
            if max_hold > 0 and hold_period > 0:
                days_left = max_hold - hold_period
                is_expiring = days_left <= 0

            # TP sell order
            if tp_price and tp_price > 0:
                tp_change = ((tp_price / buy_price) - 1) * 100 if buy_price else None
                order_sheet.append({
                    "구분": "매도 (TP)",
                    "주문가": tp_price,
                    "수량": buy_qty,
                    "변화율": f"{tp_change:+.1f}%" if tp_change is not None else "-",
                    "비고": f"매수일: {buy_date}, 매수가: ${buy_price:.2f}" if buy_price else ""
                })

            # SL sell order (render separately to reduce clutter)
            if sl_price and sl_price > 0:
                sl_change = ((sl_price / buy_price) - 1) * 100 if buy_price else None
                sl_order_sheet.append({
                    "구분": "매도 (SL)",
                    "주문가": sl_price,
                    "수량": buy_qty,
                    "변화율": f"{sl_change:+.1f}%" if sl_change is not None else "-",
                    "비고": f"매수일: {buy_date}, 매수가: ${buy_price:.2f}" if buy_price else ""
                })

            # Expiration sell order (if near expiration)
            if is_expiring:
                order_sheet.append({
                    "구분": "매도 (만료)",
                    "주문가": prev_close,
                    "수량": buy_qty,
                    "변화율": f"{((prev_close / buy_price) - 1) * 100:+.1f}%" if buy_price and prev_close else "-",
                    "비고": f"잔여일: {days_left}일"
                })

# Add buy order (new tranche) + spread at lower prices
if current_cash > 0 and tranche_budget and tranche_budget > 0:
    mode_params = ui_values["defense_buy"] if current_mode == "defense" else ui_values["offense_buy"]
    buy_limit_price = prev_close * (1 + mode_params / 100) if prev_close else None

    if buy_limit_price and buy_limit_price > 0:
        effective_budget = min(tranche_budget, current_cash)
        tp_pct = ui_values["defense_tp"] if current_mode == "defense" else ui_values["offense_tp"]
        sl_pct = ui_values["defense_sl"] if current_mode == "defense" else ui_values["offense_sl"]

        # Base buy order at limit price
        if allow_fractional:
            base_qty = effective_budget / buy_limit_price
        else:
            base_qty = int(effective_budget // buy_limit_price)

        if base_qty > 0:
            new_tp = buy_limit_price * (1 + tp_pct / 100)
            new_sl = buy_limit_price * (1 - sl_pct / 100) if sl_pct > 0 else None

            order_sheet.append({
                "구분": "매수",
                "주문가": buy_limit_price,
                "수량": base_qty,
                "변화율": f"{mode_params:+.1f}%",
                "비고": f"→ TP: ${new_tp:.2f}, SL: ${new_sl:.2f}" if new_sl else f"→ TP: ${new_tp:.2f}"
            })

            # Spread rows using formula: price = daily_budget / (base_qty + N * step)
            # Each row represents buying `step` additional shares at that price level
            # Formula: 추가 매수 가격 = 일일 투자금 ÷ (기본 수량 + N × step)
            daily_budget = effective_budget
            max_spread_orders = ui_values.get("spread_buy_levels", 5)
            spread_step = ui_values.get("spread_buy_step", 1)
            min_drop_pct = -50.0  # Stop adding spread orders beyond 50% drop

            for n in range(1, max_spread_orders + 1):
                shares_increment = n * spread_step
                spread_price = daily_budget / (base_qty + shares_increment)

                # Calculate drop percentage from base price
                drop_pct = ((spread_price / buy_limit_price) - 1) * 100
                if drop_pct < min_drop_pct:
                    break

                spread_tp = spread_price * (1 + tp_pct / 100)
                spread_sl = spread_price * (1 - sl_pct / 100) if sl_pct > 0 else None
                pct_from_prev = ((spread_price / prev_close) - 1) * 100 if prev_close else 0

                note = f"TP: ${spread_tp:.2f}"
                if spread_sl:
                    note += f", SL: ${spread_sl:.2f}"

                order_sheet.append({
                    "구분": f"매수 (+{shares_increment}주)",
                    "주문가": spread_price,
                    "수량": spread_step,
                    "변화율": f"{pct_from_prev:+.1f}%",
                    "비고": note,
                })

# Apply netting: offset matching sell and base-buy quantities in-place
# IMPORTANT: Netting only applies when sell_price <= buy_price (overlapping execution range)
# LOC buy executes if close <= buy_price, LOC sell executes if close >= sell_price
# Both can execute at the same close only when sell_price <= close <= buy_price
netting_msg = ""
netting_details: list[dict] = []  # tracks per-row netting for debugging

if enable_netting:
    sell_indices = [i for i, r in enumerate(order_sheet) if r["구분"].startswith("매도")]
    buy_index = next((i for i, r in enumerate(order_sheet) if r["구분"] == "매수"), None)

    if buy_index is not None and sell_indices:
        buy_price = float(order_sheet[buy_index]["주문가"])
        total_buy_qty = float(order_sheet[buy_index]["수량"])
        fmt_qty = (lambda q: f"{q:,.4f}") if allow_fractional else (lambda q: f"{int(q):,}")

        # Only net sell orders where sell_price <= buy_price (overlapping range)
        nettable_sell_indices = []
        non_nettable_sell_indices = []
        for i in sell_indices:
            sell_price = float(order_sheet[i]["주문가"])
            if sell_price <= buy_price:
                nettable_sell_indices.append(i)
            else:
                non_nettable_sell_indices.append(i)

        nettable_sell_qty = sum(float(order_sheet[i]["수량"]) for i in nettable_sell_indices)

        if nettable_sell_qty > 0 and total_buy_qty > 0:
            offset = min(nettable_sell_qty, total_buy_qty)

            # Cash impact for nettable orders only
            sell_amt = sum(float(order_sheet[i]["주문가"]) * float(order_sheet[i]["수량"]) for i in nettable_sell_indices)
            buy_amt = buy_price * min(total_buy_qty, nettable_sell_qty)
            cash_impact = sell_amt - buy_amt  # positive = inflow
            cash_str = f"순 유입 ${cash_impact:,.2f}" if cash_impact >= 0 else f"순 유출 ${-cash_impact:,.2f}"

            if total_buy_qty >= nettable_sell_qty:
                # Buy side larger: reduce buy qty, remove nettable sell rows
                net_buy = total_buy_qty - offset
                if not allow_fractional:
                    net_buy = int(net_buy)
                for i in nettable_sell_indices:
                    row = order_sheet[i]
                    qty = float(row["수량"])
                    netting_details.append({
                        "매도": row["구분"],
                        "매도가": float(row["주문가"]),
                        "매수가": buy_price,
                        "상쇄 수량": qty,
                        "사유": f"매도가 ${float(row['주문가']):.2f} ≤ 매수가 ${buy_price:.2f}",
                    })
                    order_sheet[i] = None
                if net_buy > 0:
                    order_sheet[buy_index]["수량"] = net_buy
                else:
                    order_sheet[buy_index] = None
                if net_buy > 0:
                    netting_msg = f"퉁치기 적용: 매도 {fmt_qty(nettable_sell_qty)}주 상쇄 → 순매수 {fmt_qty(net_buy)}주 ({cash_str})"
                else:
                    netting_msg = f"퉁치기 적용: 매수·매도 {fmt_qty(total_buy_qty)}주 완전상쇄 ({cash_str})"
            else:
                # Sell side larger: remove buy row, reduce nettable sell rows sequentially
                order_sheet[buy_index] = None
                remaining = total_buy_qty
                for i in nettable_sell_indices:
                    row_qty = float(order_sheet[i]["수량"])
                    reduction = min(row_qty, remaining)
                    new_qty = row_qty - reduction
                    remaining -= reduction
                    if not allow_fractional:
                        new_qty = int(new_qty)
                    if reduction > 0:
                        netting_details.append({
                            "매도": order_sheet[i]["구분"],
                            "매도가": float(order_sheet[i]["주문가"]),
                            "매수가": buy_price,
                            "상쇄 수량": reduction,
                            "사유": f"매도가 ${float(order_sheet[i]['주문가']):.2f} ≤ 매수가 ${buy_price:.2f}",
                        })
                    if new_qty > 0:
                        order_sheet[i]["수량"] = new_qty
                    else:
                        order_sheet[i] = None
                    if remaining <= 0:
                        break
                net_sell = nettable_sell_qty - offset
                netting_msg = f"퉁치기 적용: 매수 {fmt_qty(total_buy_qty)}주 상쇄 → 순매도 {fmt_qty(net_sell)}주 ({cash_str})"

            # Note about non-nettable sells
            if non_nettable_sell_indices:
                non_nettable_qty = sum(float(order_sheet[i]["수량"]) for i in non_nettable_sell_indices if order_sheet[i] is not None)
                if non_nettable_qty > 0:
                    netting_msg += f" | 퉁치기 불가 매도 {fmt_qty(non_nettable_qty)}주 (매도가 > 매수가)"

            order_sheet = [r for r in order_sheet if r is not None]

# Display order sheet
if order_sheet:
    order_df = pd.DataFrame(order_sheet)
    order_df["주문가"] = order_df["주문가"].apply(lambda x: f"${x:.2f}")
    st.dataframe(order_df, width="stretch", hide_index=True)
    if netting_msg:
        st.caption(netting_msg)
elif netting_msg:
    st.info(netting_msg)
else:
    st.write("예정된 주문이 없습니다.")

# Show netting breakdown in expander for debugging
if netting_details:
    with st.expander("퉁치기 상세 내역", expanded=False):
        st.markdown("#### 이번 상쇄 내역")
        net_df = pd.DataFrame(netting_details)
        net_df["매도가"] = net_df["매도가"].apply(lambda x: f"${x:.2f}")
        net_df["매수가"] = net_df["매수가"].apply(lambda x: f"${x:.2f}")
        fmt = (lambda v: f"{v:,.4f}") if allow_fractional else (lambda v: f"{int(v):,}" if v == int(v) else f"{v:,.1f}")
        net_df["상쇄 수량"] = net_df["상쇄 수량"].apply(fmt)
        st.dataframe(net_df, width="stretch", hide_index=True)

        st.divider()
        st.markdown("#### 퉁치기 동작 원리")
        st.markdown(
            "LOC 주문은 모두 **장 마감가**에 체결됩니다.\n\n"
            "**핵심 원칙**: 매도가 ≤ 매수가 일 때만 퉁치기 발생\n"
            "- LOC 매수: 종가 ≤ 매수가이면 체결\n"
            "- LOC 매도: 종가 ≥ 매도가이면 체결\n"
            "- 둘이 동시 체결되려면: 매도가 ≤ 종가 ≤ 매수가\n"
            "- 따라서 **매도가 > 매수가**이면 겹치는 구간이 없어 퉁치기 불가"
        )

        st.divider()
        st.markdown("#### Case 1 — 매도가 < 매수가 (퉁치기 발생)")
        st.markdown(
            "```\n"
            "매수 $100 500주 / 매도 $98 300주\n"
            "→ $98~$100 구간에서 둘 다 체결 가능\n"
            "→ 순매수 200주\n"
            "```"
        )

        st.divider()
        st.markdown("#### Case 2 — 매도가 > 매수가 (퉁치기 없음)")
        st.markdown(
            "```\n"
            "매수 $100 500주 / 매도 $105 300주\n"
            "→ 겹치는 구간 없음 (종가가 둘 다 체결시키는 가격이 존재하지 않음)\n"
            "→ 각각 독립 체결, 퉁치기 불가\n"
            "```"
        )

        st.divider()
        st.markdown("#### Case 3 — 매도가 = 매수가")
        st.markdown(
            "```\n"
            "매수 $100 500주 / 매도 $100 300주\n"
            "→ 종가가 정확히 $100일 때만 둘 다 체결\n"
            "→ 순매수 200주\n"
            "```"
        )

        st.divider()
        st.markdown("#### Case 4 — 여러 주문 혼합")
        st.markdown(
            "```\n"
            "매수 $100 500주, $95 300주\n"
            "매도 $98 200주, $102 400주\n"
            "\n"
            "매도 $98 vs 매수 $100: $98 ≤ $100 → 퉁치기 O\n"
            "매도 $102 vs 매수 $100: $102 > $100 → 퉁치기 X\n"
            "```"
        )

        st.divider()
        st.markdown("#### 스프레드 행 제외")
        st.markdown(
            "스프레드 행(`매수 (-3%)` 등)은 \"더 떨어졌을 때\" 시나리오입니다.\n"
            "기본 매수와 동시에 체결되지 않으므로 퉁치기 대상에서 제외됩니다."
        )

# Show SL orders in a collapsible table to keep the main sheet compact
if sl_order_sheet:
    with st.expander("매도 SL 주문 보기", expanded=False):
        sl_df = pd.DataFrame(sl_order_sheet)
        sl_df["주문가"] = sl_df["주문가"].apply(lambda x: f"${x:.2f}")
        st.dataframe(sl_df, width="stretch", hide_index=True)

st.markdown("---")

# Equity curve and performance metrics
equity = result.get("equity", pd.Series())
if not equity.empty:
    st.subheader("Equity Curve vs Target Price")
    eq_df, combined_df = prepare_equity_price_frames(equity, df_target_filtered['Close'])
    chart_config = EquityPriceChartConfig(
        target_label=ui_values['target'],
        log_scale=log_scale_enabled,
    )
    chart = build_equity_price_chart(eq_df, combined_df, chart_config)
    if chart is not None:
        st.altair_chart(chart, width="stretch")

    # Calculate summary metrics
    summary_metrics = summarize(equity)

    # Calculate Buy & Hold returns
    target_hold_pct = None
    if not df_target_filtered.empty and "Close" in df_target_filtered.columns:
        closes = df_target_filtered["Close"].dropna()
        if isinstance(closes, pd.DataFrame):
            closes = closes.squeeze("columns")
        if len(closes) > 1:
            start_price = closes.iloc[0]
            end_price = closes.iloc[-1]
            target_hold_pct = ((float(end_price) / float(start_price)) - 1) * 100.0

    momo_hold_pct = None
    if not df_momo_filtered.empty and "Close" in df_momo_filtered.columns:
        closes = df_momo_filtered["Close"].dropna()
        if isinstance(closes, pd.DataFrame):
            closes = closes.squeeze("columns")
        if len(closes) > 1:
            start_price = closes.iloc[0]
            end_price = closes.iloc[-1]
            momo_hold_pct = ((float(end_price) / float(start_price)) - 1) * 100.0

    strategy_pct = None
    if len(equity) > 1:
        strategy_pct = float(((equity.iloc[-1] / equity.iloc[0]) - 1) * 100.0)

    st.subheader("요약 지표")
    summary_top = st.columns(4)
    summary_top[0].metric("Final Equity", f"${summary_metrics['Final Equity']:,.0f}")
    summary_top[1].metric("Sharpe (rf=0)", f"{summary_metrics['Sharpe (rf=0)']:.2f}")
    summary_top[2].metric("Volatility (ann)", f"{summary_metrics['Volatility (ann)']:.2%}")
    summary_top[3].metric("Max Drawdown", f"{summary_metrics['Max Drawdown']:.2%}")

    summary_bottom = st.columns(4)
    summary_bottom[0].metric(
        f"{ui_values['momentum']} 보유 수익률",
        f"{momo_hold_pct:.2f}%" if momo_hold_pct is not None else "-",
    )
    summary_bottom[1].metric(
        f"{ui_values['target']} 보유 수익률",
        f"{target_hold_pct:.2f}%" if target_hold_pct is not None else "-",
    )
    summary_bottom[2].metric(
        "전략 누적 수익률",
        f"{strategy_pct:.2f}%" if strategy_pct is not None else "-",
    )
    summary_bottom[3].metric("CAGR", f"{summary_metrics['CAGR']:.2%}")

# Trade metrics
metrics = compute_trade_metrics(trade_log, float(ui_values["init_cash"]))
if metrics:
    st.markdown("---")
    st.subheader("실현 지표")
    tm_row1 = st.columns(4)
    tm_row1[0].metric("거래횟수", f"{metrics['trade_count']:,}")
    tm_row1[1].metric("MOC 횟수", f"{metrics['moc_count']:,}")
    tm_row1[2].metric("평균 보유일", f"{metrics['avg_hold_days']:.2f}" if metrics['avg_hold_days'] is not None else "-")
    tm_row1[3].metric("이익금", f"${metrics['net_profit']:,.2f}")

    tm_row2 = st.columns(4)
    tm_row2[0].metric("평균 이익률", f"{metrics['avg_gain_pct']:.2f}%" if metrics['avg_gain_pct'] is not None else "-")
    tm_row2[1].metric("평균 손해률", f"{metrics['avg_loss_pct']:.2f}%" if metrics['avg_loss_pct'] is not None else "-")
    tm_row2[2].metric("평균 실현이익", f"${metrics['avg_gain']:,.2f}" if metrics['avg_gain'] is not None else "-")
    tm_row2[3].metric("평균 실현손해", f"${metrics['avg_loss']:,.2f}" if metrics['avg_loss'] is not None else "-")
else:
    st.info("완료된 거래가 없습니다.")

# Show detailed logs
st.markdown("---")
st.subheader("일일 거래 요약")
if not journal.empty:
    # Filter out technical columns
    display_cols = [col for col in journal.columns
                   if col not in ["row_id", "거래ID"] and not col.startswith("_")]
    journal_display = journal[display_cols].copy()

    # Format date columns to show only date (no time)
    for col in journal_display.columns:
        if "일자" in col or "날짜" in col:
            journal_display[col] = pd.to_datetime(journal_display[col], errors="coerce").dt.date

    # Format money columns with $ and 2 decimal places
    money_keywords = ["금액", "현금", "예산", "가", "손익", "체결가", "목표가", "주문가"]
    for col in journal_display.columns:
        if any(keyword in col for keyword in money_keywords):
            journal_display[col] = journal_display[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x
            )

    st.dataframe(journal_display, width="stretch", height=360)
else:
    st.write("거래 요약이 없습니다.")

st.subheader("트랜치별 매수·매도 기록")
if not trade_log.empty:
    # Filter out technical columns
    display_cols = [col for col in trade_log.columns
                   if col not in ["row_id", "거래ID"] and not col.startswith("_")]
    trade_display = trade_log[display_cols].copy()

    # Format date columns to show only date (no time)
    for col in trade_display.columns:
        if "일자" in col or "날짜" in col or "예정일" in col:
            trade_display[col] = pd.to_datetime(trade_display[col], errors="coerce").dt.date

    # Format money columns with $ and 2 decimal places
    money_keywords = ["금액", "현금", "예산", "가", "손익", "체결가", "목표가", "주문가"]
    for col in trade_display.columns:
        if any(keyword in col for keyword in money_keywords):
            trade_display[col] = trade_display[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x
            )

    st.dataframe(trade_display, width="stretch", height=400)
    st.caption("TP=익절, SL=손절, MOC=보유기간 만료 청산")
else:
    st.write("거래 내역이 없습니다.")

st.caption(
    f"이 페이지는 {start_date}부터 {last_date}까지 백테스트를 실행하여 "
    "현재 포지션과 다음 거래일 LOC 주문 계획을 계산합니다."
)
