"""LOC 주문 스케줄 페이지."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf

from dongpa_engine import (
    CapitalParams,
    DongpaBacktester,
    ModeParams,
    StrategyParams,
    summarize,
)


SETTINGS_PATH = Path("config") / "order_book_settings.json"
LOOKBACK_DAYS = 400

NAV_LINKS = [
    ("backtest.py", "backtest"),
    ("pages/1_Optimizer.py", "Optimizer"),
    ("pages/2_orderBook.py", "orderBook"),
]


def render_navigation() -> None:
    st.markdown(
        """
        <style>
        [data-testid='stSidebarNav'] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("### Pages")
    for path, label in NAV_LINKS:
        st.sidebar.page_link(path, label=label)
    st.sidebar.divider()


def _load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_settings(payload: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


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


def _prepare_defaults(saved: dict) -> dict:
    return {
        "target": saved.get("target", "SOXL"),
        "momentum": saved.get("momentum", "QQQ"),
        "bench": saved.get("bench", "SOXX"),
        "enable_netting": saved.get("enable_netting", True),
        "pcr": float(saved.get("pcr", 0.80)),
        "lcr": float(saved.get("lcr", 0.30)),
        "cycle": int(saved.get("cycle", 10)),
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

    strategy = StrategyParams(
        target_ticker=ui_values["target"],
        momentum_ticker=ui_values["momentum"],
        benchmark_ticker=ui_values["bench"] if ui_values["bench"].strip() else None,
        rsi_period=14,
        reset_on_mode_change=True,
        enable_netting=ui_values["enable_netting"],
        allow_fractional_shares=ui_values["allow_fractional"],
        defense=defense,
        offense=offense,
    )

    capital = CapitalParams(
        initial_cash=float(ui_values["init_cash"]),
        refresh_cycle_days=int(ui_values["cycle"]),
        profit_compound_rate=float(ui_values["pcr"]),
        loss_compound_rate=float(ui_values["lcr"]),
        slippage_pct=0.0,
    )
    return strategy, capital




def _compute_metrics(trade_log: pd.DataFrame, initial_cash: float) -> dict[str, float | int | None] | None:
    if trade_log is None or trade_log.empty:
        return None

    closed = trade_log[trade_log.get("상태") == "완료"].copy()
    if closed.empty:
        return {
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

    for col in ("실현손익", "보유기간(일)", "수익률(%)"):
        if col in closed.columns:
            closed[col] = pd.to_numeric(closed[col], errors="coerce")

    closed = closed.dropna(subset=["실현손익"])
    if closed.empty:
        return {
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

    net_profit = float(closed["실현손익"].sum())
    trade_count = int(len(closed))
    moc_count = int((closed.get("청산사유") == "MOC").sum()) if "청산사유" in closed.columns else 0
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


st.set_page_config(page_title="orderBook", layout="wide")

render_navigation()


today = date.today()
saved_values = _load_settings()
defaults = _prepare_defaults(saved_values)

st.title("orderBook")
st.caption("동파 LOC 주문 스케줄러. 오늘 기준 LOC 예약 주문과 퉁치기 결과, 누적 실적을 확인합니다.")

with st.sidebar:
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

    st.header("거래 옵션")
    enable_netting = st.checkbox(
        "퉁치기(동일 종가 상쇄)",
        value=defaults["enable_netting"],
        help="같은 날 종가 기준으로 실행된 매수·매도 물량을 순매수/순매도로 상쇄합니다.",
    )
    allow_fractional = st.checkbox(
        "소수점 거래 허용",
        value=defaults.get("allow_fractional", False),
        help="BTC와 같은 자산의 소수점 매수를 허용합니다 (예: 0.00123 BTC). 기본적으로는 정수 주식만 거래합니다.",
    )

    st.header("투자금 갱신 (복리)")
    pcr = st.number_input(
        "이익복리율 PCR (%)",
        value=float(defaults["pcr"] * 100),
        step=1.0,
    ) / 100.0
    lcr = st.number_input(
        "손실복리율 LCR (%)",
        value=float(defaults["lcr"] * 100),
        step=1.0,
    ) / 100.0
    cycle = st.number_input(
        "투자금 갱신 주기(거래일)",
        value=int(defaults["cycle"]),
        step=1,
    )
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
            "enable_netting": enable_netting,
            "allow_fractional": allow_fractional,
            "pcr": pcr,
            "lcr": lcr,
            "cycle": cycle,
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
        }
        _save_settings(settings_payload)
        st.success("설정을 저장했습니다.")


ui_values = {
    "start_date": start_date,
    "target": target.strip().upper(),
    "momentum": momentum.strip().upper(),
    "bench": bench.strip().upper(),
    "enable_netting": enable_netting,
    "allow_fractional": allow_fractional,
    "pcr": pcr,
    "lcr": lcr,
    "cycle": cycle,
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
}


# Calculate data fetch range
# We need extra data before start_date for RSI calculation (at least 100 days for weekly RSI with 14 period)
data_fetch_start = start_date - timedelta(days=LOOKBACK_DAYS)
end_fetch = today + timedelta(days=1)

with st.spinner(f"{start_date}부터 {today}까지 백테스트 실행 중..."):
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

# Filter data to start from start_date for backtesting
df_target_filtered = df_target[df_target.index >= pd.Timestamp(start_date)]
df_momo_filtered = df_momo[df_momo.index >= pd.Timestamp(start_date)]

if df_target_filtered.empty:
    st.error(f"{start_date} 이후 데이터가 없습니다. 시작일을 확인해주세요.")
    st.stop()

strategy, capital = _collect_params(ui_values)
backtester = DongpaBacktester(df_target_filtered, df_momo_filtered, strategy, capital)
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

# Extract current state
current_mode = last_row.get("모드", "안전")
current_cash = _safe_float(last_row.get("현금")) or float(ui_values["init_cash"])
current_position_qty = _safe_int(last_row.get("보유수량"))
prev_close = _safe_float(last_row.get("종가"))
tranche_budget = _safe_float(last_row.get("일일트렌치예산"))

# Get RSI value
rsi_value = None
if hasattr(backtester, "daily_rsi") and last_timestamp in backtester.daily_rsi.index:
    rsi_raw = backtester.daily_rsi.loc[last_timestamp]
    if isinstance(rsi_raw, pd.Series):
        rsi_raw = rsi_raw.squeeze()
    if rsi_raw is not None and not pd.isna(rsi_raw):
        rsi_value = float(rsi_raw)

# Get open positions from trade_log
open_trades = trade_log[trade_log.get("상태") != "완료"].copy() if not trade_log.empty else pd.DataFrame()

st.subheader(f"백테스트 결과 ({start_date} ~ {last_date})")
mode_label = "공세" if current_mode == "offense" else "안전"
mode_line = f"현재 모드: **{mode_label}**"
if rsi_value is not None:
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

            # Check expiration
            buy_date_ts = pd.to_datetime(buy_date, errors="coerce")
            days_left = None
            if pd.notna(buy_date_ts) and max_hold > 0:
                expire_date = buy_date_ts + pd.Timedelta(days=max_hold - 1)
                days_left = (expire_date - last_timestamp).days

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
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)

        # Summary
        total_qty = sum(h["수량"] for h in holdings)
        st.caption(f"총 보유 수량: {total_qty}주 | 보유 종목: {ui_values['target']}")
    else:
        st.write("보유 중인 포지션이 없습니다.")
else:
    st.write("보유 중인 포지션이 없습니다.")

st.markdown("---")

st.subheader(f"다음 거래일 LOC 주문 시트")

# Build unified order sheet
order_sheet = []

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
            # Calculate days left
            buy_date_ts = pd.to_datetime(buy_date, errors="coerce")
            days_left = None
            is_expiring = False
            if pd.notna(buy_date_ts) and max_hold > 0:
                expire_date = buy_date_ts + pd.Timedelta(days=max_hold - 1)
                days_left = (expire_date - last_timestamp).days
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

            # SL sell order
            if sl_price and sl_price > 0:
                sl_change = ((sl_price / buy_price) - 1) * 100 if buy_price else None
                order_sheet.append({
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

# Add buy order (new tranche)
if current_cash > 0 and tranche_budget and tranche_budget > 0:
    mode_params = ui_values["defense_buy"] if current_mode == "defense" else ui_values["offense_buy"]
    buy_limit_price = prev_close * (1 + mode_params / 100) if prev_close else None

    if buy_limit_price and buy_limit_price > 0:
        effective_budget = min(tranche_budget, current_cash)
        buy_qty = int(effective_budget // buy_limit_price)

        if buy_qty > 0:
            # Calculate TP and SL for the new position
            tp_pct = ui_values["defense_tp"] if current_mode == "defense" else ui_values["offense_tp"]
            sl_pct = ui_values["defense_sl"] if current_mode == "defense" else ui_values["offense_sl"]

            new_tp = buy_limit_price * (1 + tp_pct / 100)
            new_sl = buy_limit_price * (1 - sl_pct / 100) if sl_pct > 0 else None

            order_sheet.append({
                "구분": "매수",
                "주문가": buy_limit_price,
                "수량": buy_qty,
                "변화율": f"{mode_params:+.1f}%",
                "비고": f"→ TP: ${new_tp:.2f}, SL: ${new_sl:.2f}" if new_sl else f"→ TP: ${new_tp:.2f}"
            })

# Display order sheet
if order_sheet:
    order_df = pd.DataFrame(order_sheet)

    # Format price column
    order_df["주문가"] = order_df["주문가"].apply(lambda x: f"${x:.2f}")

    st.dataframe(order_df, use_container_width=True, hide_index=True)

    # Netting summary
    net_buy_qty = _safe_int(last_row.get("매수수량", 0))
    net_sell_qty = _safe_int(last_row.get("매도수량", 0))
    netting_applied = last_row.get("퉁치기적용", False)

    if netting_applied:
        st.info(f"✅ 퉁치기 적용: 실제 매수 {net_buy_qty}주, 매도 {net_sell_qty}주로 상쇄됨")
else:
    st.write("예정된 주문이 없습니다.")

st.markdown("---")

# Equity curve and performance metrics
equity = result.get("equity", pd.Series())
if not equity.empty:
    st.subheader("Equity Curve vs Target Price")
    # Prepare equity data
    equity_df = equity.reset_index()
    equity_df.columns = ['Date', 'Equity']

    # Prepare target price data
    target_close = df_target_filtered['Close'].copy()
    if isinstance(target_close, pd.DataFrame):
        target_close = target_close.squeeze("columns")
    target_close = target_close.dropna()

    # Align target price with equity dates
    target_df = target_close.reset_index()
    target_df.columns = ['Date', 'Price']

    # Merge data on Date
    combined_df = pd.merge(equity_df, target_df, on='Date', how='inner')

    if not combined_df.empty:
        # Create hover selection
        hover = alt.selection_point(
            fields=['Date'],
            nearest=True,
            on='mouseover',
            empty=False
        )

        # Create base chart
        base = alt.Chart(combined_df).encode(
            x=alt.X('Date:T', title='Date')
        )

        # Equity line (left y-axis)
        equity_line = base.mark_line(color='steelblue', strokeWidth=2).encode(
            y=alt.Y('Equity:Q',
                   title='Strategy Equity ($)',
                   scale=alt.Scale(type='log'),
                   axis=alt.Axis(titleColor='steelblue', format='$,.0f'))
        )

        # Price line (right y-axis)
        price_line = base.mark_line(color='orange', strokeWidth=2).encode(
            y=alt.Y('Price:Q',
                   title=f'{ui_values["target"]} Price ($)',
                   scale=alt.Scale(type='log'),
                   axis=alt.Axis(titleColor='orange', orient='right', format='$,.2f'))
        )

        # Add points on hover for equity
        equity_points = equity_line.mark_point(size=100, filled=True, color='steelblue').encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0))
        )

        # Add points on hover for price
        price_points = price_line.mark_point(size=100, filled=True, color='orange').encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0))
        )

        # Add vertical rule on hover
        rule = base.mark_rule(color='gray', strokeWidth=1).encode(
            opacity=alt.condition(hover, alt.value(0.7), alt.value(0))
        ).add_params(hover)

        # Add text labels for equity
        equity_text = equity_line.mark_text(
            align='left', dx=5, dy=-10, color='steelblue', fontWeight='bold'
        ).encode(
            text=alt.condition(hover, alt.Text('Equity:Q', format='$,.0f'), alt.value(' ')),
            opacity=alt.condition(hover, alt.value(1), alt.value(0))
        )

        # Add text labels for price
        price_text = price_line.mark_text(
            align='left', dx=5, dy=10, color='orange', fontWeight='bold'
        ).encode(
            text=alt.condition(hover, alt.Text('Price:Q', format='$,.2f'), alt.value(' ')),
            opacity=alt.condition(hover, alt.value(1), alt.value(0))
        )

        # Add date text at top
        date_text = base.mark_text(
            align='center', dx=0, dy=-220, fontSize=14, fontWeight='bold', color='black'
        ).encode(
            text=alt.condition(hover, alt.Text('Date:T', format='%Y-%m-%d'), alt.value(' ')),
            y=alt.value(0)
        )

        # Combine all layers
        chart = alt.layer(
            equity_line, price_line,
            equity_points, price_points,
            rule,
            equity_text, price_text, date_text
        ).resolve_scale(
            y='independent'
        ).properties(height=400).interactive()

        st.altair_chart(chart, use_container_width=True)
    else:
        # Fallback to equity only
        chart = (
            alt.Chart(equity_df)
            .mark_line(color='steelblue', strokeWidth=2)
            .encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Equity:Q', title='Equity ($)', scale=alt.Scale(type='log')),
                tooltip=[
                    alt.Tooltip('Date:T', format='%Y-%m-%d'),
                    alt.Tooltip('Equity:Q', format='$,.2f')
                ]
            )
            .properties(height=400)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    # Calculate summary metrics
    summary_metrics = summarize(equity)

    # Calculate Buy & Hold returns
    target_hold_pct = None
    if not df_target_filtered.empty and "Close" in df_target_filtered.columns:
        closes = df_target_filtered["Close"].dropna()
        if len(closes) > 1:
            target_hold_pct = float(((closes.iloc[-1] / closes.iloc[0]) - 1) * 100.0)

    momo_hold_pct = None
    if not df_momo_filtered.empty and "Close" in df_momo_filtered.columns:
        closes = df_momo_filtered["Close"].dropna()
        if len(closes) > 1:
            momo_hold_pct = float(((closes.iloc[-1] / closes.iloc[0]) - 1) * 100.0)

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
metrics = _compute_metrics(trade_log, float(ui_values["init_cash"]))
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

    st.dataframe(journal_display, use_container_width=True, height=360)
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

    st.dataframe(trade_display, use_container_width=True, height=400)
    st.caption("TP=익절, SL=손절, MOC=보유기간 만료 청산")
else:
    st.write("거래 내역이 없습니다.")

st.caption(
    f"이 페이지는 {start_date}부터 {last_date}까지 백테스트를 실행하여 "
    "현재 포지션과 다음 거래일 LOC 주문 계획을 계산합니다."
)
