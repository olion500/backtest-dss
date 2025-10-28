"""LOC 주문 스케줄 페이지."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

from dongpa_engine import (
    CapitalParams,
    DongpaBacktester,
    ModeParams,
    StrategyParams,
)


SETTINGS_PATH = Path("config") / "order_book_settings.json"
HISTORY_PATH = Path("config") / "order_book_history.csv"
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


def _build_buy_orders(daily_row: pd.Series, available_cash: float | None = None) -> pd.DataFrame:
    limit_price = _safe_float(daily_row.get("매수주문가"))
    tranche_budget = _safe_float(daily_row.get("일일트렌치예산"))
    if limit_price is None or limit_price <= 0 or tranche_budget is None or tranche_budget <= 0:
        return pd.DataFrame()

    effective_budget = tranche_budget
    if available_cash is not None and available_cash > 0:
        effective_budget = min(effective_budget, available_cash)

    qty = int(effective_budget // limit_price)
    raw_qty = _safe_int(daily_row.get("원매수수량", 0))
    if qty <= 0 and raw_qty > 0:
        qty = raw_qty
    if raw_qty > 0:
        qty = min(qty, raw_qty)

    if qty <= 0:
        return pd.DataFrame()

    planned_value = limit_price * qty
    if planned_value > effective_budget:
        planned_value = effective_budget

    record = {
        "주문가": limit_price,
        "수량": qty,
        "예상금액": planned_value,
        "트렌치예산": tranche_budget,
    }
    if available_cash is not None:
        record["사용가능현금"] = available_cash

    return pd.DataFrame([record])


def _build_sell_orders(trades: pd.DataFrame, target_day: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    df = trades.copy()
    df["매수일자"] = pd.to_datetime(df["매수일자"], errors="coerce")
    df["매도일자"] = pd.to_datetime(df["매도일자"], errors="coerce")
    df = df.loc[
        (df["매수일자"] <= target_day)
        & (df["매도일자"].isna() | (df["매도일자"] >= target_day))
    ].copy()
    if df.empty:
        return pd.DataFrame()

    df["매수수량"] = pd.to_numeric(df.get("매수수량"), errors="coerce")
    df["최대보유일"] = pd.to_numeric(df.get("최대보유일"), errors="coerce")
    df = df.dropna(subset=["매수수량", "최대보유일"])
    if df.empty:
        return pd.DataFrame()

    df["매수수량"] = df["매수수량"].astype(int)
    df["최대보유일"] = df["최대보유일"].astype(int)
    df["만료예정일"] = df["매수일자"] + pd.to_timedelta(df["최대보유일"] - 1, unit="D")
    df["잔여일수"] = (df["만료예정일"] - target_day).dt.days
    df["TP목표가"] = pd.to_numeric(df["TP목표가"], errors="coerce")
    df["SL목표가"] = pd.to_numeric(df["SL목표가"], errors="coerce")

    columns = [
        "거래ID",
        "매수일자",
        "매수수량",
        "TP목표가",
        "SL목표가",
        "만료예정일",
        "잔여일수",
        "청산사유",
        "상태",
    ]
    present_cols = [c for c in columns if c in df.columns]
    return df[present_cols].sort_values("거래ID")


def _build_net_summary(daily_row: pd.Series) -> pd.DataFrame:
    data = {
        "순매수수량": _safe_int(daily_row.get("매수수량", 0)),
        "순매도수량": _safe_int(daily_row.get("매도수량", 0)),
        "순매수금액": _safe_float(daily_row.get("매수금액")),
        "순매도금액": _safe_float(daily_row.get("매도금액")),
        "퉁치기적용": bool(daily_row.get("퉁치기적용", False)),
        "퉁치기상세": daily_row.get("퉁치기상세"),
    }
    return pd.DataFrame([data])


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


def _serialize_settings(ui_values: dict) -> dict:
    return ui_values.copy()


def _load_trade_history() -> pd.DataFrame:
    if HISTORY_PATH.exists():
        try:
            df = pd.read_csv(HISTORY_PATH)
            if not df.empty:
                for col in ("매수일자", "매도일자", "만료예정일"):
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
            return df
        except (OSError, pd.errors.ParserError):
            return pd.DataFrame()
    return pd.DataFrame()


def _save_trade_history(new_trades: pd.DataFrame, existing: pd.DataFrame | None = None) -> pd.DataFrame:
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_trades], ignore_index=True)
        dedupe_cols = [col for col in ("거래ID", "매수일자", "매수체결가") if col in combined.columns]
        if dedupe_cols:
            combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
        combined = combined.sort_values(by=dedupe_cols or combined.columns.tolist()).reset_index(drop=True)
    else:
        combined = new_trades.copy().reset_index(drop=True)

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")
    return combined


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
            "target": target,
            "momentum": momentum,
            "bench": bench,
            "enable_netting": enable_netting,
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
        _save_settings(_serialize_settings(settings_payload))
        st.success("설정을 저장했습니다.")


ui_values = {
    "target": target.strip().upper(),
    "momentum": momentum.strip().upper(),
    "bench": bench.strip().upper(),
    "enable_netting": enable_netting,
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


start_fetch = today - timedelta(days=LOOKBACK_DAYS)
end_fetch = today + timedelta(days=1)

with st.spinner("전일 종가와 RSI를 계산하는 중..."):
    df_target = yf.download(
        ui_values["target"],
        start=start_fetch,
        end=end_fetch,
        progress=False,
        auto_adjust=False,
    )
    df_momo = yf.download(
        ui_values["momentum"],
        start=start_fetch,
        end=end_fetch,
        progress=False,
        auto_adjust=False,
    )

if df_target.empty or df_momo.empty:
    st.error("데이터가 비어 있습니다. 티커를 확인하거나 거래 가능일을 기다려 주세요.")
    st.stop()

strategy, capital = _collect_params(ui_values)
backtester = DongpaBacktester(df_target, df_momo, strategy, capital)
result = backtester.run()
journal = result.get("journal", pd.DataFrame())
trade_log = result.get("trade_log", pd.DataFrame())

if journal.empty:
    st.warning("거래 기록이 없습니다.")
    st.stop()

journal["거래일자"] = pd.to_datetime(journal["거래일자"], errors="coerce")
plan_row = journal.iloc[-1].copy()
plan_date = plan_row["거래일자"].date()
plan_timestamp = pd.Timestamp(plan_date)

plan_mode = plan_row.get("모드", "안전")
rsi_value = None
if hasattr(backtester, "daily_rsi") and plan_timestamp in backtester.daily_rsi.index:
    rsi_raw = backtester.daily_rsi.loc[plan_timestamp]
    if isinstance(rsi_raw, pd.Series):
        rsi_raw = rsi_raw.squeeze()
    try:
        if rsi_raw is not None and not pd.isna(rsi_raw):
            rsi_value = float(rsi_raw)
    except (TypeError, ValueError):
        rsi_value = None

prev_close = _safe_float(plan_row.get("종가"))

cash_available = _safe_float(plan_row.get("현금"))
position_qty = _safe_int(plan_row.get("보유수량"))

history_df = _load_trade_history()
open_trades_df = pd.DataFrame()
if not history_df.empty:
    open_trades_df = history_df.copy()
    if "상태" in open_trades_df.columns:
        open_trades_df = open_trades_df[open_trades_df["상태"] != "완료"].copy()

if not open_trades_df.empty:
    for col in ("매수일자", "매도일자", "만료예정일"):
        if col in open_trades_df.columns:
            open_trades_df[col] = pd.to_datetime(open_trades_df[col], errors="coerce")

if history_df.empty:
    cash_available = float(ui_values["init_cash"])
    position_qty = 0
    slices = ui_values["offense_slices"] if plan_mode == "offense" else ui_values["defense_slices"]
    tranche_budget = cash_available / max(1, slices)
    plan_row["일일트렌치예산"] = tranche_budget

    limit_price = _safe_float(plan_row.get("매수주문가"))
    if limit_price is not None and limit_price > 0:
        qty = int(tranche_budget // limit_price)
        if qty > 0:
            buy_value = limit_price * qty
            for col in ("원매수수량", "매수수량", "순매수수량"):
                plan_row[col] = qty
            for col in ("원매수금액", "매수금액", "순매수금액"):
                plan_row[col] = buy_value
        else:
            for col in ("원매수수량", "매수수량", "순매수수량"):
                plan_row[col] = 0
            for col in ("원매수금액", "매수금액", "순매수금액"):
                plan_row[col] = 0.0
else:
    if cash_available is None:
        cash_available = float(ui_values["init_cash"])
    position_qty = position_qty if position_qty is not None else 0

st.subheader(f"{plan_date:%Y-%m-%d} LOC 주문 계획")
mode_label = "공세" if plan_mode == "offense" else "안전"
mode_line = f"현재 모드: **{mode_label}**"
if rsi_value is not None:
    mode_line += f" (주봉 RSI {rsi_value:.2f})"
st.markdown(mode_line)
if prev_close is not None:
    st.markdown(f"전일 종가: **${prev_close:,.2f}**")
if cash_available is not None:
    st.markdown(f"잔여 현금: **${cash_available:,.2f}**, 보유 수량: **{position_qty}주**")

col_buy, col_sell = st.columns(2)

with col_buy:
    st.markdown("**매수 주문**")
    buy_df = _build_buy_orders(plan_row, cash_available)
    if buy_df.empty:
        st.write("매수 주문 없음")
    else:
        st.dataframe(buy_df, use_container_width=True)

with col_sell:
    st.markdown("**매도 주문 (보유 트랜치)**")
    sell_source = open_trades_df if not open_trades_df.empty else pd.DataFrame()
    sell_df = _build_sell_orders(sell_source, plan_timestamp)
    if sell_df.empty:
        if open_trades_df.empty:
            st.write("보유 중인 트랜치가 없습니다.")
        else:
            st.write("매도 주문 없음")
    else:
        st.dataframe(sell_df, use_container_width=True)

st.markdown("**최종 주문 (퉁치기 반영)**")
net_df = _build_net_summary(plan_row)
st.dataframe(net_df, use_container_width=True)

if st.button("이 주문대로 실행 완료", type="primary"):
    plan_date_str = plan_date.strftime("%Y-%m-%d")
    today_trades = pd.DataFrame()
    if not trade_log.empty and "매수일자" in trade_log.columns:
        today_trades = trade_log.loc[trade_log["매수일자"] == plan_date_str].copy()

    if today_trades.empty:
        st.info("오늘 기준으로 저장할 신규 트랜치가 없습니다.")
    else:
        history_df = _save_trade_history(today_trades, history_df if not history_df.empty else None)
        st.success("거래 내역을 저장했습니다.")

st.caption(
    "매도 주문은 선택한 거래일 시작 시점 기준 보유 중인 트랜치의 TP/SL/MOC 설정입니다. "
    "만료 예정일에는 MOC로 종가 청산이 진행됩니다."
)

if not history_df.empty:
    if "거래ID" in history_df.columns:
        history_df = history_df.sort_values("거래ID")
    st.markdown("---")
    st.subheader("누적 거래 내역")
    st.dataframe(history_df, use_container_width=True, height=360)

    metrics = _compute_metrics(history_df, float(ui_values["init_cash"]))
    if metrics is not None:
        st.markdown("**실적 지표**")
        row_top = st.columns(4)
        row_top[0].metric("거래횟수", f"{metrics['trade_count']:,}")
        row_top[1].metric("MOC 횟수", f"{metrics['moc_count']:,}")
        row_top[2].metric("평균 보유일", f"{metrics['avg_hold_days']:.2f}" if metrics['avg_hold_days'] is not None else "-")
        row_top[3].metric("이익금", f"${metrics['net_profit']:,.2f}")

        row_bottom = st.columns(4)
        row_bottom[0].metric("평균 이익률", f"{metrics['avg_gain_pct']:.2f}%" if metrics['avg_gain_pct'] is not None else "-")
        row_bottom[1].metric("평균 손해률", f"{metrics['avg_loss_pct']:.2f}%" if metrics['avg_loss_pct'] is not None else "-")
        row_bottom[2].metric("평균 실현이익", f"${metrics['avg_gain']:,.2f}" if metrics['avg_gain'] is not None else "-")
        row_bottom[3].metric("평균 실현손해", f"${metrics['avg_loss']:,.2f}" if metrics['avg_loss'] is not None else "-")

        if metrics.get("period_return_pct") is not None:
            st.caption(f"초기 가용현금 대비 누적 실현 수익률: {metrics['period_return_pct']:.2f}%")
else:
    st.info("거래 내역이 아직 저장되지 않았습니다. 주문 실행 후 '이 주문대로 실행 완료' 버튼을 눌러 기록하세요.")
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
