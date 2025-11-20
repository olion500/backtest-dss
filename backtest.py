
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import yfinance as yf
import json
from datetime import date, datetime
from pathlib import Path

from dongpa_engine import (ModeParams, CapitalParams, StrategyParams, DongpaBacktester, summarize)
from chart_utils import (
    EquityPriceChartConfig,
    prepare_equity_price_frames,
    build_equity_price_chart,
)


NAV_LINKS = [
    ("backtest.py", "backtest"),
    ("pages/1_Optimizer.py", "Optimizer"),
    ("pages/2_orderBook.py", "orderBook"),
]

SETTINGS_PATH = Path("config") / "order_book_settings.json"
CONFIG_DIR = Path("config")


def get_available_config_files() -> list[Path]:
    """Get all JSON config files in the config directory."""
    if not CONFIG_DIR.exists():
        return []
    # Find all .json files including backups (.json.backup_*)
    json_files = list(CONFIG_DIR.glob("*.json*"))
    # Sort by modification time (newest first)
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files


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


def _load_settings(config_path: Path | None = None) -> dict:
    """Load settings from a config file."""
    path = config_path if config_path else SETTINGS_PATH
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _prepare_defaults(saved: dict, year_start: date, today: date) -> dict:
    return {
        "start_date": year_start,  # Always use year_start, don't load from config
        "end_date": today,  # Always use today for end date
        "target": saved.get("target", "SOXL"),
        "momentum": saved.get("momentum", "QQQ"),
        "bench": saved.get("bench", "SOXX"),
        "log_scale": saved.get("log_scale", True),
        "mode_switch_strategy_index": int(saved.get("mode_switch_strategy_index", 0)),
        "ma_short": int(saved.get("ma_short", 3)),
        "ma_long": int(saved.get("ma_long", 7)),
        "enable_netting": saved.get("enable_netting", True),
        "allow_fractional": saved.get("allow_fractional", False),
        "pcr": float(saved.get("pcr", 0.8)) * 100,  # Convert from decimal to percentage
        "lcr": float(saved.get("lcr", 0.3)) * 100,  # Convert from decimal to percentage
        "cycle": int(saved.get("cycle", 10)),
        "init_cash": 10000,  # Always use default 10000, don't load from config
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


def compute_buy_and_hold_return(df: pd.DataFrame) -> float | None:
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
    if series.empty:
        return None
    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    if start == 0:
        return None
    return ((end / start) - 1) * 100.0


def compute_trade_metrics(trade_log: pd.DataFrame | None, initial_cash: float) -> dict[str, float | int | None] | None:
    if trade_log is None or trade_log.empty:
        return None

    closed = trade_log[trade_log["상태"] == "완료"].copy()
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
        }

    net_profit = float(closed["실현손익"].sum())
    trade_count = int(len(closed))
    moc_count = int((closed["청산사유"] == "MOC").sum()) if "청산사유" in closed.columns else 0
    avg_hold = float(closed["보유기간(일)"].mean()) if "보유기간(일)" in closed.columns else None
    avg_return_pct = None
    if "수익률(%)" in closed.columns and closed["수익률(%)"].notna().any():
        avg_return_pct = float(closed["수익률(%)"].dropna().mean())
    gain_series = closed.loc[closed["실현손익"] > 0, "실현손익"]
    loss_series = closed.loc[closed["실현손익"] < 0, "실현손익"]
    gain_pct_series = closed.loc[closed["수익률(%)"] > 0, "수익률(%)"] if "수익률(%)" in closed.columns else pd.Series(dtype=float)
    loss_pct_series = closed.loc[closed["수익률(%)"] < 0, "수익률(%)"] if "수익률(%)" in closed.columns else pd.Series(dtype=float)
    avg_gain = float(gain_series.mean()) if not gain_series.empty else None
    avg_loss = float(loss_series.mean()) if not loss_series.empty else None
    avg_gain_pct = float(gain_pct_series.mean()) if not gain_pct_series.empty else None
    avg_loss_pct = float(loss_pct_series.mean()) if not loss_pct_series.empty else None

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
    }

st.set_page_config(page_title="backtest", layout="wide")

today = date.today()
year_start = date(today.year, 1, 1)

st.title("backtest")
st.caption("동파법 백테스트 (LOC 전용, 일일 N등분 매수). 주문은 LOC 기준 · 레일 없음 · 하루 최대 1회 매수(트렌치 예산 한도, 정수 주식만). 결과 지표는 Equity 등 영문 용어 사용.")

render_navigation()

# Initialize session state for loaded settings
if "config_loaded" not in st.session_state:
    st.session_state.config_loaded = False
if "loaded_defaults" not in st.session_state:
    st.session_state.loaded_defaults = None

# Auto-load settings on first page load
if not st.session_state.config_loaded:
    saved_values = _load_settings()
    if saved_values:
        st.session_state.loaded_defaults = _prepare_defaults(saved_values, year_start, today)
        st.session_state.config_loaded = True

# Determine defaults to use
if st.session_state.config_loaded and st.session_state.loaded_defaults:
    defaults = st.session_state.loaded_defaults
else:
    # Use hardcoded defaults (fallback if no saved settings)
    defaults = {
        "start_date": year_start,
        "end_date": today,
        "target": "SOXL",
        "momentum": "QQQ",
        "bench": "SOXX",
        "log_scale": True,
        "mode_switch_strategy_index": 0,
        "ma_short": 3,
        "ma_long": 7,
        "enable_netting": True,
        "allow_fractional": False,
        "pcr": 80,
        "lcr": 30,
        "cycle": 10,
        "init_cash": 10000,
        "defense_slices": 7,
        "defense_buy": 3.0,
        "defense_tp": 0.2,
        "defense_sl": 0.0,
        "defense_hold": 30,
        "offense_slices": 7,
        "offense_buy": 5.0,
        "offense_tp": 2.5,
        "offense_sl": 0.0,
        "offense_hold": 7,
    }

with st.sidebar:
    log_scale_enabled = st.toggle(
        "Equity 로그 스케일",
        value=defaults.get("log_scale", True),
        key="equity_log_scale_toggle",
    )
    st.header("기본 설정")

    st.subheader("📊 모드 전환 전략")
    mode_switch_strategy = st.radio(
        "모드 전환 방식",
        options=["RSI", "Golden Cross"],
        index=int(defaults.get("mode_switch_strategy_index", 0)),
        help="RSI: 기존 RSI 기반 모드 전환 | Golden Cross: 이동평균 교차 기반 모드 전환"
    )

    ma_short = None
    ma_long = None
    if mode_switch_strategy == "Golden Cross":
        col_ma1, col_ma2 = st.columns(2)
        ma_short = col_ma1.number_input(
            "Short MA (주)",
            min_value=1,
            max_value=50,
            value=int(defaults.get("ma_short", 3)),
            step=1,
            help="짧은 이동평균 기간 (주 단위)"
        )
        ma_long = col_ma2.number_input(
            "Long MA (주)",
            min_value=2,
            max_value=50,
            value=int(defaults.get("ma_long", 7)),
            step=1,
            help="긴 이동평균 기간 (주 단위)"
        )

        if ma_short >= ma_long:
            st.warning("⚠️ Short MA는 Long MA보다 작아야 합니다!")

    st.divider()

    st.subheader("📁 설정 파일 선택")
    available_configs = get_available_config_files()

    if available_configs:
        config_options = {str(p.name): p for p in available_configs}
        config_names = ["기본 설정 (default)"] + list(config_options.keys())

        selected_config_name = st.selectbox(
            "설정 파일",
            options=config_names,
            help="config/ 폴더의 JSON 파일을 선택하세요"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            load_button = st.button(
                "🔄 선택한 설정 불러오기",
                type="primary",
                use_container_width=True,
                help="선택한 파일의 설정을 불러옵니다 (시작일, 초기현금 제외)"
            )
        with col2:
            if st.button("ℹ️", help="파일 정보 보기"):
                if selected_config_name != "기본 설정 (default)":
                    selected_path = config_options[selected_config_name]
                    file_size = selected_path.stat().st_size
                    from datetime import datetime
                    mod_time = datetime.fromtimestamp(selected_path.stat().st_mtime)
                    st.info(f"**파일**: {selected_path.name}\n\n**크기**: {file_size} bytes\n\n**수정일**: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if load_button:
            if selected_config_name == "기본 설정 (default)":
                st.session_state.loaded_defaults = None
                st.session_state.config_loaded = False
                st.success("기본 설정으로 초기화되었습니다!")
                st.rerun()
            else:
                selected_path = config_options[selected_config_name]
                saved_values = _load_settings(selected_path)
                if saved_values:
                    st.session_state.loaded_defaults = _prepare_defaults(saved_values, year_start, today)
                    st.session_state.config_loaded = True
                    st.success(f"✅ '{selected_path.name}' 설정을 불러왔습니다!")
                    st.rerun()
                else:
                    st.error(f"❌ '{selected_path.name}' 파일을 읽을 수 없습니다.")
    else:
        st.info("config/ 폴더에 설정 파일이 없습니다.")
        if st.button("🔄 기본 설정으로 초기화", type="secondary"):
            st.session_state.loaded_defaults = None
            st.session_state.config_loaded = False
            st.success("기본 설정으로 초기화되었습니다!")
            st.rerun()

    st.divider()

    colA, colB = st.columns(2)
    target = colA.text_input("투자 종목 코드", value=defaults["target"])
    momentum = colB.text_input("모멘텀 종목(주봉 RSI 계산)", value=defaults["momentum"])
    bench = st.text_input("벤치마크(선택)", value=defaults["bench"])
    start = st.date_input("시작일", value=defaults["start_date"])
    end = st.date_input("종료일", value=defaults["end_date"])

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
    pcr = st.number_input("이익복리율 PCR (%)", value=int(defaults["pcr"]), step=1) / 100.0
    lcr = st.number_input("손실복리율 LCR (%)", value=int(defaults["lcr"]), step=1) / 100.0
    cyc = st.number_input("투자금 갱신 주기(거래일)", value=int(defaults["cycle"]), step=1)
    init_cash = st.number_input("초기 가용현금", value=float(defaults["init_cash"]), step=1000.0)

    st.header("안전 모드")
    s1 = st.number_input("분할수(N) - 안전", value=int(defaults["defense_slices"]), step=1)
    cond1 = st.number_input("매수조건(%) - 안전", value=float(defaults["defense_buy"]), step=0.1, format="%.2f")
    tp1 = st.number_input("익절(%) - 안전", value=float(defaults["defense_tp"]), step=0.1, format="%.2f")
    sl1 = st.number_input("손절(%) - 안전", value=float(defaults["defense_sl"]), step=0.1, format="%.2f")
    hold1 = st.number_input("최대 보유일(거래일) - 안전", value=int(defaults["defense_hold"]), step=1)

    st.header("공세 모드")
    s2 = st.number_input("분할수(N) - 공세", value=int(defaults["offense_slices"]), step=1)
    cond2 = st.number_input("매수조건(%) - 공세", value=float(defaults["offense_buy"]), step=0.1, format="%.2f")
    tp2 = st.number_input("익절(%) - 공세", value=float(defaults["offense_tp"]), step=0.1, format="%.2f")
    sl2 = st.number_input("손절(%) - 공세", value=float(defaults["offense_sl"]), step=0.1, format="%.2f")
    hold2 = st.number_input("최대 보유일(거래일) - 공세", value=int(defaults["offense_hold"]), step=1)

    st.divider()
    st.header("💾 설정 저장")
    save_config_name = st.text_input(
        "설정 파일 이름",
        placeholder="예: my_strategy",
        help="설정을 저장할 파일 이름을 입력하세요 (config/ 폴더에 JSON 파일로 저장됩니다)"
    )

    if st.button("💾 설정 저장", type="secondary", use_container_width=True):
        if not save_config_name or save_config_name.strip() == "":
            st.error("❌ 파일 이름을 입력해주세요!")
        elif save_config_name.lower() in ["default", "order_book_settings"]:
            st.error("❌ 'default'와 'order_book_settings'는 예약된 이름입니다. 다른 이름을 사용해주세요!")
        else:
            save_payload = {
                "target": target,
                "momentum": momentum,
                "bench": bench,
                "log_scale": log_scale_enabled,
                "enable_netting": enable_netting,
                "allow_fractional": allow_fractional,
                "pcr": float(pcr),
                "lcr": float(lcr),
                "cycle": int(cyc),
                "defense_slices": int(s1),
                "defense_buy": float(cond1),
                "defense_tp": float(tp1),
                "defense_sl": float(sl1),
                "defense_hold": int(hold1),
                "offense_slices": int(s2),
                "offense_buy": float(cond2),
                "offense_tp": float(tp2),
                "offense_sl": float(sl2),
                "offense_hold": int(hold2),
                "mode_switch_strategy_index": 0 if mode_switch_strategy == "RSI" else 1,
            }

            if mode_switch_strategy == "Golden Cross":
                save_payload["ma_short"] = int(ma_short)
                save_payload["ma_long"] = int(ma_long)

            save_filename = save_config_name.strip()
            if not save_filename.endswith(".json"):
                save_filename += ".json"

            save_path = CONFIG_DIR / save_filename
            CONFIG_DIR.mkdir(exist_ok=True)

            try:
                with save_path.open("w", encoding="utf-8") as fh:
                    json.dump(save_payload, fh, ensure_ascii=False, indent=2)
                st.success(f"✅ 설정이 '{save_filename}'에 저장되었습니다!")
            except Exception as e:
                st.error(f"❌ 저장 실패: {e}")
run = st.button("백테스트 실행")

if run:
    st.info("데이터 로딩 중...")
    df_t = yf.download(target, start=start, end=end, progress=False, auto_adjust=False)
    df_m = yf.download(momentum, start=start, end=end, progress=False, auto_adjust=False)

    if df_t.empty or df_m.empty:
        st.error("데이터가 비어 있습니다. 티커/기간을 확인하세요.")
    else:
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        target_path = outputs_dir / f"{target}_{start:%Y%m%d}_{end:%Y%m%d}_target.csv"
        momo_path = outputs_dir / f"{momentum}_{start:%Y%m%d}_{end:%Y%m%d}_momentum.csv"
        df_t.to_csv(target_path)
        df_m.to_csv(momo_path)

        defense = ModeParams(
            buy_cond_pct=cond1,
            tp_pct=tp1,
            max_hold_days=int(hold1),
            slices=int(s1),
            stop_loss_pct=float(sl1) if sl1 > 0 else None,
        )
        offense = ModeParams(
            buy_cond_pct=cond2,
            tp_pct=tp2,
            max_hold_days=int(hold2),
            slices=int(s2),
            stop_loss_pct=float(sl2) if sl2 > 0 else None,
        )

        cap = CapitalParams(
            initial_cash=float(init_cash),
            refresh_cycle_days=int(cyc),
            profit_compound_rate=float(pcr),
            loss_compound_rate=float(lcr),
            slippage_pct=0.0,
        )

        # Set mode switch strategy parameters
        if mode_switch_strategy == "Golden Cross":
            if ma_short >= ma_long:
                st.error("❌ Short MA는 Long MA보다 작아야 합니다!")
                st.stop()

            params = StrategyParams(
                target_ticker=target,
                momentum_ticker=momentum,
                benchmark_ticker=bench if bench.strip() else None,
                mode_switch_strategy="ma_cross",
                ma_short_period=int(ma_short),
                ma_long_period=int(ma_long),
                reset_on_mode_change=True,
                enable_netting=enable_netting,
                allow_fractional_shares=allow_fractional,
                defense=defense,
                offense=offense,
            )
        else:
            # RSI mode (default)
            params = StrategyParams(
                target_ticker=target,
                momentum_ticker=momentum,
                benchmark_ticker=bench if bench.strip() else None,
            mode_switch_strategy="rsi",
            rsi_period=14,
            reset_on_mode_change=True,
            enable_netting=enable_netting,
            allow_fractional_shares=allow_fractional,
            defense=defense,
            offense=offense,
        )

    bt = DongpaBacktester(df_t, df_m, params, cap)
    res = bt.run()
    eq = res['equity']
    journal = res['journal']
    trade_log = res.get('trade_log')
    trade_metrics = compute_trade_metrics(trade_log, float(init_cash))

    st.success("완료! 가격 데이터는 outputs/ 아래 CSV로 저장되었습니다.")

    summary_metrics = summarize(eq)
    momentum_hold_pct = compute_buy_and_hold_return(df_m)
    target_hold_pct = compute_buy_and_hold_return(df_t)
    strategy_perf_pct = compute_equity_return(eq)

    st.subheader("Equity Curve vs Target Price")
    eq_df, combined_df = prepare_equity_price_frames(eq, df_t['Close'])

    chart_config = EquityPriceChartConfig(target_label=target, log_scale=log_scale_enabled)
    chart = build_equity_price_chart(eq_df, combined_df, chart_config)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)

    st.subheader("요약 지표")
    summary_top = st.columns(4)
    summary_top[0].metric("Final Equity", f"${summary_metrics['Final Equity']:,.0f}")
    summary_top[1].metric("Sharpe (rf=0)", f"{summary_metrics['Sharpe (rf=0)']:.2f}")
    summary_top[2].metric("Volatility (ann)", f"{summary_metrics['Volatility (ann)']:.2%}")
    summary_top[3].metric("Max Drawdown", f"{summary_metrics['Max Drawdown']:.2%}")

    summary_bottom = st.columns(4)
    summary_bottom[0].metric(
        f"{momentum} 보유 수익률",
        f"{momentum_hold_pct:.2f}%" if momentum_hold_pct is not None else "-",
    )
    summary_bottom[1].metric(
        f"{target} 보유 수익률",
        f"{target_hold_pct:.2f}%" if target_hold_pct is not None else "-",
    )
    summary_bottom[2].metric(
        "전략 누적 수익률",
        f"{strategy_perf_pct:.2f}%" if strategy_perf_pct is not None else "-",
    )
    summary_bottom[3].metric("CAGR", f"{summary_metrics['CAGR']:.2%}")

    if trade_metrics is not None:
        st.markdown("---")
        st.subheader("실현 지표")
        tm_row1 = st.columns(4)
        tm_row1[0].metric("거래횟수", f"{trade_metrics['trade_count']:,}")
        tm_row1[1].metric("MOC 횟수", f"{trade_metrics['moc_count']:,}")
        tm_row1[2].metric("평균 보유일", f"{trade_metrics['avg_hold_days']:.2f}" if trade_metrics['avg_hold_days'] is not None else "-")
        tm_row1[3].metric("이익금", f"${trade_metrics['net_profit']:,.2f}")

        tm_row2 = st.columns(4)
        tm_row2[0].metric("평균 이익률", f"{trade_metrics['avg_gain_pct']:.2f}%" if trade_metrics['avg_gain_pct'] is not None else "-")
        tm_row2[1].metric("평균 손해률", f"{trade_metrics['avg_loss_pct']:.2f}%" if trade_metrics['avg_loss_pct'] is not None else "-")
        tm_row2[2].metric("평균 실현이익", f"${trade_metrics['avg_gain']:,.2f}" if trade_metrics['avg_gain'] is not None else "-")
        tm_row2[3].metric("평균 실현손해", f"${trade_metrics['avg_loss']:,.2f}" if trade_metrics['avg_loss'] is not None else "-")
        st.markdown("---")

    st.subheader("일일 거래 요약 (장이 열린 모든 날 포함)")
    st.dataframe(journal, use_container_width=True, height=360)

    st.download_button("일일 요약 CSV 다운로드", data=journal.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f"dongpa_daily_{target}.csv", mime="text/csv")

    if trade_log is not None and not trade_log.empty:
        st.subheader("트랜치별 매수·매도 기록")
        st.caption("TP=익절, SL=손절, MOC=보유기간 만료 청산")
        st.dataframe(trade_log, use_container_width=True, height=360)
        st.download_button("트랜치 로그 CSV 다운로드", data=trade_log.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f"dongpa_trades_{target}.csv", mime="text/csv")
    st.download_button("Equity CSV 다운로드", data=eq.to_csv().encode('utf-8'),
                       file_name=f"equity_{target}.csv", mime="text/csv")

    st.caption("일일 요약과 트랜치 로그 모두 한국어 컬럼을 사용합니다. 트랜치 로그의 상태=보유중은 미청산 트랜치입니다. (Equity 등 성과 지표는 영문 표기)")
