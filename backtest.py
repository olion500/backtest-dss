
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import yfinance as yf
import json
from datetime import date, datetime, timedelta
from pathlib import Path

from dongpa_engine import (ModeParams, CapitalParams, StrategyParams, DongpaBacktester, summarize)
from chart_utils import (
    EquityPriceChartConfig,
    prepare_equity_price_frames,
    build_equity_price_chart,
)
from ui_common import (
    CONFIG_DIR,
    DEFAULT_PARAMS,
    LOOKBACK_DAYS,
    compute_trade_metrics,
    load_settings,
    render_navigation,
)


def _prepare_defaults(saved: dict, year_start: date, today: date) -> dict:
    result = {
        "start_date": year_start,  # Always use year_start, don't load from config
        "end_date": today,  # Always use today for end date
        "init_cash": 10000,  # Always use default 10000, don't load from config
    }
    for key, default_val in DEFAULT_PARAMS.items():
        if key in ("init_cash",):
            continue  # Already set above
        raw = saved.get(key, default_val)
        if isinstance(default_val, int):
            result[key] = int(raw)
        elif isinstance(default_val, float):
            result[key] = float(raw)
        else:
            result[key] = raw
    return result


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
    saved_values = load_settings()
    if saved_values:
        st.session_state.loaded_defaults = _prepare_defaults(saved_values, year_start, today)
        st.session_state.config_loaded = True

# Determine defaults to use
if st.session_state.config_loaded and st.session_state.loaded_defaults:
    defaults = st.session_state.loaded_defaults
else:
    # Use hardcoded defaults (fallback if no saved settings)
    defaults = {"start_date": year_start, "end_date": today, **DEFAULT_PARAMS}

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
        options=["RSI", "Golden Cross", "ROC", "BTC Overnight"],
        index=int(defaults.get("mode_switch_strategy_index", 0)),
        help="RSI: RSI 기반 모드 전환 | Golden Cross: 이동평균 교차 기반 | ROC: N주 변화율 기반 | BTC Overnight: BTC 야간 수익률 기반 (일일 시그널)"
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
                step=1.0, format="%.1f", help="RSI가 이 값 이상이고 하락 중이면 안전 모드"
            )
            rsi_mid_high = st.number_input(
                "중상 (Mid-High)", value=float(defaults["rsi_mid_high"]),
                step=1.0, format="%.1f", help="RSI가 neutral~이 값 사이이고 상승 중이면 공세 모드"
            )
            rsi_neutral = st.number_input(
                "중립선 (Neutral)", value=float(defaults["rsi_neutral"]),
                step=1.0, format="%.1f", help="RSI 교차 감지 기준선"
            )
            rsi_mid_low = st.number_input(
                "중하 (Mid-Low)", value=float(defaults["rsi_mid_low"]),
                step=1.0, format="%.1f", help="RSI가 이 값~neutral 사이이고 하락 중이면 안전 모드"
            )
            rsi_low_threshold = st.number_input(
                "하한 (Low)", value=float(defaults["rsi_low_threshold"]),
                step=1.0, format="%.1f", help="RSI가 이 값 미만이고 상승 중이면 공세 모드"
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

    roc_period = int(defaults.get("roc_period", 4))
    if mode_switch_strategy == "ROC":
        roc_period = st.number_input(
            "ROC 기간 (주)",
            min_value=1,
            max_value=52,
            value=int(defaults.get("roc_period", 4)),
            step=1,
            help="N주 변화율 기간. 양수면 공세, 음수면 안전 모드"
        )

    btc_ticker = defaults.get("btc_ticker", "BTC-USD")
    btc_lookback_days = int(defaults.get("btc_lookback_days", 1))
    btc_threshold_pct = float(defaults.get("btc_threshold_pct", 0.0))
    if mode_switch_strategy == "BTC Overnight":
        btc_ticker = st.text_input(
            "BTC 티커",
            value=defaults.get("btc_ticker", "BTC-USD"),
            help="비트코인 가격 데이터 티커 (기본: BTC-USD)"
        )
        col_btc1, col_btc2 = st.columns(2)
        btc_lookback_days = col_btc1.number_input(
            "BTC Lookback (일)",
            min_value=1,
            max_value=7,
            value=int(defaults.get("btc_lookback_days", 1)),
            step=1,
            help="BTC 수익률 계산 기간 (캘린더 일수). 1=전일 대비"
        )
        btc_threshold_pct = col_btc2.number_input(
            "임계값 (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(defaults.get("btc_threshold_pct", 0.0)),
            step=0.1,
            format="%.1f",
            help="BTC 수익률이 이 값 초과시 공세, -이 값 미만시 안전. 0=양수면 공세"
        )

    st.divider()

    # Classify config files: start_date 키가 있으면 개인(제외), 없으면 전략
    strategy_files: list[Path] = []
    if CONFIG_DIR.exists():
        for p in sorted(CONFIG_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if p.name == "personal_settings.json":
                continue
            try:
                with p.open("r", encoding="utf-8") as fh:
                    keys = set(json.load(fh).keys())
            except Exception:
                keys = set()
            if "start_date" not in keys:
                strategy_files.append(p)

    st.subheader("📁 전략 설정")
    if strategy_files:
        strat_options = {p.name: p for p in strategy_files}
        strat_names = list(strat_options.keys())
        default_strat_idx = strat_names.index("strategy.json") if "strategy.json" in strat_names else 0

        selected_config_name = st.selectbox(
            "전략 설정 파일",
            options=strat_names,
            index=default_strat_idx,
            help="전략 파라미터(슬라이스, 매수조건, 익절 등)가 담긴 파일",
        )

        if st.button("🔄 전략 설정 불러오기", type="primary", width="stretch"):
            selected_path = strat_options[selected_config_name]
            saved_values = load_settings(selected_path)
            if saved_values:
                st.session_state.loaded_defaults = _prepare_defaults(saved_values, year_start, today)
                st.session_state.config_loaded = True
                st.success(f"✅ '{selected_path.name}' 설정을 불러왔습니다!")
                st.rerun()
            else:
                st.error(f"❌ '{selected_path.name}' 파일을 읽을 수 없습니다.")
    else:
        st.info("전략 설정 파일이 없습니다.")

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
    cash_limited_buy = st.checkbox(
        "현금 한도 매수",
        value=defaults.get("cash_limited_buy", False),
        help="트렌치 예산 > 잔여 현금일 때, 현금 한도 내에서 매수합니다. OFF면 예산 부족 시 매수를 건너뜁니다.",
    )

    st.header("초기 자금")
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

    def _build_strategy_payload() -> dict:
        payload = {
            "target": target,
            "momentum": momentum,
            "bench": bench,
            "cash_limited_buy": cash_limited_buy,
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
            "mode_switch_strategy_index": {"RSI": 0, "Golden Cross": 1, "ROC": 2, "BTC Overnight": 3}[mode_switch_strategy],
            "rsi_high_threshold": float(rsi_high_threshold),
            "rsi_mid_high": float(rsi_mid_high),
            "rsi_neutral": float(rsi_neutral),
            "rsi_mid_low": float(rsi_mid_low),
            "rsi_low_threshold": float(rsi_low_threshold),
        }
        if mode_switch_strategy == "Golden Cross":
            payload["ma_short"] = int(ma_short)
            payload["ma_long"] = int(ma_long)
        elif mode_switch_strategy == "ROC":
            payload["roc_period"] = int(roc_period)
        elif mode_switch_strategy == "BTC Overnight":
            payload["btc_ticker"] = btc_ticker
            payload["btc_lookback_days"] = int(btc_lookback_days)
            payload["btc_threshold_pct"] = float(btc_threshold_pct)
        return payload

    st.divider()
    st.header("💾 전략 설정 저장")
    save_config_name = st.text_input(
        "전략 설정 파일 이름",
        placeholder="예: my_strategy",
        help="전략 파라미터를 config/ 폴더에 JSON 파일로 저장합니다",
    )

    if st.button("💾 전략 설정 저장", type="secondary", width="stretch"):
        reserved = {"default", "strategy", "personal_settings"}
        if not save_config_name or save_config_name.strip() == "":
            st.error("❌ 파일 이름을 입력해주세요!")
        elif save_config_name.strip().lower().removesuffix(".json") in reserved:
            st.error("❌ 예약된 이름입니다. 다른 이름을 사용해주세요!")
        else:
            save_filename = save_config_name.strip()
            if not save_filename.endswith(".json"):
                save_filename += ".json"

            save_path = CONFIG_DIR / save_filename
            CONFIG_DIR.mkdir(exist_ok=True)

            try:
                with save_path.open("w", encoding="utf-8") as fh:
                    json.dump(_build_strategy_payload(), fh, ensure_ascii=False, indent=2)
                st.success(f"✅ 전략 설정이 '{save_filename}'에 저장되었습니다!")
            except Exception as e:
                st.error(f"❌ 저장 실패: {e}")

run = st.button("백테스트 실행")

if run:
    st.info("데이터 로딩 중...")
    df_t = yf.download(target, start=start, end=end, progress=False, auto_adjust=False)
    # Download extra data for RSI/MA warm-up
    momo_start = start - timedelta(days=LOOKBACK_DAYS)
    df_m = yf.download(momentum, start=momo_start, end=end, progress=False, auto_adjust=False)

    # Download BTC data if needed
    df_btc = None
    if mode_switch_strategy == "BTC Overnight":
        btc_start = start - timedelta(days=LOOKBACK_DAYS)
        df_btc = yf.download(btc_ticker, start=btc_start, end=end, progress=False, auto_adjust=False)
        if df_btc.empty:
            st.error(f"BTC 데이터가 비어 있습니다. 티커({btc_ticker})를 확인하세요.")
            st.stop()

    if df_t.empty or df_m.empty:
        st.error("데이터가 비어 있습니다. 티커/기간을 확인하세요.")
        st.stop()
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

        cap = CapitalParams(initial_cash=float(init_cash))

        # Set mode switch strategy parameters
        if mode_switch_strategy == "Golden Cross":
            if ma_short >= ma_long:
                st.error("❌ Short MA는 Long MA보다 작아야 합니다!")
                st.stop()

            params = StrategyParams(
                target_ticker=target,
                momentum_ticker=momentum,
                mode_switch_strategy="ma_cross",
                ma_short_period=int(ma_short),
                ma_long_period=int(ma_long),
                enable_netting=enable_netting,
                allow_fractional_shares=allow_fractional,
                cash_limited_buy=cash_limited_buy,
                defense=defense,
                offense=offense,
            )
        elif mode_switch_strategy == "ROC":
            params = StrategyParams(
                target_ticker=target,
                momentum_ticker=momentum,
                mode_switch_strategy="roc",
                roc_period=int(roc_period),
                enable_netting=enable_netting,
                allow_fractional_shares=allow_fractional,
                cash_limited_buy=cash_limited_buy,
                defense=defense,
                offense=offense,
            )
        elif mode_switch_strategy == "BTC Overnight":
            params = StrategyParams(
                target_ticker=target,
                momentum_ticker=momentum,
                mode_switch_strategy="btc_overnight",
                btc_lookback_days=int(btc_lookback_days),
                btc_threshold_pct=float(btc_threshold_pct),
                enable_netting=enable_netting,
                allow_fractional_shares=allow_fractional,
                cash_limited_buy=cash_limited_buy,
                defense=defense,
                offense=offense,
            )
        else:
            # RSI mode (default)
            params = StrategyParams(
                target_ticker=target,
                momentum_ticker=momentum,
                mode_switch_strategy="rsi",
                rsi_period=14,
                rsi_high_threshold=float(rsi_high_threshold),
                rsi_mid_high=float(rsi_mid_high),
                rsi_neutral=float(rsi_neutral),
                rsi_mid_low=float(rsi_mid_low),
                rsi_low_threshold=float(rsi_low_threshold),
                enable_netting=enable_netting,
                allow_fractional_shares=allow_fractional,
                cash_limited_buy=cash_limited_buy,
                defense=defense,
                offense=offense,
            )

    bt = DongpaBacktester(df_t, df_m, params, cap, btc_data=df_btc)
    res = bt.run()
    eq = res['equity']
    journal = res['journal']
    trade_log = res.get('trade_log')
    trade_metrics = compute_trade_metrics(trade_log, float(init_cash))

    st.success("완료! 가격 데이터는 outputs/ 아래 CSV로 저장되었습니다.")

    summary_metrics = summarize(eq)
    df_m_period = df_m[(df_m.index >= pd.Timestamp(start)) & (df_m.index <= pd.Timestamp(end))]
    momentum_hold_pct = compute_buy_and_hold_return(df_m_period)
    target_hold_pct = compute_buy_and_hold_return(df_t)
    strategy_perf_pct = compute_equity_return(eq)

    st.subheader("Equity Curve vs Target Price")
    eq_df, combined_df = prepare_equity_price_frames(eq, df_t['Close'])

    # Extract offense/defense mode periods for background coloring
    mode_bg = pd.DataFrame()
    if not journal.empty and "모드" in journal.columns:
        mj = journal[["거래일자", "모드"]].copy()
        mj["거래일자"] = pd.to_datetime(mj["거래일자"])
        mj["grp"] = (mj["모드"] != mj["모드"].shift(1)).cumsum()
        mode_bg = mj.groupby("grp").agg(
            start=("거래일자", "first"),
            end=("거래일자", "last"),
            mode=("모드", "first"),
        ).reset_index(drop=True)
        mode_bg["end"] = mode_bg["end"] + pd.Timedelta(days=1)

    chart_config = EquityPriceChartConfig(target_label=target, log_scale=log_scale_enabled)
    chart = build_equity_price_chart(eq_df, combined_df, chart_config, mode_backgrounds=mode_bg)
    if chart is not None:
        st.altair_chart(chart, width="stretch")

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
    st.dataframe(journal, width="stretch", height=360)

    st.download_button("일일 요약 CSV 다운로드", data=journal.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f"dongpa_daily_{target}.csv", mime="text/csv")

    if trade_log is not None and not trade_log.empty:
        st.subheader("트랜치별 매수·매도 기록")
        st.caption("TP=익절, SL=손절, MOC=보유기간 만료 청산")
        st.dataframe(trade_log, width="stretch", height=360)
        st.download_button("트랜치 로그 CSV 다운로드", data=trade_log.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f"dongpa_trades_{target}.csv", mime="text/csv")
    st.download_button("Equity CSV 다운로드", data=eq.to_csv().encode('utf-8'),
                       file_name=f"equity_{target}.csv", mime="text/csv")

    st.caption("일일 요약과 트랜치 로그 모두 한국어 컬럼을 사용합니다. 트랜치 로그의 상태=보유중은 미청산 트랜치입니다. (Equity 등 성과 지표는 영문 표기)")
