"""LOC 주문 스케줄 페이지."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yfinance as yf

from engines.dongpa_engine import (
    _scalar,
    summarize,
    run_backtest,
    compute_indicators,
    compute_buy_and_hold_return,
    compute_equity_return,
    explain_mode,
)
from ui.charts import (
    EquityPriceChartConfig,
    prepare_equity_price_frames,
    build_equity_price_chart,
)
from ui.common import (
    CONFIG_DIR,
    LOCAL_KEYS,
    LOOKBACK_DAYS,
    compute_trade_metrics,
    build_strategy_params,
    get_available_config_files,
    load_settings,
    render_navigation,
    save_settings,
)
from engines.order_book_engine import (
    extract_state,
    build_holdings,
    build_order_sheet,
    apply_netting,
    build_spread_orders,
)


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
        "start_date": saved.get("start_date"),
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
        "roc_period": int(saved.get("roc_period", 4)),
        "btc_ticker": saved.get("btc_ticker", "BTC-USD"),
        "btc_lookback_days": int(saved.get("btc_lookback_days", 1)),
        "btc_threshold_pct": float(saved.get("btc_threshold_pct", 0.0)),
    }


st.set_page_config(page_title="orderBook", layout="wide")

render_navigation()

# Initialize session state for config loading
if "ob_config_loaded" not in st.session_state:
    st.session_state.ob_config_loaded = False
if "ob_loaded_defaults" not in st.session_state:
    st.session_state.ob_loaded_defaults = None

# Use the New York trading date, not the local (KST) date: between local
# midnight and the NY close, date.today() is already one day ahead of the
# NY session, which would pull the live in-progress bar in as a final close.
try:
    today = datetime.now(ZoneInfo("America/New_York")).date()
except Exception:
    today = date.today()

# Determine defaults: use session state if a config was loaded, otherwise default merge
if st.session_state.ob_config_loaded and st.session_state.ob_loaded_defaults:
    defaults = st.session_state.ob_loaded_defaults
    saved_values = st.session_state.ob_loaded_defaults
else:
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

    # Classify config files: start_date 키가 있으면 개인, 없으면 전략
    import json as _json_classify
    all_configs = sorted(
        (p for p in CONFIG_DIR.glob("*.json") if p.name != "personal_settings.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    ) if CONFIG_DIR.exists() else []
    strategy_files: list[Path] = []
    local_files: list[Path] = []
    for p in all_configs:
        try:
            with p.open("r", encoding="utf-8") as fh:
                keys = set(_json_classify.load(fh).keys())
        except Exception:
            keys = set()
        if "start_date" in keys:
            local_files.append(p)
        else:
            strategy_files.append(p)
    # personal_settings.json은 항상 개인 설정 목록 맨 앞
    ls_path = CONFIG_DIR / "personal_settings.json"
    if ls_path.exists():
        local_files.insert(0, ls_path)

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
            key="ob_config_select",
        )

        if st.button("🔄 전략 설정 불러오기", type="primary", width="stretch", key="ob_load_config"):
            selected_path = strat_options[selected_config_name]
            loaded_values = load_settings(selected_path)
            if loaded_values:
                st.session_state.ob_loaded_defaults = _prepare_defaults(loaded_values)
                st.session_state.ob_config_loaded = True
                st.success(f"✅ '{selected_path.name}' 설정을 불러왔습니다!")
                st.rerun()
            else:
                st.error(f"❌ '{selected_path.name}' 파일을 읽을 수 없습니다.")
    else:
        st.info("전략 설정 파일이 없습니다.")

    st.subheader("📌 개인 설정")
    if local_files:
        local_options = {p.name: p for p in local_files}
        local_names = list(local_options.keys())

        selected_local_name = st.selectbox(
            "개인 설정 파일",
            options=local_names,
            help="시작일 · 초기자금 · 종목 등 개인 설정이 담긴 파일",
            key="ob_local_select",
        )

        if st.button("📌 개인 설정 불러오기", width="stretch", key="ob_load_local"):
            local_path = local_options[selected_local_name]
            try:
                with local_path.open("r", encoding="utf-8") as fh:
                    local_data = _json_classify.load(fh)
            except (OSError, ValueError):
                local_data = {}

            if local_data:
                current = dict(defaults)
                local_apply_keys = LOCAL_KEYS | {"target", "momentum", "bench"}
                for k in local_apply_keys:
                    if k in local_data:
                        current[k] = local_data[k]
                st.session_state.ob_loaded_defaults = _prepare_defaults(current)
                st.session_state.ob_config_loaded = True
                st.success(f"✅ '{local_path.name}'에서 개인 설정을 불러왔습니다!")
                st.rerun()
            else:
                st.error(f"❌ '{local_path.name}' 파일을 읽을 수 없습니다.")
    else:
        st.info("개인 설정 파일이 없습니다.")

    st.divider()
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
        options=["RSI", "Golden Cross", "ROC", "BTC Overnight"],
        index=saved_values.get("mode_switch_strategy_index", 0),
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
            help="비트코인 가격 데이터 티커 (기본: BTC-USD)",
            key="ob_btc_ticker",
        )
        col_btc1, col_btc2 = st.columns(2)
        btc_lookback_days = col_btc1.number_input(
            "BTC Lookback (일)",
            min_value=1,
            max_value=7,
            value=int(defaults.get("btc_lookback_days", 1)),
            step=1,
            help="BTC 수익률 계산 기간 (캘린더 일수). 1=전일 대비",
            key="ob_btc_lookback",
        )
        btc_threshold_pct = col_btc2.number_input(
            "임계값 (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(defaults.get("btc_threshold_pct", 0.0)),
            step=0.1,
            format="%.1f",
            help="BTC 수익률이 이 값 초과시 공세, -이 값 미만시 안전. 0=양수면 공세",
            key="ob_btc_threshold",
        )

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

    def _build_settings_payload() -> dict:
        payload = {
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
            "mode_switch_strategy_index": {"RSI": 0, "Golden Cross": 1, "ROC": 2, "BTC Overnight": 3}[mode_switch_strategy],
            "rsi_high_threshold": float(rsi_high_threshold),
            "rsi_mid_high": float(rsi_mid_high),
            "rsi_neutral": float(rsi_neutral),
            "rsi_mid_low": float(rsi_mid_low),
            "rsi_low_threshold": float(rsi_low_threshold),
        }
        if mode_switch_strategy == "Golden Cross":
            payload["ma_short"] = ma_short
            payload["ma_long"] = ma_long
        elif mode_switch_strategy == "ROC":
            payload["roc_period"] = roc_period
        elif mode_switch_strategy == "BTC Overnight":
            payload["btc_ticker"] = btc_ticker
            payload["btc_lookback_days"] = int(btc_lookback_days)
            payload["btc_threshold_pct"] = float(btc_threshold_pct)
        return payload

    if st.button("설정 저장"):
        save_settings(_build_settings_payload())
        st.success("설정을 저장했습니다.")

    st.divider()
    st.header("💾 다른 이름으로 저장")
    save_config_name = st.text_input(
        "설정 파일 이름",
        placeholder="예: my_strategy",
        help="설정을 저장할 파일 이름을 입력하세요 (config/ 폴더에 JSON 파일로 저장됩니다)",
        key="ob_save_config_name",
    )

    if st.button("💾 설정 저장", type="secondary", width="stretch", key="ob_save_as"):
        reserved = {"default", "strategy", "personal_settings"}
        if not save_config_name or save_config_name.strip() == "":
            st.error("❌ 파일 이름을 입력해주세요!")
        elif save_config_name.strip().lower().removesuffix(".json") in reserved:
            st.error("❌ 예약된 이름입니다. 다른 이름을 사용해주세요!")
        else:
            import json as _json

            save_filename = save_config_name.strip()
            if not save_filename.endswith(".json"):
                save_filename += ".json"

            save_path = CONFIG_DIR / save_filename
            CONFIG_DIR.mkdir(exist_ok=True)

            try:
                with save_path.open("w", encoding="utf-8") as fh:
                    _json.dump(_build_settings_payload(), fh, ensure_ascii=False, indent=2)
                st.success(f"✅ 설정이 '{save_filename}'에 저장되었습니다!")
            except Exception as e:
                st.error(f"❌ 저장 실패: {e}")

    st.divider()
    st.header("📌 개인 설정 저장")
    save_local_name = st.text_input(
        "개인 설정 파일 이름",
        placeholder="예: my_local",
        help="시작일 · 초기자금 · 종목 등 개인 설정만 별도 파일로 저장합니다",
        key="ob_save_local_name",
    )

    if st.button("📌 개인 설정 저장", type="secondary", width="stretch", key="ob_save_local"):
        reserved = {"default", "strategy"}
        if not save_local_name or save_local_name.strip() == "":
            st.error("❌ 파일 이름을 입력해주세요!")
        elif save_local_name.strip().lower().removesuffix(".json") in reserved:
            st.error("❌ 예약된 이름입니다. 다른 이름을 사용해주세요!")
        else:
            import json as _json_local

            local_payload = {
                "start_date": start_date.isoformat(),
                "init_cash": init_cash,
                "target": target,
                "momentum": momentum,
                "bench": bench,
                "log_scale": log_scale_enabled,
                "allow_fractional": allow_fractional,
                "enable_netting": enable_netting,
                "spread_buy_levels": spread_buy_levels,
                "spread_buy_step": spread_buy_step,
            }

            local_filename = save_local_name.strip()
            if not local_filename.endswith(".json"):
                local_filename += ".json"

            local_save_path = CONFIG_DIR / local_filename
            CONFIG_DIR.mkdir(exist_ok=True)

            try:
                with local_save_path.open("w", encoding="utf-8") as fh:
                    _json_local.dump(local_payload, fh, ensure_ascii=False, indent=2)
                st.success(f"✅ 개인 설정이 '{local_filename}'에 저장되었습니다!")
            except Exception as e:
                st.error(f"❌ 저장 실패: {e}")


ui_values = {
    "start_date": start_date,
    "target": target.strip().upper(),
    "momentum": momentum.strip().upper(),
    "bench": bench.strip().upper(),
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
    "mode_switch_strategy": mode_switch_strategy,
    "rsi_high_threshold": rsi_high_threshold,
    "rsi_mid_high": rsi_mid_high,
    "rsi_neutral": rsi_neutral,
    "rsi_mid_low": rsi_mid_low,
    "rsi_low_threshold": rsi_low_threshold,
    "btc_ticker": btc_ticker,
    "btc_lookback_days": btc_lookback_days,
    "btc_threshold_pct": btc_threshold_pct,
}

# Add strategy-specific parameters
if mode_switch_strategy == "Golden Cross":
    if ma_short >= ma_long:
        st.error("❌ Short MA는 Long MA보다 작아야 합니다!")
        st.stop()
    ui_values["ma_short"] = ma_short
    ui_values["ma_long"] = ma_long
elif mode_switch_strategy == "ROC":
    ui_values["roc_period"] = roc_period


# Calculate data fetch range
data_fetch_start = start_date - timedelta(days=LOOKBACK_DAYS)

# Check if market has closed today using timezone
market_closed_today = _is_market_closed_today()

if market_closed_today:
    backtest_end_date = today
    end_fetch = today + timedelta(days=1)
    market_started = False
else:
    backtest_end_date = today - timedelta(days=1)
    end_fetch = today
    market_started = True

@st.cache_data(ttl=600, show_spinner=False)
def _download_prices(ticker: str, start, end):
    """Cached yfinance download (TTL 10 min).

    repair=True reconstructs daily bars that Yahoo occasionally returns
    with null OHLC (using intraday data), adding a "Repaired?" column.
    NOTE: repair requires scipy; without it yf.download swallows the
    ImportError and silently returns an empty DataFrame.
    """
    return yf.download(
        ticker, start=start, end=end, progress=False, auto_adjust=False, repair=True
    )


def _repaired_dates(df: pd.DataFrame) -> list:
    """Dates whose daily bar was reconstructed by yfinance repair."""
    if df is None or "Repaired?" not in df.columns.get_level_values(0):
        return []
    rep = df["Repaired?"]
    if isinstance(rep, pd.DataFrame):
        rep = rep.iloc[:, 0]
    mask = rep.fillna(False).astype(bool)
    return [ts.date() for ts in rep.index[mask]]

with st.spinner(f"{start_date}부터 {backtest_end_date}까지 백테스트 실행 중..."):
    df_target = _download_prices(ui_values["target"], data_fetch_start, end_fetch)
    df_momo = _download_prices(ui_values["momentum"], data_fetch_start, end_fetch)
    df_btc = None
    if mode_switch_strategy == "BTC Overnight":
        df_btc = _download_prices(ui_values.get("btc_ticker", "BTC-USD"), data_fetch_start, end_fetch)

if df_target.empty or df_momo.empty:
    st.error("데이터가 비어 있습니다. 티커를 확인하거나 거래 가능일을 기다려 주세요.")
    st.stop()

if mode_switch_strategy == "BTC Overnight" and (df_btc is None or df_btc.empty):
    st.error(f"BTC 데이터가 비어 있습니다. 티커({ui_values.get('btc_ticker', 'BTC-USD')})를 확인하세요.")
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

_repaired = sorted(set(_repaired_dates(df_target_filtered)) | set(_repaired_dates(df_momo_filtered)))
if _repaired:
    _rep_str = ", ".join(d.isoformat() for d in _repaired)
    st.warning(
        f"⚠️ Yahoo 일봉 결손 복구: {_rep_str} 종가는 인트라데이 데이터로 재구성한 근사치입니다. "
        "실제 공식 종가와 소폭 차이가 날 수 있습니다."
    )

strategy, capital = build_strategy_params(ui_values)
bt_result = run_backtest(df_target_filtered, df_momo, strategy, capital, btc_data=df_btc)
_, _indicators = compute_indicators(df_target_filtered, df_momo, strategy, btc_data=df_btc)
journal = bt_result.journal
trade_log = bt_result.trade_log

if journal.empty:
    st.warning("거래 기록이 없습니다.")
    st.stop()

# Extract state via order_book_engine
state = extract_state(journal, _indicators, ui_values, float(ui_values["init_cash"]))

# Get open positions from trade_log
open_trades = trade_log[trade_log.get("상태") != "완료"].copy() if not trade_log.empty else pd.DataFrame()

# Show header
st.subheader(f"백테스트 결과 ({start_date} ~ {state.last_date})")
if market_started:
    st.info(f"⏰ 오늘({today}) 장이 진행 중입니다. {state.last_date}까지의 보유 포지션을 표시하고, 오늘 마감 시 실행될 LOC 주문을 아래에서 확인하세요.")

mode_label = "공세" if state.current_mode == "offense" else "안전"
mode_line = f"현재 모드: **{mode_label}**"

# Show mode indicator based on strategy
last_timestamp = pd.Timestamp(state.last_date)
if ui_values.get("mode_switch_strategy") == "Golden Cross":
    mode_line += f" (Golden Cross 전략: {ui_values['ma_short']}주 × {ui_values['ma_long']}주 MA)"
elif ui_values.get("mode_switch_strategy") == "ROC":
    roc_val = None
    if _indicators.daily_roc is not None and last_timestamp in _indicators.daily_roc.index:
        roc_raw = _scalar(_indicators.daily_roc.loc[last_timestamp])
        if roc_raw is not None and not pd.isna(roc_raw):
            roc_val = float(roc_raw)
    if roc_val is not None:
        mode_line += f" (ROC {roc_val:.4f}, {ui_values.get('roc_period', 4)}주)"
    else:
        mode_line += f" (ROC {ui_values.get('roc_period', 4)}주)"
elif ui_values.get("mode_switch_strategy") == "BTC Overnight":
    btc_sig_val = None
    if _indicators.daily_btc_signal is not None and last_timestamp in _indicators.daily_btc_signal.index:
        btc_sig_raw = _scalar(_indicators.daily_btc_signal.loc[last_timestamp])
        if btc_sig_raw is not None and not pd.isna(btc_sig_raw):
            btc_sig_val = float(btc_sig_raw)
    if btc_sig_val is not None:
        mode_line += f" (BTC signal {btc_sig_val:+.4f}, lookback {ui_values.get('btc_lookback_days', 1)}일)"
    else:
        mode_line += f" (BTC Overnight, lookback {ui_values.get('btc_lookback_days', 1)}일)"
elif state.rsi_value is not None:
    mode_line += f" (주봉 RSI {state.rsi_value:.2f})"

_explanation = explain_mode(last_timestamp, _indicators, strategy)

# Convert markdown bold to HTML
def _md_to_html(md: str) -> str:
    parts = md.split("**")
    out = []
    for i, p in enumerate(parts):
        out.append(f"<b>{p}</b>" if i % 2 == 1 else p)
    return "".join(out)

_mode_html = _md_to_html(mode_line)
_tip_html = _md_to_html(_explanation)

st.markdown(
    '<style>'
    '.mode-hover{position:relative;display:inline-block;cursor:help;'
    'border-bottom:1px dashed rgba(128,128,128,.5)}'
    '.mode-hover .mode-tip{display:none;position:absolute;left:0;top:125%;'
    'background:#262730;color:#fafafa;border:1px solid #4a4a5a;border-radius:8px;'
    'padding:12px 16px;min-width:320px;max-width:480px;z-index:9999;'
    'font-size:.85em;line-height:1.6;box-shadow:0 4px 12px rgba(0,0,0,.25);'
    'white-space:pre-wrap}'
    '.mode-hover:hover .mode-tip{display:block}'
    '</style>'
    f'<span class="mode-hover">{_mode_html}'
    f'<span class="mode-tip">{_tip_html}</span></span>',
    unsafe_allow_html=True,
)
if state.prev_close is not None:
    st.markdown(f"최근 종가 ({state.last_date}): **${state.prev_close:,.2f}**")
st.markdown(f"잔여 현금: **${state.current_cash:,.2f}**, 보유 수량: **{state.current_position_qty}주**")
if state.tranche_budget:
    st.markdown(f"현재 트렌치 예산: **${state.tranche_budget:,.2f}**")

st.markdown("---")

# Show current holdings
st.subheader("보유 포지션 현황")
holdings = build_holdings(open_trades, state.prev_close)
if holdings:
    holdings_df = pd.DataFrame(holdings)
    st.dataframe(holdings_df, width="stretch", hide_index=True)

    total_qty = sum(h["수량"] for h in holdings)
    st.caption(f"총 보유 수량: {total_qty}주 | 보유 종목: {ui_values['target']}")
else:
    st.write("보유 중인 포지션이 없습니다.")

st.markdown("---")

# LOC orders are for next trading day's market close
if market_started:
    st.subheader(f"오늘({today}) 마감 시 실행될 LOC 주문 시트")
    st.caption("아래 주문들은 오늘 장 마감(4:00 PM EST)에 실행됩니다.")
else:
    st.subheader("다음 거래일 LOC 주문 시트")

# Build order sheet via order_book_engine
order_sheet, sl_order_sheet, spread_ctx = build_order_sheet(
    open_trades,
    state.prev_close,
    state.current_cash,
    state.tranche_budget,
    state.current_mode,
    ui_values,
    allow_fractional,
)

# Apply netting
netting_msg = ""
netting_details: list[dict] = []
netting_floor_price = None
netting_scenario_rows: list[dict] = []

if enable_netting:
    netting_result = apply_netting(order_sheet, state.prev_close, allow_fractional)
    order_sheet = netting_result.order_sheet
    netting_msg = netting_result.netting_msg
    netting_details = netting_result.netting_details
    netting_floor_price = netting_result.netting_floor_price
    netting_scenario_rows = netting_result.scenario_rows

# Generate spread buy orders
if spread_ctx is not None:
    spread_rows = build_spread_orders(
        spread_ctx, netting_floor_price, state.prev_close, ui_values, allow_fractional,
    )
    order_sheet.extend(spread_rows)

# Display order sheet
if order_sheet:
    order_df = pd.DataFrame(order_sheet)
    order_df = order_df.sort_values("주문가", ascending=False).reset_index(drop=True)
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

        if netting_scenario_rows:
            st.divider()
            st.markdown("#### 종가 구간별 순결과 (참고용)")
            st.caption(
                "실제 주문이 아닌 시나리오 요약입니다. "
                "위 주문 시트를 전부 예약하면 각 종가 구간에서 아래 순결과가 재현됩니다."
            )
            st.dataframe(pd.DataFrame(netting_scenario_rows), width="stretch", hide_index=True)

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
        st.markdown("#### 주문 시트 읽는 법")
        st.markdown(
            "주문 시트의 모든 행은 **실제 예약할 LOC 주문**입니다. "
            "전부 예약하면 어떤 종가에서도 원래 주문과 동일한 순결과가 나옵니다.\n\n"
            "예시: 매수 \\$100 500주, 매도(TP) \\$98 300주일 때\n"
            "```\n"
            "매도   $100.01  300주  종가 > $100 시 매수미체결 → 매도\n"
            "매수   $100.00  200주  퉁치기 후 순매수 (300주 상쇄)\n"
            "매수    $97.99  300주  종가 < $98 시 매도미체결 → 매수\n"
            "```\n\n"
            "| 종가 구간 | 체결되는 행 | 순결과 |\n"
            "|-----------|------|------|\n"
            "| < \\$98 | 매수 200 + 매수 300 | **매수 500주** |\n"
            "| \\$98 ~ \\$100 | 매수 200 | **순매수 200주** (퉁치기) |\n"
            "| > \\$100 | 매도 300 | **매도 300주** |\n\n"
            "**미러 주문 규칙**\n"
            "- 매수 (하단): 상쇄된 매도가 `- \\$0.01` — 종가가 그 아래면 매도 미체결이므로 원래 매수를 되살림\n"
            "- 매도 (상단): `매수가 + \\$0.01` — 종가가 그 위면 매수 미체결이므로 원래 매도를 되살림\n\n"
            "**퉁치기 불가**: 매도가 > 매수가이면 겹치는 구간이 없어 각각 독립 체결"
        )

        st.divider()
        st.markdown("#### 스프레드 행 제외")
        st.markdown(
            "스프레드 행(`매수 (+N주)` 등)은 \"더 떨어졌을 때\" 시나리오입니다.\n"
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
equity = bt_result.equity
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
    target_hold_pct = compute_buy_and_hold_return(df_target_filtered)
    momo_hold_pct = compute_buy_and_hold_return(df_momo_filtered)
    strategy_pct = compute_equity_return(equity)

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
    f"이 페이지는 {start_date}부터 {state.last_date}까지 백테스트를 실행하여 "
    "현재 포지션과 다음 거래일 LOC 주문 계획을 계산합니다."
)
