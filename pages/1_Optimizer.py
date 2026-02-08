"""Streamlit page for running the Dongpa parameter optimizer."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dongpa_optimizer import (
    CapitalParamRanges,
    MAPeriodRanges,
    ModeParamRanges,
    OptimizerConfig,
    ParamRange,
    optimize,
)

NAV_LINKS = [
    ("backtest.py", "backtest"),
    ("pages/1_Optimizer.py", "Optimizer"),
    ("pages/2_orderBook.py", "orderBook"),
    ("pages/3_Optuna.py", "Optuna"),
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


st.set_page_config(page_title="Optimizer", layout="wide")

render_navigation()

st.title("동파 파라미터 최적화 (Random Search)")
st.caption(
    "사용자가 설정한 기간으로 학습 및 테스트 구간을 분리하여 최적화합니다. "
    "각 조합은 학습·테스트에서의 CAGR과 MDD를 계산하고, 평균 CAGR에서 평균 MDD×가중치(기본 0.6)를 뺀 점수로 순위를 매깁니다."
)

st.write(
    "랜덤 샘플링 방식을 사용하여 설정한 범위 내에서 파라미터를 랜덤하게 선택합니다. "
    "샘플 개수를 조절하여 탐색 시간과 커버리지를 조정할 수 있습니다."
)

with st.sidebar:
    st.header("기본 설정")
    target = st.text_input("투자 종목 코드", value="SOXL")
    momentum = st.text_input("모멘텀 종목", value="QQQ")
    bench = st.text_input("벤치마크(선택)", value="SOXX")
    initial_cash = st.number_input("초기 현금", value=10000, step=1000)

    st.divider()
    st.subheader("학습/테스트 기간 설정")

    # Date range setup
    min_date = pd.Timestamp("2015-01-01")
    max_date = pd.Timestamp("2026-12-31")

    # Training period selection with slider
    st.markdown("**학습 기간 (Training)**")
    train_dates = st.slider(
        "학습 날짜 범위",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(pd.Timestamp("2020-01-01").to_pydatetime(), pd.Timestamp("2022-12-31").to_pydatetime()),
        format="YYYY-MM-DD"
    )
    train_ranges = [(str(train_dates[0].date()), str(train_dates[1].date()))]

    # Test period selection with slider
    st.markdown("**테스트 기간 (Test)**")
    test_dates = st.slider(
        "테스트 날짜 범위",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(pd.Timestamp("2023-01-01").to_pydatetime(), pd.Timestamp("2024-12-31").to_pydatetime()),
        format="YYYY-MM-DD"
    )
    test_range = (str(test_dates[0].date()), str(test_dates[1].date()))

    st.divider()
    st.subheader("모드 전환 전략")
    mode_switch_strategy = st.radio(
        "모드 전환 방식",
        options=["RSI", "Golden Cross"],
        index=0,
        help="RSI: 기존 RSI 기반 모드 전환 | Golden Cross: 이동평균 교차 기반 모드 전환"
    )

    # MA period optimization (only for Golden Cross)
    optimize_ma_periods = False
    ma_short_min = ma_short_max = ma_long_min = ma_long_max = None
    if mode_switch_strategy == "Golden Cross":
        optimize_ma_periods = st.checkbox("MA 기간 최적화", value=False, help="Short/Long MA 기간을 최적화합니다")
        if optimize_ma_periods:
            with st.expander("MA 기간 범위", expanded=True):
                ma_short_min, ma_short_max = st.slider("Short MA (주)", 1, 20, (3, 10), 1)
                ma_long_min, ma_long_max = st.slider("Long MA (주)", 5, 50, (15, 30), 1)
                if ma_short_max >= ma_long_min:
                    st.warning("⚠️ Short MA 최댓값은 Long MA 최솟값보다 작아야 합니다!")

    st.divider()
    st.subheader("샘플링 설정")
    n_samples = st.number_input("샘플 개수", value=100, min_value=10, max_value=1000, step=10,
                                 help="생성할 랜덤 파라미터 조합의 개수")
    score_penalty = st.slider(
        "MDD 패널티 가중치", min_value=0.0, max_value=2.0, value=0.6, step=0.05,
        help="점수 = 평균 CAGR - 패널티 × 평균 MDD"
    )
    enable_netting = st.checkbox("퉁치기(순매수/순매도 상쇄)", value=True)
    rsi_period = st.number_input("RSI 기간(주봉)", value=14, step=1, min_value=2)

    st.divider()
    st.subheader("Defense 모드 범위")
    with st.expander("매수 조건 & TP", expanded=False):
        def_buy_min, def_buy_max = st.slider("매수 조건 (%)", 0.5, 10.0, (2.0, 4.0), 0.1)
        def_tp_min, def_tp_max = st.slider("목표 수익 (%)", 0.05, 3.0, (0.2, 0.5), 0.05)

    with st.expander("보유 기간 & 분할", expanded=False):
        def_hold_min, def_hold_max = st.slider("최대 보유 일수", 5, 90, (20, 40), 1)
        def_slices_min, def_slices_max = st.slider("분할 매수 횟수", 2, 20, (5, 10), 1)

    with st.expander("손절 설정", expanded=False):
        def_sl_min, def_sl_max = st.slider("손절 비율 (%)", 1.0, 50.0, (4.0, 20.0), 1.0)
        def_allow_no_sl = st.checkbox("손절 없음 허용 (30% 확률)", value=True)

    st.divider()
    st.subheader("Offense 모드 범위")
    with st.expander("매수 조건 & TP", expanded=False):
        off_buy_min, off_buy_max = st.slider("매수 조건 (%) ", 1.0, 15.0, (4.0, 7.0), 0.1)
        off_tp_min, off_tp_max = st.slider("목표 수익 (%) ", 0.5, 10.0, (1.5, 4.5), 0.1)

    with st.expander("보유 기간 & 분할 ", expanded=False):
        off_hold_min, off_hold_max = st.slider("최대 보유 일수 ", 2, 60, (5, 20), 1)
        off_slices_min, off_slices_max = st.slider("분할 매수 횟수 ", 2, 20, (4, 10), 1)

    with st.expander("손절 설정 ", expanded=False):
        off_sl_min, off_sl_max = st.slider("손절 비율 (%) ", 2.0, 50.0, (6.0, 25.0), 1.0)
        off_allow_no_sl = st.checkbox("손절 없음 허용 (30% 확률) ", value=True)

    st.divider()
    run = st.button("최적화 실행", type="primary")

if run:
    bench_arg = bench.strip() or None

    # Build parameter ranges from UI inputs
    defense_ranges = ModeParamRanges(
        buy_cond_pct=ParamRange(def_buy_min, def_buy_max, is_int=False),
        tp_pct=ParamRange(def_tp_min, def_tp_max, is_int=False),
        max_hold_days=ParamRange(def_hold_min, def_hold_max, is_int=True),
        slices=ParamRange(def_slices_min, def_slices_max, is_int=True),
        stop_loss_pct=ParamRange(def_sl_min, def_sl_max, is_int=False),
        allow_no_stop_loss=def_allow_no_sl,
    )

    offense_ranges = ModeParamRanges(
        buy_cond_pct=ParamRange(off_buy_min, off_buy_max, is_int=False),
        tp_pct=ParamRange(off_tp_min, off_tp_max, is_int=False),
        max_hold_days=ParamRange(off_hold_min, off_hold_max, is_int=True),
        slices=ParamRange(off_slices_min, off_slices_max, is_int=True),
        stop_loss_pct=ParamRange(off_sl_min, off_sl_max, is_int=False),
        allow_no_stop_loss=off_allow_no_sl,
    )

    capital_ranges = CapitalParamRanges(
    )

    # Build MA period ranges if Golden Cross mode
    ma_period_ranges = None
    if mode_switch_strategy == "Golden Cross" and optimize_ma_periods:
        if ma_short_max >= ma_long_min:
            st.error("❌ Short MA 최댓값은 Long MA 최솟값보다 작아야 합니다!")
            st.stop()
        ma_period_ranges = MAPeriodRanges(
            ma_short_period=ParamRange(ma_short_min, ma_short_max, is_int=True),
            ma_long_period=ParamRange(ma_long_min, ma_long_max, is_int=True),
        )

    # Progress tracking UI elements
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Progress callback function
    def update_progress(current, total, success, failed):
        progress = current / total
        progress_bar.progress(progress)
        status_text.text(f"진행 중: {current}/{total} (성공: {success}, 실패: {failed})")

    config = OptimizerConfig(
        target_ticker=target.strip(),
        momentum_ticker=momentum.strip(),
        benchmark_ticker=bench_arg,
        initial_cash=float(initial_cash),
        train_ranges=train_ranges,  # Custom training date ranges
        test_range=test_range,      # Custom test date range
        score_penalty=float(score_penalty),
        top_n=int(n_samples),  # Show all results
        n_samples=int(n_samples),
        enable_netting=enable_netting,
        rsi_period=int(rsi_period),
        defense_ranges=defense_ranges,
        offense_ranges=offense_ranges,
        capital_ranges=capital_ranges,
        mode_switch_strategy="ma_cross" if mode_switch_strategy == "Golden Cross" else "rsi",
        optimize_ma_periods=optimize_ma_periods,
        ma_period_ranges=ma_period_ranges,
        progress_callback=update_progress,  # Add progress callback
    )

    try:
        results = optimize(config)
        progress_bar.progress(1.0)
        status_text.text(f"완료: {len(results)}개 조합 평가 완료")
    except Exception as exc:  # noqa: BLE001 - UI 오류 안내용
        st.error(f"최적화에 실패했습니다: {exc}")
        st.stop()

    if not results:
        st.warning("평가 가능한 조합이 없습니다. 데이터 범위 또는 티커를 확인하세요.")
    else:
        table_rows = []
        chart_rows = []
        for idx, res in enumerate(results, start=1):
            defense_sl = f"{res.defense.stop_loss_pct:.1f}%" if res.defense.stop_loss_pct is not None else "없음"
            offense_sl = f"{res.offense.stop_loss_pct:.1f}%" if res.offense.stop_loss_pct is not None else "없음"

            row = {
                "순위": idx,
                "모드 전환": res.mode_switch_strategy.upper(),
                "Defense 조건": (
                    f"조건 {res.defense.buy_cond_pct:.1f}% / TP {res.defense.tp_pct:.1f}% / "
                    f"보유 {res.defense.max_hold_days}일 / 분할 {res.defense.slices} / SL {defense_sl}"
                ),
                "Offense 조건": (
                    f"조건 {res.offense.buy_cond_pct:.1f}% / TP {res.offense.tp_pct:.1f}% / "
                    f"보유 {res.offense.max_hold_days}일 / 분할 {res.offense.slices} / SL {offense_sl}"
                ),
                "자금 관리": f"초기자금 {res.capital.initial_cash:,.0f}",
                "점수": round(res.score, 4),
                "Train CAGR(%)": round(res.train_metrics.get("CAGR", 0.0) * 100, 2),
                "Train MDD(%)": round(res.train_metrics.get("Max Drawdown", 0.0) * 100, 2),
                "Test CAGR(%)": round(res.test_metrics.get("CAGR", 0.0) * 100, 2),
                "Test MDD(%)": round(res.test_metrics.get("Max Drawdown", 0.0) * 100, 2),
                "Combined CAGR(%)": round(res.combined_metrics.get("CAGR", 0.0) * 100, 2),
                "Combined MDD(%)": round(res.combined_metrics.get("Max Drawdown", 0.0) * 100, 2),
                "Combined Calmar": round(res.combined_metrics.get("Calmar Ratio", 0.0), 2),
            }

            # Add MA periods if Golden Cross mode
            if res.ma_periods:
                row["MA Periods"] = f"Short {res.ma_periods['ma_short_period']}주, Long {res.ma_periods['ma_long_period']}주"

            table_rows.append(row)
            chart_rows.append(
                {
                    "Phase": "Train",
                    "CAGR": res.train_metrics.get("CAGR", 0.0) * 100,
                    "MDD": abs(res.train_metrics.get("Max Drawdown", 0.0)) * 100,
                    "Rank": idx,
                }
            )
            chart_rows.append(
                {
                    "Phase": "Test",
                    "CAGR": res.test_metrics.get("CAGR", 0.0) * 100,
                    "MDD": abs(res.test_metrics.get("Max Drawdown", 0.0)) * 100,
                    "Rank": idx,
                }
            )

        summary_df = pd.DataFrame(table_rows)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        # CSV download button
        csv_data = summary_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📥 CSV 다운로드",
            data=csv_data,
            file_name=f"optimization_results_{target}_{config.mode_switch_strategy}.csv",
            mime="text/csv",
            width="stretch"
        )

        if chart_rows:
            chart_df = pd.DataFrame(chart_rows)
            scatter = (
                alt.Chart(chart_df)
                .mark_circle(size=80, opacity=0.8)
                .encode(
                    x=alt.X("MDD", title="Max Drawdown (%)", scale=alt.Scale(zero=False)),
                    y=alt.Y("CAGR", title="CAGR (%)", scale=alt.Scale(zero=False)),
                    color=alt.Color("Phase", legend=alt.Legend(title="구간")),
                    tooltip=["Phase", "Rank", alt.Tooltip("CAGR", format=".2f"), alt.Tooltip("MDD", format=".2f")],
                    size=alt.Size("Rank", legend=None, scale=alt.Scale(domain=[1, len(results)], range=[200, 50])),
                )
                .interactive()
            )
            st.altair_chart(scatter, width="stretch")
else:
    st.info("왼쪽 사이드바에서 파라미터를 입력하고 '최적화 실행' 버튼을 눌러주세요.")
