"""Streamlit page for running the Dongpa Optuna optimizer."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dongpa_optuna import (
    OptunaConfig,
    extract_results,
    format_results_df,
    get_history_df,
    narrow_config,
    run_optuna,
)

def _best_score_from_study(study):
    """Compute best score from multi-objective study using stored user attrs."""
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        return 0.0
    return max(t.user_attrs.get("score", float("-inf")) for t in completed)


NAV_LINKS = [
    ("backtest.py", "backtest"),
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


def render_results(results, study, label=""):
    """Render optimization results: history chart, table, pareto, best params."""
    if not results:
        st.warning("평가 가능한 조합이 없습니다.")
        return

    # --- Optimization History Chart ---
    st.subheader(f"최적화 히스토리 {label}")
    history_df = get_history_df(study)
    if not history_df.empty:
        history_chart = (
            alt.Chart(history_df)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("Trial:Q", title="Trial"),
                y=alt.Y("Best Score:Q", title="Best Score", scale=alt.Scale(zero=False)),
                tooltip=["Trial", alt.Tooltip("Score:Q", format=".4f"), alt.Tooltip("Best Score:Q", format=".4f")],
            )
            .interactive()
        )
        scatter = (
            alt.Chart(history_df)
            .mark_circle(size=30, opacity=0.4, color="gray")
            .encode(
                x="Trial:Q",
                y=alt.Y("Score:Q", scale=alt.Scale(zero=False)),
                tooltip=["Trial", alt.Tooltip("Score:Q", format=".4f")],
            )
        )
        st.altair_chart(scatter + history_chart, width="stretch")

    # --- Results Table ---
    st.subheader(f"상위 {len(results)}개 결과 {label}")
    summary_df = format_results_df(results)
    st.dataframe(summary_df, hide_index=True, width="stretch")

    # CSV download
    csv_data = summary_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        f"CSV 다운로드 {label}",
        data=csv_data,
        file_name=f"optuna_results{label.replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"csv_{label}",
    )

    # --- CAGR vs MDD Scatter (Pareto Chart) ---
    st.subheader(f"CAGR vs MDD (Pareto) {label}")
    chart_rows = []
    for idx, res in enumerate(results, start=1):
        chart_rows.append({
            "Phase": "Train",
            "CAGR": res.train_metrics.get("CAGR", 0.0) * 100,
            "MDD": abs(res.train_metrics.get("Max Drawdown", 0.0)) * 100,
            "Rank": idx,
        })
        chart_rows.append({
            "Phase": "Test",
            "CAGR": res.test_metrics.get("CAGR", 0.0) * 100,
            "MDD": abs(res.test_metrics.get("Max Drawdown", 0.0)) * 100,
            "Rank": idx,
        })

    chart_df = pd.DataFrame(chart_rows)
    scatter_chart = (
        alt.Chart(chart_df)
        .mark_circle(size=80, opacity=0.8)
        .encode(
            x=alt.X("MDD:Q", title="Max Drawdown (%)", scale=alt.Scale(zero=False)),
            y=alt.Y("CAGR:Q", title="CAGR (%)", scale=alt.Scale(zero=False)),
            color=alt.Color("Phase:N", legend=alt.Legend(title="구간")),
            tooltip=[
                "Phase", "Rank",
                alt.Tooltip("CAGR:Q", format=".2f"),
                alt.Tooltip("MDD:Q", format=".2f"),
            ],
            size=alt.Size("Rank:Q", legend=None, scale=alt.Scale(domain=[1, len(results)], range=[200, 50])),
        )
        .interactive()
    )
    st.altair_chart(scatter_chart, width="stretch")

    # --- Best Parameters Detail ---
    st.subheader(f"최적 파라미터 상세 {label}")
    best = results[0]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Defense 모드**")
        st.write(f"- 매수 조건: {best.defense.buy_cond_pct:.2f}%")
        st.write(f"- 목표 수익: {best.defense.tp_pct:.2f}%")
        st.write(f"- 최대 보유: {best.defense.max_hold_days}일")
        st.write(f"- 분할 매수: {best.defense.slices}회")
        sl_str = f"{best.defense.stop_loss_pct:.1f}%" if best.defense.stop_loss_pct else "없음"
        st.write(f"- 손절: {sl_str}")
    with col2:
        st.markdown("**Offense 모드**")
        st.write(f"- 매수 조건: {best.offense.buy_cond_pct:.2f}%")
        st.write(f"- 목표 수익: {best.offense.tp_pct:.2f}%")
        st.write(f"- 최대 보유: {best.offense.max_hold_days}일")
        st.write(f"- 분할 매수: {best.offense.slices}회")
        sl_str = f"{best.offense.stop_loss_pct:.1f}%" if best.offense.stop_loss_pct else "없음"
        st.write(f"- 손절: {sl_str}")
    with col3:
        st.markdown("**자금 관리**")
        st.write(f"- 초기자금: {best.capital.initial_cash:,.0f}")
        st.markdown("**성과**")
        st.write(f"- Score: {best.score:.4f}")
        st.write(f"- Combined CAGR: {best.combined_metrics.get('CAGR', 0) * 100:.2f}%")
        st.write(f"- Combined MDD: {best.combined_metrics.get('Max Drawdown', 0) * 100:.2f}%")

    if best.rsi_thresholds:
        st.markdown("---")
        st.markdown("**RSI 임계값**")
        t = best.rsi_thresholds
        st.write(f"- High: {t['rsi_high_threshold']:.1f}")
        st.write(f"- Mid-High: {t['rsi_mid_high']:.1f}")
        st.write(f"- Neutral: {t['rsi_neutral']:.1f}")
        st.write(f"- Mid-Low: {t['rsi_mid_low']:.1f}")
        st.write(f"- Low: {t['rsi_low_threshold']:.1f}")

    return best


def _result_to_config_dict(res) -> dict:
    """Convert an OptResult to a config dictionary."""
    config = {
        "defense_slices": res.defense.slices,
        "defense_buy": round(res.defense.buy_cond_pct, 2),
        "defense_tp": round(res.defense.tp_pct, 2),
        "defense_sl": round(res.defense.stop_loss_pct, 1) if res.defense.stop_loss_pct else 0.0,
        "defense_hold": res.defense.max_hold_days,
        "offense_slices": res.offense.slices,
        "offense_buy": round(res.offense.buy_cond_pct, 2),
        "offense_tp": round(res.offense.tp_pct, 2),
        "offense_sl": round(res.offense.stop_loss_pct, 1) if res.offense.stop_loss_pct else 0.0,
        "offense_hold": res.offense.max_hold_days,
        "mode_switch_strategy_index": 0 if res.mode_switch_strategy == "rsi" else 1,
        "cash_limited_buy": res.cash_limited_buy,
    }
    if res.rsi_thresholds:
        config.update({
            "rsi_high_threshold": round(res.rsi_thresholds["rsi_high_threshold"], 1),
            "rsi_mid_high": round(res.rsi_thresholds["rsi_mid_high"], 1),
            "rsi_neutral": round(res.rsi_thresholds["rsi_neutral"], 1),
            "rsi_mid_low": round(res.rsi_thresholds["rsi_mid_low"], 1),
            "rsi_low_threshold": round(res.rsi_thresholds["rsi_low_threshold"], 1),
        })
    if res.ma_periods:
        config.update({
            "ma_short": res.ma_periods.get("ma_short_period", 3),
            "ma_long": res.ma_periods.get("ma_long_period", 7),
        })
    return config


def apply_to_config(res, config_path="config/order_book_settings.json"):
    """Apply best result to order_book_settings.json."""
    path = Path(config_path)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    else:
        current = {}
    current.update(_result_to_config_dict(res))
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_result_as_config(res, filename: str) -> Path:
    """Save a single OptResult as a named config JSON file."""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    if not filename.endswith(".json"):
        filename += ".json"
    path = config_dir / filename
    config = _result_to_config_dict(res)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# ======================== Page Config ========================

st.set_page_config(page_title="Optuna Optimizer", layout="wide")

render_navigation()

st.title("동파 파라미터 최적화 (Optuna TPE)")
st.caption(
    "Optuna의 TPE(Tree-structured Parzen Estimator) 알고리즘으로 파라미터를 탐색합니다. "
    "2-Phase 모드는 넓은 탐색 후 수렴 영역을 자동으로 좁혀 정밀 탐색합니다."
)

# ======================== Sidebar ========================

with st.sidebar:
    st.header("기본 설정")
    target = st.text_input("투자 종목 코드", value="SOXL", key="optuna_target")
    momentum = st.text_input("모멘텀 종목", value="QQQ", key="optuna_momentum")
    bench = st.text_input("벤치마크(선택)", value="SOXX", key="optuna_bench")
    initial_cash = st.number_input("초기 현금", value=10000, step=1000, key="optuna_cash")

    st.divider()
    st.subheader("학습/테스트 기간 설정")

    min_date = pd.Timestamp("2015-01-01")
    max_date = pd.Timestamp("2026-12-31")

    st.markdown("**학습 기간 (Training)**")
    train_dates = st.slider(
        "학습 날짜 범위",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(pd.Timestamp("2020-01-01").to_pydatetime(), pd.Timestamp("2022-12-31").to_pydatetime()),
        format="YYYY-MM-DD",
        key="optuna_train_dates",
    )
    train_ranges = [(str(train_dates[0].date()), str(train_dates[1].date()))]

    st.markdown("**테스트 기간 (Test)**")
    test_dates = st.slider(
        "테스트 날짜 범위",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(pd.Timestamp("2023-01-01").to_pydatetime(), pd.Timestamp("2024-12-31").to_pydatetime()),
        format="YYYY-MM-DD",
        key="optuna_test_dates",
    )
    test_range = (str(test_dates[0].date()), str(test_dates[1].date()))

    st.divider()
    st.subheader("모드 전환 전략")
    mode_switch_strategy = st.radio(
        "모드 전환 방식",
        options=["RSI", "Golden Cross", "Both (탐색)"],
        index=0,
        help="RSI: RSI 기반 | Golden Cross: MA 교차 기반 | Both: 둘 다 탐색",
        key="optuna_mode_strategy",
    )

    optimize_rsi_thresholds = False
    optimize_ma_periods = False

    if mode_switch_strategy == "RSI":
        optimize_rsi_thresholds = st.checkbox("RSI 임계값 최적화", value=False, key="optuna_opt_rsi")
    elif mode_switch_strategy == "Golden Cross":
        optimize_ma_periods = st.checkbox("MA 기간 최적화", value=False, key="optuna_opt_ma")
    elif mode_switch_strategy == "Both (탐색)":
        optimize_rsi_thresholds = st.checkbox("RSI 임계값 최적화", value=False, key="optuna_opt_rsi_both")
        optimize_ma_periods = st.checkbox("MA 기간 최적화", value=False, key="optuna_opt_ma_both")

    optimize_cash_limited_buy = st.checkbox(
        "현금 한도 매수 최적화",
        value=False,
        help="트렌치 예산 > 현금일 때 현금 한도 내 매수 vs 스킵을 Optuna가 탐색합니다.",
        key="optuna_opt_cash_limited",
    )

    st.divider()
    st.subheader("Optuna 설정")

    phase_mode = st.radio(
        "탐색 모드",
        options=["1-Phase (단일)", "2-Phase (넓은 탐색 + 집중)"],
        index=1,
        help="1-Phase: 설정한 범위로 한 번 탐색 | 2-Phase: 넓게 탐색 후 수렴 영역 자동 집중",
        key="optuna_phase_mode",
    )
    use_two_phase = phase_mode.startswith("2")

    n_trials = st.number_input(
        "Phase 1 Trial 수" if use_two_phase else "Trial 수",
        value=300, min_value=10, max_value=5000, step=50,
        help="Phase 1 탐색 횟수 (2-Phase시 Phase 2는 x3)" if use_two_phase else "탐색 횟수",
        key="optuna_n_trials",
    )
    if use_two_phase:
        phase2_multiplier = st.slider(
            "Phase 2 배수", min_value=1, max_value=5, value=3, step=1,
            help="Phase 2 trials = Phase 1 trials x 배수",
            key="optuna_p2_mult",
        )

    score_penalty = st.slider(
        "MDD 패널티 가중치 (결과 정렬용)", min_value=0.0, max_value=2.0, value=0.6, step=0.05,
        help="Pareto 최적화 후 결과 정렬: 점수 = 평균 CAGR - 패널티 × 평균 |MDD|",
        key="optuna_penalty",
    )
    top_n = st.number_input(
        "상위 결과 수", value=20, min_value=5, max_value=100, step=5,
        key="optuna_top_n",
    )
    enable_netting = st.checkbox("퉁치기(순매수/순매도 상쇄)", value=True, key="optuna_netting")
    rsi_period = st.number_input("RSI 기간(주봉)", value=14, step=1, min_value=2, key="optuna_rsi_period")

    st.divider()
    st.subheader("제약 조건")
    use_constraints = st.checkbox(
        "제약 조건 사용",
        value=False,
        help="특정 기간에서 MDD/CAGR 조건을 만족하지 못하면 탈락",
        key="optuna_use_constraints",
    )
    constraints_list = []
    if use_constraints:
        n_constraints = st.number_input("제약 조건 수", value=1, min_value=1, max_value=3, step=1, key="n_constraints")
        for ci in range(int(n_constraints)):
            with st.expander(f"제약 조건 {ci + 1}", expanded=True):
                c_dates = st.slider(
                    f"기간 {ci + 1}",
                    min_value=min_date.to_pydatetime(),
                    max_value=max_date.to_pydatetime(),
                    value=(pd.Timestamp("2022-01-01").to_pydatetime(), pd.Timestamp("2022-12-31").to_pydatetime()),
                    format="YYYY-MM-DD",
                    key=f"c_dates_{ci}",
                )
                c_range = (str(c_dates[0].date()), str(c_dates[1].date()))
                c_use_mdd = st.checkbox(f"MDD 제한", value=True, key=f"c_use_mdd_{ci}")
                c_max_mdd = None
                if c_use_mdd:
                    c_max_mdd_pct = st.slider(
                        f"최대 MDD (%)", -80.0, 0.0, -30.0, 1.0,
                        help="이 값보다 MDD가 낮으면 탈락 (예: -30% = MDD가 -30%보다 나쁘면 탈락)",
                        key=f"c_mdd_{ci}",
                    )
                    c_max_mdd = c_max_mdd_pct / 100.0
                c_use_cagr = st.checkbox(f"최소 수익률 제한", value=True, key=f"c_use_cagr_{ci}")
                c_min_cagr = None
                if c_use_cagr:
                    c_min_cagr_pct = st.slider(
                        f"최소 CAGR (%)", -50.0, 50.0, 0.0, 1.0,
                        help="이 값보다 CAGR이 낮으면 탈락 (0% = 손실 나면 탈락)",
                        key=f"c_cagr_{ci}",
                    )
                    c_min_cagr = c_min_cagr_pct / 100.0
                constraints_list.append((c_range, c_max_mdd, c_min_cagr))

    st.divider()
    st.subheader("Defense 모드 범위")
    with st.expander("매수 조건 & TP", expanded=False):
        def_buy_min, def_buy_max = st.slider("매수 조건 (%)", 0.5, 10.0, (2.0, 4.0), 0.1, key="od_buy")
        def_tp_min, def_tp_max = st.slider("목표 수익 (%)", 0.05, 3.0, (0.2, 0.5), 0.05, key="od_tp")

    with st.expander("보유 기간 & 분할", expanded=False):
        def_hold_min, def_hold_max = st.slider("최대 보유 일수", 5, 90, (20, 40), 1, key="od_hold")
        def_slices_min, def_slices_max = st.slider("분할 매수 횟수", 1, 20, (5, 10), 1, key="od_slices")

    with st.expander("손절 설정", expanded=False):
        def_sl_min, def_sl_max = st.slider("손절 비율 (%)", 1.0, 50.0, (4.0, 20.0), 1.0, key="od_sl")

    st.divider()
    st.subheader("Offense 모드 범위")
    with st.expander("매수 조건 & TP", expanded=False):
        off_buy_min, off_buy_max = st.slider("매수 조건 (%) ", 1.0, 15.0, (4.0, 7.0), 0.1, key="oo_buy")
        off_tp_min, off_tp_max = st.slider("목표 수익 (%) ", 0.5, 10.0, (1.5, 4.5), 0.1, key="oo_tp")

    with st.expander("보유 기간 & 분할 ", expanded=False):
        off_hold_min, off_hold_max = st.slider("최대 보유 일수 ", 2, 60, (5, 20), 1, key="oo_hold")
        off_slices_min, off_slices_max = st.slider("분할 매수 횟수 ", 1, 20, (4, 10), 1, key="oo_slices")

    with st.expander("손절 설정 ", expanded=False):
        off_sl_min, off_sl_max = st.slider("손절 비율 (%) ", 2.0, 50.0, (6.0, 25.0), 1.0, key="oo_sl")

    st.divider()
    run = st.button("Optuna 최적화 실행", type="primary", key="optuna_run")

# ======================== Main Area ========================

if run:
    bench_arg = bench.strip() or None

    if mode_switch_strategy == "RSI":
        mode_val = "rsi"
    elif mode_switch_strategy == "Golden Cross":
        mode_val = "ma_cross"
    else:
        mode_val = "both"

    config = OptunaConfig(
        target_ticker=target.strip(),
        momentum_ticker=momentum.strip(),
        benchmark_ticker=bench_arg,
        initial_cash=float(initial_cash),
        train_ranges=train_ranges,
        test_range=test_range,
        rsi_period=int(rsi_period),
        enable_netting=enable_netting,
        score_penalty=float(score_penalty),
        n_trials=int(n_trials),
        top_n=int(top_n),
        mode_switch_strategy=mode_val,
        optimize_rsi_thresholds=optimize_rsi_thresholds,
        optimize_ma_periods=optimize_ma_periods,
        optimize_cash_limited_buy=optimize_cash_limited_buy,
        def_buy_range=(def_buy_min, def_buy_max),
        def_tp_range=(def_tp_min, def_tp_max),
        def_hold_range=(int(def_hold_min), int(def_hold_max)),
        def_slices_range=(int(def_slices_min), int(def_slices_max)),
        def_sl_range=(def_sl_min, def_sl_max),
        off_buy_range=(off_buy_min, off_buy_max),
        off_tp_range=(off_tp_min, off_tp_max),
        off_hold_range=(int(off_hold_min), int(off_hold_max)),
        off_slices_range=(int(off_slices_min), int(off_slices_max)),
        off_sl_range=(off_sl_min, off_sl_max),
        constraints=constraints_list if constraints_list else None,
    )

    # --- Progress UI ---
    progress_bar = st.progress(0)
    status_text = st.empty()

    if use_two_phase:
        total_trials = int(n_trials) + int(n_trials) * phase2_multiplier
    else:
        total_trials = int(n_trials)

    def update_progress(current, phase_total, best_score, *, phase_offset=0):
        progress = min((phase_offset + current) / total_trials, 1.0)
        progress_bar.progress(progress)
        best_str = f"{best_score:.4f}" if best_score is not None else "N/A"
        phase_label = ""
        if use_two_phase:
            phase_label = "Phase 1 | " if phase_offset == 0 else "Phase 2 | "
        status_text.text(f"{phase_label}Trial {current}/{phase_total} | Best Score: {best_str}")

    # Phase 1 progress
    def p1_progress(current, phase_total, best_score):
        update_progress(current, phase_total, best_score, phase_offset=0)

    config.progress_callback = p1_progress

    try:
        # ── Phase 1 ──
        phase_label = "Phase 1 (Wide)" if use_two_phase else ""
        with st.spinner(f"{'Phase 1: 넓은 탐색' if use_two_phase else 'Optuna 최적화'} 실행 중..."):
            study1 = run_optuna(config)
        results1 = extract_results(study1, config)

        completed1 = sum(1 for t in study1.trials if t.state.name == "COMPLETE")
        pruned1 = sum(1 for t in study1.trials if t.state.name == "PRUNED")

        if use_two_phase and results1:
            # Show Phase 1 summary briefly
            st.info(
                f"Phase 1 완료: {completed1}개 완료, {pruned1}개 pruned, "
                f"Best Score: {_best_score_from_study(study1):.4f} → Phase 2 집중 탐색 시작..."
            )

            # ── Phase 2 ──
            p2_trials = int(n_trials) * phase2_multiplier
            config2 = narrow_config(config, results1, phase2_trials=p2_trials)

            def p2_progress(current, phase_total, best_score):
                update_progress(current, phase_total, best_score, phase_offset=int(n_trials))

            config2.progress_callback = p2_progress

            with st.spinner("Phase 2: 수렴 영역 집중 탐색 중..."):
                study2 = run_optuna(config2)
            results2 = extract_results(study2, config2)

            completed2 = sum(1 for t in study2.trials if t.state.name == "COMPLETE")
            pruned2 = sum(1 for t in study2.trials if t.state.name == "PRUNED")

            progress_bar.progress(1.0)
            improvement = ""
            if results1 and results2:
                diff = ((results2[0].score - results1[0].score) / abs(results1[0].score)) * 100
                improvement = f" | Score 변화: {diff:+.1f}%"
            status_msg = f"완료: Phase 1 ({completed1}개) + Phase 2 ({completed2}개){improvement}"
            status_text.text(status_msg)

            final_results = results2 if results2 else results1
            st.session_state["optuna_results1"] = results1
            st.session_state["optuna_study1"] = study1
            st.session_state["optuna_results2"] = results2
            st.session_state["optuna_study2"] = study2
            st.session_state["optuna_two_phase"] = True
            st.session_state["optuna_status"] = status_msg
        else:
            # Single phase
            progress_bar.progress(1.0)
            status_msg = (
                f"완료: {completed1}개 완료, {pruned1}개 pruned, "
                f"Best Score: {_best_score_from_study(study1):.4f}"
            )
            status_text.text(status_msg)
            final_results = results1
            st.session_state["optuna_results1"] = results1
            st.session_state["optuna_study1"] = study1
            st.session_state["optuna_results2"] = None
            st.session_state["optuna_study2"] = None
            st.session_state["optuna_two_phase"] = False
            st.session_state["optuna_status"] = status_msg

        # Store results in session_state so they persist across re-renders
        st.session_state["optuna_results"] = final_results

    except Exception as exc:
        st.error(f"최적화에 실패했습니다: {exc}")
        st.stop()

# --- Render results & save (outside `if run:` so they persist across reruns) ---
if "optuna_results" in st.session_state and st.session_state["optuna_results"]:
    final_results = st.session_state["optuna_results"]

    if st.session_state.get("optuna_status"):
        st.success(st.session_state["optuna_status"])

    if st.session_state.get("optuna_two_phase") and st.session_state.get("optuna_results2"):
        tab1, tab2 = st.tabs(["Phase 2 (최종)", "Phase 1 (탐색)"])
        with tab1:
            render_results(st.session_state["optuna_results2"], st.session_state["optuna_study2"], label="(Phase 2)")
        with tab2:
            render_results(st.session_state["optuna_results1"], st.session_state["optuna_study1"], label="(Phase 1)")
    elif st.session_state.get("optuna_results1") and st.session_state.get("optuna_study1"):
        render_results(st.session_state["optuna_results1"], st.session_state["optuna_study1"])

    st.divider()
    st.subheader("결과 저장")

    col_rank, col_name = st.columns([1, 2])
    with col_rank:
        save_rank = st.number_input(
            "순위 선택",
            min_value=1,
            max_value=len(final_results),
            value=1,
            step=1,
            key="save_rank",
            help="저장할 결과의 순위를 선택하세요",
        )
    with col_name:
        save_filename = st.text_input(
            "파일 이름",
            placeholder="예: optuna_best_1",
            key="save_config_name",
            help="config/ 폴더에 JSON 파일로 저장됩니다 (backtest 페이지에서 불러올 수 있음)",
        )

    selected_res = final_results[save_rank - 1]
    st.caption(
        f"#{save_rank} — Score: {selected_res.score:.4f} | "
        f"CAGR: {selected_res.combined_metrics.get('CAGR', 0) * 100:.2f}% | "
        f"MDD: {selected_res.combined_metrics.get('Max Drawdown', 0) * 100:.2f}%"
    )

    col_save, col_apply = st.columns(2)
    with col_save:
        if st.button("💾 JSON으로 저장", key="save_config", type="primary"):
            name = save_filename.strip() if save_filename else ""
            if not name:
                ts = datetime.now().strftime("%m%d_%H%M")
                name = f"optuna_rank_{save_rank}_{ts}"
            if name.lower() in ["order_book_settings", "default"]:
                st.error("예약된 이름입니다. 다른 이름을 사용해주세요!")
            else:
                saved_path = save_result_as_config(selected_res, name)
                st.success(f"'{saved_path.name}'에 저장 완료! backtest 페이지에서 불러올 수 있습니다.")
    with col_apply:
        if st.button("⚡ order_book_settings에 적용", key="apply_config"):
            apply_to_config(selected_res)
            st.success(f"#{save_rank} 결과를 order_book_settings.json에 적용 완료!")
elif not run:
    st.info("왼쪽 사이드바에서 파라미터를 설정하고 'Optuna 최적화 실행' 버튼을 눌러주세요.")
