"""Streamlit page for running the Dongpa parameter optimizer."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dongpa_optimizer import OptimizerConfig, optimize

st.title("동파 파라미터 최적화")
st.caption(
    "2022년과 2025년 데이터를 학습 구간으로, 2023~2024년을 테스트 구간으로 사용합니다. "
    "각 조합은 학습·테스트에서의 CAGR과 MDD를 계산하고, 평균 CAGR에서 평균 MDD×가중치(기본 0.6)를 뺀 점수로 순위를 매깁니다."
)

st.write(
    "최적화는 Yahoo Finance 데이터를 새로 다운로드합니다. 네트워크 환경에 따라 시간이 소요될 수 있으며, "
    "완료 시 `strategy_performance.md` 파일이 갱신됩니다. 기본 그리드는 약 320개의 조합을 평가합니다."
)

with st.sidebar:
    st.header("최적화 설정")
    target = st.text_input("투자 종목 코드", value="SOXL")
    momentum = st.text_input("모멘텀 종목", value="QQQ")
    bench = st.text_input("벤치마크(선택)", value="SOXX")
    initial_cash = st.number_input("초기 현금", value=10000, step=1000)
    score_penalty = st.slider(
        "MDD 패널티 가중치", min_value=0.0, max_value=2.0, value=0.6, step=0.05,
        help="점수 = 평균 CAGR - 패널티 × 평균 MDD"
    )
    top_n = st.slider("상위 조합 수", min_value=1, max_value=320, value=50)
    enable_netting = st.checkbox("퉁치기(순매수/순매도 상쇄)", value=True)
    rsi_period = st.number_input("RSI 기간(주봉)", value=14, step=1, min_value=2)
    run = st.button("최적화 실행")

if run:
    bench_arg = bench.strip() or None
    config = OptimizerConfig(
        target_ticker=target.strip(),
        momentum_ticker=momentum.strip(),
        benchmark_ticker=bench_arg,
        initial_cash=float(initial_cash),
        score_penalty=float(score_penalty),
        top_n=int(top_n),
        enable_netting=enable_netting,
        rsi_period=int(rsi_period),
    )

    with st.spinner("최적화 실행 중..."):
        try:
            results, md_path = optimize(config)
        except Exception as exc:  # noqa: BLE001 - UI 오류 안내용
            st.error(f"최적화에 실패했습니다: {exc}")
            st.stop()

    if not results:
        st.warning("평가 가능한 조합이 없습니다. 데이터 범위 또는 티커를 확인하세요.")
    else:
        st.success("최적화가 완료되었습니다.")

        table_rows = []
        chart_rows = []
        for idx, res in enumerate(results[: config.top_n], start=1):
            defense_sl = f"{res.defense.stop_loss_pct:.1f}%" if res.defense.stop_loss_pct is not None else "없음"
            offense_sl = f"{res.offense.stop_loss_pct:.1f}%" if res.offense.stop_loss_pct is not None else "없음"
            table_rows.append(
                {
                    "순위": idx,
                    "Defense 조건": (
                        f"조건 {res.defense.buy_cond_pct:.1f}% / TP {res.defense.tp_pct:.1f}% / "
                        f"보유 {res.defense.max_hold_days}일 / 분할 {res.defense.slices} / SL {defense_sl}"
                    ),
                    "Offense 조건": (
                        f"조건 {res.offense.buy_cond_pct:.1f}% / TP {res.offense.tp_pct:.1f}% / "
                        f"보유 {res.offense.max_hold_days}일 / 분할 {res.offense.slices} / SL {offense_sl}"
                    ),
                    "자금 관리": f"주기 {res.capital.refresh_cycle_days}일 / PCR {res.capital.profit_compound_rate:.2f} / LCR {res.capital.loss_compound_rate:.2f}",
                    "점수": round(res.score, 4),
                    "Train CAGR(%)": round(res.train_metrics.get("CAGR", 0.0) * 100, 2),
                    "Train MDD(%)": round(res.train_metrics.get("Max Drawdown", 0.0) * 100, 2),
                    "Test CAGR(%)": round(res.test_metrics.get("CAGR", 0.0) * 100, 2),
                    "Test MDD(%)": round(res.test_metrics.get("Max Drawdown", 0.0) * 100, 2),
                }
            )
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
        st.dataframe(summary_df, use_container_width=True)

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
                    size=alt.Size("Rank", legend=None, scale=alt.Scale(domain=[1, config.top_n], range=[200, 50])),
                )
                .interactive()
            )
            st.altair_chart(scatter, use_container_width=True)

        if isinstance(md_path, Path) and md_path.exists():
            md_text = md_path.read_text(encoding="utf-8")
            st.download_button(
                "마크다운 다운로드", data=md_text, file_name=md_path.name, mime="text/markdown"
            )
            with st.expander("생성된 마크다운 미리보기"):
                st.markdown(md_text)
        else:
            st.info("작성된 마크다운 파일을 찾을 수 없습니다.")
else:
    st.info("왼쪽 사이드바에서 파라미터를 입력하고 '최적화 실행' 버튼을 눌러주세요.")
