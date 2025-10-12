
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date

from dongpa_engine import (ModeParams, CapitalParams, StrategyParams, DongpaBacktester, summarize)

st.set_page_config(page_title="동파법 백테스트 (LOC, 일일 N등분)", layout="wide")

today = date.today()
year_start = date(today.year, 1, 1)

st.title("동파법 백테스트 (LOC 전용, 일일 N등분 매수)")
st.caption("주문은 LOC 기준 · 레일 없음 · 하루 최대 1회 매수(트렌치 예산 한도, 정수 주식만). 결과 지표는 Equity 등 영문 용어 사용.")

with st.sidebar:
    st.header("기본 설정")
    colA, colB = st.columns(2)
    target = colA.text_input("투자 종목 코드", value="SOXL")
    momentum = colB.text_input("모멘텀 종목(주봉 RSI 계산)", value="QQQ")
    bench = st.text_input("벤치마크(선택)", value="SOXX")
    start = st.date_input("시작일", value=year_start)
    end = st.date_input("종료일", value=today)

    st.header("투자금 갱신 (복리)")
    pcr = st.number_input("이익복리율 PCR (%)", value=80, step=1) / 100.0
    lcr = st.number_input("손실복리율 LCR (%)", value=30, step=1) / 100.0
    cyc = st.number_input("투자금 갱신 주기(거래일)", value=10, step=1)
    init_cash = st.number_input("초기 가용현금", value=10000, step=1000)

    st.header("안전 모드")
    s1 = st.number_input("분할수(N) - 안전", value=7, step=1)
    cond1 = st.number_input("매수조건(%) - 안전", value=3.0, step=0.1, format="%.2f")
    tp1 = st.number_input("익절(%) - 안전", value=0.2, step=0.1, format="%.2f")
    hold1 = st.number_input("최대 보유일(거래일) - 안전", value=30, step=1)

    st.header("공세 모드")
    s2 = st.number_input("분할수(N) - 공세", value=7, step=1)
    cond2 = st.number_input("매수조건(%) - 공세", value=5.0, step=0.1, format="%.2f")
    tp2 = st.number_input("익절(%) - 공세", value=2.5, step=0.1, format="%.2f")
    hold2 = st.number_input("최대 보유일(거래일) - 공세", value=7, step=1)

run = st.button("백테스트 실행")

if run:
    st.info("데이터 로딩 중...")
    df_t = yf.download(target, start=start, end=end, progress=False)
    df_m = yf.download(momentum, start=start, end=end, progress=False)

    if df_t.empty or df_m.empty:
        st.error("데이터가 비어 있습니다. 티커/기간을 확인하세요.")
    else:
        defense = ModeParams(buy_cond_pct=cond1, tp_pct=tp1, max_hold_days=int(hold1), slices=int(s1))
        offense = ModeParams(buy_cond_pct=cond2, tp_pct=tp2, max_hold_days=int(hold2), slices=int(s2))

        cap = CapitalParams(
            initial_cash=float(init_cash),
            refresh_cycle_days=int(cyc),
            profit_compound_rate=float(pcr),
            loss_compound_rate=float(lcr),
            slippage_pct=0.0,
        )

        params = StrategyParams(
            target_ticker=target,
            momentum_ticker=momentum,
            benchmark_ticker=bench if bench.strip() else None,
            rsi_period=14,
            reset_on_mode_change=True,
            defense=defense,
            offense=offense,
        )

        bt = DongpaBacktester(df_t, df_m, params, cap)
        res = bt.run()
        eq = res['equity']
        journal = res['journal']

        st.success("완료!")

        summary_metrics = summarize(eq)

        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader("Equity Curve")
            st.line_chart(eq)

        with col2:
            st.subheader("요약 지표")
            c1, c2, c3 = st.columns(3)
            c1.metric("Final Equity", f"{summary_metrics['Final Equity']:,.0f}")
            c2.metric("CAGR", f"{summary_metrics['CAGR']:.2%}")
            c3.metric("Sharpe (rf=0)", f"{summary_metrics['Sharpe (rf=0)']:.2f}")

            c4, c5 = st.columns(2)
            c4.metric("Volatility (ann)", f"{summary_metrics['Volatility (ann)']:.2%}")
            c5.metric("Max Drawdown", f"{summary_metrics['Max Drawdown']:.2%}")

        st.subheader("거래 일지 (일자별 요약, 정수 주식 기준)")
        st.dataframe(journal, use_container_width=True, height=420)

        st.download_button("거래일지 CSV 다운로드", data=journal.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f"dongpa_journal_{target}.csv", mime="text/csv")
        st.download_button("Equity CSV 다운로드", data=eq.to_csv().encode('utf-8'),
                           file_name=f"equity_{target}.csv", mime="text/csv")

        st.caption("거래일지 컬럼은 한국어입니다. (Equity 등 성과 지표는 영문 표기)")
