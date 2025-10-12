
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
from pathlib import Path

from dongpa_engine import (ModeParams, CapitalParams, StrategyParams, DongpaBacktester, summarize)


def compute_trade_metrics(trade_log: pd.DataFrame | None, initial_cash: float) -> dict[str, float | int | None] | None:
    if trade_log is None or trade_log.empty:
        return None

    closed = trade_log[trade_log["상태"] == "청산"].copy()
    if closed.empty:
        return {
            "period_return_pct": None,
            "trade_count": 0,
            "moc_count": 0,
            "net_profit": 0.0,
            "avg_hold_days": None,
            "avg_return_pct": None,
            "avg_gain": None,
            "avg_loss": None,
        }

    for col in ("실현손익", "보유기간(일)", "수익률(%)"):
        if col in closed.columns:
            closed[col] = pd.to_numeric(closed[col], errors="coerce")

    closed = closed.dropna(subset=["실현손익"])
    if closed.empty:
        return {
            "period_return_pct": None,
            "trade_count": 0,
            "moc_count": 0,
            "net_profit": 0.0,
            "avg_hold_days": None,
            "avg_return_pct": None,
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
    avg_gain = float(gain_series.mean()) if not gain_series.empty else None
    avg_loss = float(loss_series.mean()) if not loss_series.empty else None
    period_return_pct = None
    if initial_cash > 0:
        period_return_pct = (net_profit / initial_cash) * 100.0

    return {
        "period_return_pct": period_return_pct,
        "trade_count": trade_count,
        "moc_count": moc_count,
        "net_profit": net_profit,
        "avg_hold_days": avg_hold,
        "avg_return_pct": avg_return_pct,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
    }

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
        trade_log = res.get('trade_log')
        trade_metrics = compute_trade_metrics(trade_log, float(init_cash))

        st.success("완료! 가격 데이터는 outputs/ 아래 CSV로 저장되었습니다.")

        summary_metrics = summarize(eq)

        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader("Equity Curve")
            st.line_chart(eq)

        with col2:
            st.subheader("요약 지표")
            c1, c2, c3 = st.columns(3)
            c1.metric("Final Equity", f"${summary_metrics['Final Equity']:,.0f}")
            c2.metric("CAGR", f"{summary_metrics['CAGR']:.2%}")
            c3.metric("Sharpe (rf=0)", f"{summary_metrics['Sharpe (rf=0)']:.2f}")

            c4, c5 = st.columns(2)
            c4.metric("Volatility (ann)", f"{summary_metrics['Volatility (ann)']:.2%}")
            c5.metric("Max Drawdown", f"{summary_metrics['Max Drawdown']:.2%}")

            if trade_metrics is not None:
                st.markdown("**실현 지표**")
                m1, m2, m3 = st.columns(3)
                m1.metric("기간 내 이익률", f"{trade_metrics['period_return_pct']:.2f}%" if trade_metrics['period_return_pct'] is not None else "-")
                m2.metric("거래횟수", f"{trade_metrics['trade_count']:,}")
                m3.metric("MOC 횟수", f"{trade_metrics['moc_count']:,}")

                m4, m5, m6 = st.columns(3)
                m4.metric("이익금", f"${trade_metrics['net_profit']:,.2f}")
                m5.metric("평균 보유일", f"{trade_metrics['avg_hold_days']:.2f}" if trade_metrics['avg_hold_days'] is not None else "-")
                m6.metric("평균 이익률", f"{trade_metrics['avg_return_pct']:.2f}%" if trade_metrics['avg_return_pct'] is not None else "-")

                m7, m8 = st.columns(2)
                m7.metric("평균 실현이익", f"${trade_metrics['avg_gain']:,.2f}" if trade_metrics['avg_gain'] is not None else "-")
                m8.metric("평균 실현손해", f"${trade_metrics['avg_loss']:,.2f}" if trade_metrics['avg_loss'] is not None else "-")

        st.subheader("일일 거래 요약 (장이 열린 모든 날 포함)")
        st.dataframe(journal, use_container_width=True, height=360)

        st.download_button("일일 요약 CSV 다운로드", data=journal.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f"dongpa_daily_{target}.csv", mime="text/csv")

        if trade_log is not None and not trade_log.empty:
            st.subheader("트랜치별 매수·매도 기록")
            st.dataframe(trade_log, use_container_width=True, height=360)
            st.download_button("트랜치 로그 CSV 다운로드", data=trade_log.to_csv(index=False).encode('utf-8-sig'),
                               file_name=f"dongpa_trades_{target}.csv", mime="text/csv")
        st.download_button("Equity CSV 다운로드", data=eq.to_csv().encode('utf-8'),
                           file_name=f"equity_{target}.csv", mime="text/csv")

        st.caption("일일 요약과 트랜치 로그 모두 한국어 컬럼을 사용합니다. 트랜치 로그의 상태=보유중은 미청산 트랜치입니다. (Equity 등 성과 지표는 영문 표기)")
