import { useEffect, useState } from "react";

import { getTraderStatus } from "../api/client";
import type { BacktestRequest, OrderBookPayload, TraderStatus, ViewerResult } from "../types";
import { AutoTradePanel } from "./AutoTradePanel";
import { DailyLog } from "./DailyLog";
import { DataTable } from "./DataTable";
import { EquityChart } from "./EquityChart";
import { LocLadder } from "./LocLadder";
import { MetricGrid } from "./MetricGrid";
import { OrderPipButton } from "./OrderPip";

export function OrderBookView({
  orderBook,
  result,
  request,
  accountId,
}: {
  orderBook: OrderBookPayload;
  result: ViewerResult;
  request: BacktestRequest;
  accountId?: string;
}) {
  const state = orderBook.state;
  const period = `${result.meta.start_date} – ${result.meta.end_date}`;
  const [trader, setTrader] = useState<{ status: TraderStatus | null; error: string | null }>({ status: null, error: null });

  // Re-read the run records whenever the account or its order book changes.
  useEffect(() => {
    if (!accountId) return;
    let live = true;
    getTraderStatus(accountId)
      .then((status) => live && setTrader({ status, error: null }))
      .catch((reason: Error) => live && setTrader({ status: null, error: reason.message }));
    return () => { live = false; };
  }, [accountId, orderBook]);

  return (
    <div className="result-stack orderbook-stack">
      {accountId ? <AutoTradePanel status={trader.status?.account === accountId ? trader.status : null} error={trader.error} orderBook={orderBook} tradeLog={result.trade_log} ticker={result.meta.target_ticker} /> : null}
      <section className="order-overview-grid">
        <article className="current-mode-card">
          <header><span className="eyebrow">현재 모드 // CURRENT MODE</span><small>{request.strategy.mode_switch_strategy.toUpperCase()} ({request.strategy.momentum_ticker})</small></header>
          <div className="current-mode-title"><strong>{state.current_mode === "offense" ? "공세 모드" : "안전 모드"}</strong><span>{state.current_mode === "offense" ? "AGGRESSIVE" : "SAFE"}</span></div>
          <p>{state.rsi_value == null ? "현재 전략 규칙으로 다음 거래일 모드를 계산했습니다." : <>주봉 RSI <b>{state.rsi_value.toFixed(2)}</b>를 기준으로 현재 모드가 결정되었습니다.</>}</p>
          <div className="rsi-scale"><span>안전 영역</span><i style={{ left: `${Math.max(0, Math.min(100, state.rsi_value ?? 50))}%` }} /><span>공세 영역</span></div>
        </article>
        <section className="snapshot-grid">
          <article><span>최근 종가 // LAST CLOSE</span><strong>${state.prev_close?.toFixed(2) ?? "—"}</strong><small>{state.last_date}</small></article>
          <article><span>잔여 현금 // CASH</span><strong>${state.current_cash.toLocaleString("en-US", { maximumFractionDigits: 0 })}</strong><small>가용 자금</small></article>
          <article><span>보유 수량 // SHARES</span><strong>{state.current_position_qty.toLocaleString()}주</strong><small>{orderBook.holdings.rows.length} 트랜치</small></article>
          <article><span>트랜치 예산 // BUDGET</span><strong>${state.tranche_budget?.toLocaleString("en-US", { maximumFractionDigits: 0 }) ?? "—"}</strong><small>초기현금 ÷ 분할수</small></article>
        </section>
      </section>

      <LocLadder
        orderBook={orderBook}
        footer={orderBook.netting_message || "모의 계산 결과 · 실제 주문은 사용자 책임"}
        action={<OrderPipButton orderBook={orderBook} ticker={result.meta.target_ticker} accountId={accountId} />}
      />

      <DataTable title="보유 포지션" eyebrow="OPEN TRANCHES" table={orderBook.holdings} filename={`dongpa_holdings_${result.meta.target_ticker}.csv`} limit={0} />

      <section className="order-detail-grid">
        <div className="detail-card">
          <header className="panel-header"><span className="eyebrow">퉁치기 상세 // NETTING</span><small>{orderBook.netting_scenarios.rows.length} 구간</small></header>
          {orderBook.netting_scenarios.rows.length ? <DataTable title="종가 구간별 순결과" eyebrow="SCENARIOS" table={orderBook.netting_scenarios} filename={`dongpa_netting_${result.meta.target_ticker}.csv`} limit={0} embedded /> : <div className="compact-empty">상쇄 대상이 없습니다.</div>}
        </div>
      </section>

      {result.equity.length ? (
        <>
          <MetricGrid result={result} period={period} />
          <EquityChart points={result.equity} modeBands={result.mode_bands} logScale={result.meta.log_scale} target={result.meta.target_ticker} />
          <DailyLog table={result.journal} filename={`dongpa_daily_${result.meta.target_ticker}.csv`} />
        </>
      ) : <div className="compact-empty">{result.meta.start_date} 시작 // 아직 거래일이 없어 첫 주문만 표시합니다.</div>}
    </div>
  );
}
