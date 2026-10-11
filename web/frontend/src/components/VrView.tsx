import { useMemo, useState } from "react";

import type { OrderBookPayload, TablePayload, VRCycle, VRResult } from "../types";
import { DataTable } from "./DataTable";
import { LocLadder } from "./LocLadder";
import { VrBandChart } from "./VrBandChart";

// 매도표는 현재가 +40%까지라 100줄이 넘는다. 종가에 가까운 주문부터 이만큼만 먼저 보여준다
const NEAR_ORDERS = 10;

function usd(value: number | null | undefined, digits = 0) {
  if (value == null) return "—";
  return `$${value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
}

function shortDate(value: string) {
  return value.slice(5).replace("-", ".");
}

/** 현재 사이클 주문을 LocLadder가 읽는 주문표 형태로 바꾼다 */
function ladderPayload(result: VRResult, showAll: boolean): OrderBookPayload {
  const { current } = result;
  const open = result.orders.filter((order) => !order.filled_on);
  const nearest = (side: "buy" | "sell") => {
    const rows = open.filter((order) => order.side === side);
    return showAll ? rows : rows.slice(0, NEAR_ORDERS);
  };
  const filled = result.orders.filter((order) => order.filled_on);
  let buyTotal = current.shares_start;
  let sellTotal = current.shares_start;
  const note = (order: VRResult["orders"][number]) => {
    if (order.side === "buy") buyTotal += order.qty;
    else sellTotal -= order.qty;
    return `체결 시 보유 ${(order.side === "buy" ? buyTotal : sellTotal).toLocaleString()}주`;
  };
  const rows = [
    ...filled.map((order) => ({
      구분: `${order.side === "buy" ? "매수" : "매도"} (체결)`,
      주문가: order.price,
      수량: order.qty,
      비고: `${shortDate(order.filled_on ?? "")} ${usd(order.fill_price, 2)} 체결 추정`,
    })),
    ...[...nearest("buy"), ...nearest("sell")].map((order) => ({
      구분: order.side === "buy" ? "매수" : "매도",
      주문가: order.price,
      수량: order.qty,
      비고: note(order),
    })),
  ];
  return {
    state: {
      last_date: current.close_date ?? current.start,
      current_mode: "defense",
      current_cash: current.pool,
      current_position_qty: current.shares,
      prev_close: current.close,
      tranche_budget: null,
      rsi_value: null,
    },
    holdings: { columns: [], rows: [] },
    orders: { columns: ["구분", "주문가", "수량", "비고"], rows },
    netting_message: "",
    netting_details: { columns: [], rows: [] },
    netting_scenarios: { columns: [], rows: [] },
  };
}

function cycleTable(cycles: VRCycle[], startWeek: number): TablePayload {
  const columns = ["주차", "시작", "종료", "G", "V", "V최소", "V최대", "적립", "매수", "매도", "보유", "종가", "평가금", "Pool", "계좌총액", "투자원금", "수익률"];
  return {
    columns,
    rows: cycles.map((c) => ({
      주차: startWeek + c.index * 2,
      시작: c.start,
      종료: c.end,
      G: c.g,
      V: c.v,
      V최소: c.v_min,
      V최대: c.v_max,
      적립: c.deposit,
      매수: c.bought,
      매도: c.sold,
      보유: c.shares,
      종가: c.close,
      평가금: c.stock_value,
      Pool: c.pool,
      계좌총액: c.total,
      투자원금: c.principal,
      수익률: c.return_pct == null ? null : `${c.return_pct.toFixed(2)}%`,
    })),
  };
}

function pct(value: number) {
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
}

/** 매수선(첫 매수 주문가) ~ 매도선(첫 매도 주문가) 사이 현재가 위치 */
function PriceCorridor({ result }: { result: VRResult }) {
  const { current, outlook } = result;
  const close = current.close;
  const { buy_line: buy, sell_line: sell } = outlook;
  if (close == null) return null;
  const lo = Math.min(buy ?? close, close);
  const hi = Math.max(sell ?? close, close);
  const at = (value: number) => `${(((value - lo) / (hi - lo || 1)) * 100).toFixed(2)}%`;
  return (
    <div className="vr-corridor">
      <div className="vr-corridor-ends">
        <span className="buy">매수선 <b>{buy == null ? "없음" : usd(buy, 2)}</b></span>
        <span className="now">현재 <b>{usd(close, 2)}</b></span>
        <span className="sell">매도선 <b>{sell == null ? "없음" : usd(sell, 2)}</b></span>
      </div>
      <div className="vr-corridor-track" aria-hidden>
        <i className="vr-corridor-dot" style={{ left: at(close) }} />
      </div>
      <div className="vr-corridor-notes">
        <span>{buy == null ? "매수 주문 없음 (Pool 한도)" : `${pct((buy / close - 1) * 100)} 내려가면 매수`}</span>
        <span>{current.close_date ?? "—"} 종가</span>
        <span>{sell == null ? "매도 주문 없음" : `${pct((sell / close - 1) * 100)} 오르면 매도`}</span>
      </div>
    </div>
  );
}

export function VrView({ result }: { result: VRResult }) {
  const { current, account, outlook } = result;
  const [showAll, setShowAll] = useState(false);
  const ladder = useMemo(() => ladderPayload(result, showAll), [result, showAll]);
  const openCount = result.orders.filter((order) => !order.filled_on).length;
  const total = current.total ?? 0;
  const profit = total - current.principal;
  const stock = current.stock_value ?? 0;
  const stockShare = total ? (stock / total) * 100 : 0;
  const vGap = (stock / current.v - 1) * 100;
  const budgetUsed = outlook.buy_budget ? 1 - outlook.buy_budget_left / outlook.buy_budget : 0;
  const formula = account.advanced ? "V + P/G + (E−V)/(2√G)" : "V + P/G";

  return (
    <div className="result-stack">
      <section className="vr-hero">
        <header>
          <div className="vr-hero-cycle">
            <span className="eyebrow">{account.start_week + current.index * 2}주차 // {account.ticker}</span>
            <strong>{shortDate(current.start)} ~ {shortDate(current.end)}</strong>
            <span className="vr-days" aria-label={`사이클 ${outlook.day}일째`}>
              {Array.from({ length: outlook.cycle_days }, (_, i) => <i key={i} className={i < outlook.day ? "on" : ""} />)}
              <em>{outlook.day}/{outlook.cycle_days}일째</em>
            </span>
          </div>
          <div className="vr-hero-total">
            <span className="eyebrow">계좌총액 // TOTAL</span>
            <strong>{usd(current.total)}</strong>
            <small className={profit < 0 ? "negative" : "lime"}>
              {current.return_pct == null ? "—" : pct(current.return_pct)} ({profit < 0 ? "−" : "+"}{usd(Math.abs(profit))})
            </small>
            <small>원금 {usd(current.principal)}</small>
          </div>
        </header>
        <PriceCorridor result={result} />
      </section>

      <section className="vr-stats">
        <article>
          <span>목표 V // TARGET</span>
          <strong>{usd(current.v)}</strong>
          <small>G {current.g} · ±{account.band_pct}%</small>
          <small>{usd(current.v_min)} ~ {usd(current.v_max)}</small>
        </article>
        <article>
          <span>평가금 // STOCK</span>
          <strong>{usd(current.stock_value)}</strong>
          <small>{current.shares.toLocaleString()}주 · 주식 {stockShare.toFixed(0)}%</small>
          <small className={vGap < 0 ? "vr-down" : "vr-up"}>V 대비 {pct(vGap)}</small>
        </article>
        <article>
          <span>POOL // CASH</span>
          <strong>{usd(current.pool)}</strong>
          <small>P/V {(current.pool / current.v).toFixed(3)}</small>
          <small>매수 한도 남음 {usd(outlook.buy_budget_left)} / {account.pool_limit_pct}%</small>
          <i className="vr-meter" aria-hidden><b style={{ width: `${(Math.min(1, Math.max(0, budgetUsed)) * 100).toFixed(1)}%` }} /></i>
        </article>
        <article>
          <span>다음 사이클 // {shortDate(outlook.next_start)}</span>
          <strong>{usd(outlook.next_v)}</strong>
          <small>예상 V · G {outlook.next_g}{outlook.next_deposit ? ` · 적립 ${outlook.next_deposit > 0 ? "+" : "−"}${usd(Math.abs(outlook.next_deposit))}` : ""}</small>
          <small>지금 종가로 계산 · {formula}</small>
        </article>
      </section>

      <LocLadder
        orderBook={ladder}
        heading={`// ${shortDate(current.start)} ~ ${shortDate(current.end)} 예약 주문`}
        subheading={`${account.order_type === "loc" ? "LOC" : "지정가"} · 사이클 동안 유지 · 체결은 일별 가격으로 추정`}
        footer={`매수 ${account.buy_unit}주 묶음 · 매도는 현재가 +${account.sell_cap_pct}%까지 · 주문가 클릭 시 복사`}
        action={openCount > NEAR_ORDERS * 2 || showAll ? (
          <button className="button ghost" type="button" onClick={() => setShowAll((value) => !value)}>
            {showAll ? `가까운 ${NEAR_ORDERS}건씩` : `전체 ${openCount}건`}
          </button>
        ) : undefined}
      />

      <VrBandChart result={result} />

      <DataTable title="사이클 기록" eyebrow="CYCLES" table={cycleTable(result.cycles, account.start_week)} filename={`vr_cycles_${account.id}.csv`} limit={1000} />

      <footer className="footer">
        <span>VR // {account.name}</span>
        <span>실제 체결과 다르면 config/vr/{account.id}.json의 adjustments에 사이클 시작 보유·Pool을 적으세요</span>
      </footer>
    </div>
  );
}
