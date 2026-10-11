import { useMemo, useState } from "react";

import { niceTicks } from "../chart/engine";
import type { OrderBookPayload, TablePayload, VRCycle, VRResult } from "../types";
import { DataTable } from "./DataTable";
import { LocLadder } from "./LocLadder";

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

function BandGauge({ cycle }: { cycle: VRCycle }) {
  const value = cycle.stock_value ?? 0;
  const lo = Math.min(cycle.v_min, value);
  const hi = Math.max(cycle.v_max, value);
  const at = (x: number) => `${(((x - lo) / (hi - lo || 1)) * 100).toFixed(2)}%`;
  return (
    <div className="vr-gauge" aria-label="V 밴드 안의 평가금 위치">
      <div className="vr-gauge-track">
        <i className="vr-gauge-band" style={{ left: at(cycle.v_min), right: `calc(100% - ${at(cycle.v_max)})` }} />
        <i className="vr-gauge-v" style={{ left: at(cycle.v) }} />
        <i className="vr-gauge-value" style={{ left: at(value) }} />
      </div>
      <div className="vr-gauge-labels">
        <span>V최소 {usd(cycle.v_min)}</span>
        <span>V {usd(cycle.v)}</span>
        <span>V최대 {usd(cycle.v_max)}</span>
      </div>
    </div>
  );
}

const CW = 720;
const CH = 240;
const PAD = { top: 16, right: 64, bottom: 28, left: 12 };

function BandChart({ cycles }: { cycles: VRCycle[] }) {
  const chart = useMemo(() => {
    const values = cycles.flatMap((c) => [c.v_min, c.v_max, c.stock_value ?? c.v]);
    const lo = Math.min(...values) * 0.97;
    const hi = Math.max(...values) * 1.03;
    const n = cycles.length;
    const x = (i: number) => PAD.left + (i / Math.max(n - 1, 1)) * (CW - PAD.left - PAD.right);
    const y = (v: number) => PAD.top + ((hi - v) / (hi - lo)) * (CH - PAD.top - PAD.bottom);
    const line = (pick: (c: VRCycle) => number | null) =>
      cycles.map((c, i) => [i, pick(c)] as const).filter(([, v]) => v != null)
        .map(([i, v], k) => `${k ? "L" : "M"}${x(i).toFixed(1)},${y(v as number).toFixed(1)}`).join("");
    const band = `${cycles.map((c, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(c.v_max).toFixed(1)}`).join("")}`
      + `${[...cycles].reverse().map((c, k) => `L${x(n - 1 - k).toFixed(1)},${y(c.v_min).toFixed(1)}`).join("")}Z`;
    return { x, y, band, v: line((c) => c.v), value: line((c) => c.stock_value), ticks: niceTicks(lo, hi, 4) };
  }, [cycles]);

  if (cycles.length < 2) {
    return <div className="compact-empty">사이클이 2개 이상 쌓이면 V 밴드와 평가금 추이를 그립니다.</div>;
  }
  return (
    <svg className="vr-chart" viewBox={`0 0 ${CW} ${CH}`} role="img" aria-label="사이클별 V 밴드와 평가금">
      {chart.ticks.map((t) => (
        <g key={t}>
          <line x1={PAD.left} x2={CW - PAD.right} y1={chart.y(t)} y2={chart.y(t)} className="vr-chart-grid" />
          <text x={CW - PAD.right + 8} y={chart.y(t) + 4} className="vr-chart-tick">{t >= 1000 ? `${(t / 1000).toFixed(0)}k` : t}</text>
        </g>
      ))}
      <path d={chart.band} className="vr-chart-band" />
      <path d={chart.v} className="vr-chart-v" />
      <path d={chart.value} className="vr-chart-value" />
      <text x={PAD.left} y={CH - 8} className="vr-chart-tick">{cycles[0].start}</text>
      <text x={CW - PAD.right} y={CH - 8} textAnchor="end" className="vr-chart-tick">{cycles[cycles.length - 1].start}</text>
    </svg>
  );
}

export function VrView({ result }: { result: VRResult }) {
  const { current, account } = result;
  const [showAll, setShowAll] = useState(false);
  const ladder = useMemo(() => ladderPayload(result, showAll), [result, showAll]);
  const openCount = result.orders.filter((order) => !order.filled_on).length;
  const pv = current.pool / current.v;
  const formula = account.advanced ? "V + P/G + (E−V)/(2√G)" : "V + P/G";

  return (
    <div className="result-stack">
      <section className="order-overview-grid">
        <article className="current-mode-card">
          <header>
            <span className="eyebrow">{account.start_week + current.index * 2}주차 // CYCLE {current.index + 1}</span>
            <small>{account.ticker} · {shortDate(current.start)} ~ {shortDate(current.end)}</small>
          </header>
          <div className="current-mode-title"><strong>{usd(current.v)}</strong><span>목표 V · G {current.g}</span></div>
          <p>
            다음 V = <b>{formula}</b>{account.deposit ? <> + 적립 <b>{usd(account.deposit)}</b></> : null}.
            밴드 ±{account.band_pct}%, 매수는 사이클 시작 Pool의 {account.pool_limit_pct}%까지.
          </p>
          <BandGauge cycle={current} />
        </article>
        <section className="snapshot-grid">
          <article><span>평가금 // STOCK</span><strong>{usd(current.stock_value)}</strong><small>{current.shares.toLocaleString()}주 × {usd(current.close, 2)} ({current.close_date ?? "—"})</small></article>
          <article><span>POOL // CASH</span><strong>{usd(current.pool)}</strong><small>P/V {pv.toFixed(3)}</small></article>
          <article><span>계좌총액 // TOTAL</span><strong>{usd(current.total)}</strong><small>투자원금 {usd(current.principal)}</small></article>
          <article><span>수익률 // RETURN</span><strong className={current.return_pct != null && current.return_pct < 0 ? "negative" : "lime"}>{current.return_pct == null ? "—" : `${current.return_pct.toFixed(2)}%`}</strong><small>원금 대비</small></article>
        </section>
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

      <section className="chart-card">
        <header className="panel-header"><span className="eyebrow">V 밴드 // VALUE BAND</span><small>점선 V · 띠 V최소~V최대 · 흰 선 사이클 마지막 종가 평가금</small></header>
        <BandChart cycles={result.cycles} />
      </section>

      <DataTable title="사이클 기록" eyebrow="CYCLES" table={cycleTable(result.cycles, account.start_week)} filename={`vr_cycles_${account.id}.csv`} limit={0} />

      <footer className="footer">
        <span>VR // {account.name}</span>
        <span>실제 체결과 다르면 config/vr/{account.id}.json의 adjustments에 사이클 시작 보유·Pool을 적으세요</span>
      </footer>
    </div>
  );
}
