import { type RefObject, useEffect, useId, useMemo, useRef, useState } from "react";

import { niceTicks } from "../chart/engine";
import type { VRDay, VRResult } from "../types";

const PAD = { top: 14, right: 52, bottom: 26, left: 8 };
const RANGES: Array<[string, number | null]> = [["3M", 63], ["6M", 126], ["1Y", 252], ["ALL", null]];
const DEFAULT_RANGE = 126;

type Mode = "band" | "account" | "return";
const MODES: Array<[Mode, string]> = [["band", "V 밴드"], ["account", "계좌"], ["return", "수익률"]];

/** viewBox를 실제 폭에 맞춰 글자 크기가 화면 폭과 상관없이 같게 보이도록 한다 */
function useWidth(ref: RefObject<HTMLElement | null>, fallback: number) {
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(280, Math.round(entry.contentRect.width))));
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref]);
  return width;
}

function usd(value: number) {
  return `$${value.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

function signedPct(value: number) {
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
}

function tickLabel(value: number, mode: Mode) {
  if (mode === "return") return `${value}%`;
  return value >= 1000 ? `${(value / 1000).toLocaleString("en-US", { maximumFractionDigits: 1 })}k` : String(value);
}

const total = (d: VRDay) => d.stock_value + d.pool;
const returnPct = (d: VRDay) => (d.principal ? (total(d) / d.principal - 1) * 100 : 0);

/** 계단(step-after) 경로: 하루 안에서는 값이 그대로, 다음 날 x에서 꺾인다 */
function stepPath(days: VRDay[], x: (i: number) => number, y: (v: number) => number, pick: (d: VRDay) => number) {
  return days.map((d, i) => {
    const next = i + 1 < days.length ? x(i + 1) : x(i);
    return `${i ? "L" : "M"}${x(i).toFixed(1)},${y(pick(d)).toFixed(1)}H${next.toFixed(1)}`;
  }).join("");
}

function linePath(days: VRDay[], x: (i: number) => number, y: (v: number) => number, pick: (d: VRDay) => number) {
  return days.map((d, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(pick(d)).toFixed(1)}`).join("");
}

/** 값 범위 + 위아래 여백 (밴드나 선이 차트 테두리에 묻히지 않게) */
function paddedRange(values: number[], floorAtZero: boolean) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || Math.abs(max) || 1;
  return [floorAtZero ? Math.max(0, min - span * 0.12) : min - span * 0.12, max + span * 0.12];
}

export function VrBandChart({ result }: { result: VRResult }) {
  const all = result.daily;
  const [mode, setMode] = useState<Mode>("band");
  const [range, setRange] = useState<number | null>(all.length > DEFAULT_RANGE ? DEFAULT_RANGE : null);
  const [hover, setHover] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const boxRef = useRef<HTMLElement>(null);
  const clipId = useId().replace(/:/g, "");
  const W = useWidth(boxRef, 960);
  const H = W < 600 ? 240 : 300;
  const days = useMemo(() => (range == null ? all : all.slice(-range)), [all, range]);

  const chart = useMemo(() => {
    if (days.length < 2) return null;
    const values = mode === "band"
      ? days.flatMap((d) => [d.v_min, d.v_max, d.stock_value])
      : mode === "account" ? days.flatMap((d) => [total(d), d.principal]) : [0, ...days.map(returnPct)];
    const [lo, hi] = paddedRange(values, mode !== "return");
    const n = days.length;
    const x = (i: number) => PAD.left + (i / (n - 1)) * (W - PAD.left - PAD.right);
    const y = (v: number) => PAD.top + ((hi - v) / (hi - lo)) * (H - PAD.top - PAD.bottom);
    const ticks = niceTicks(lo, hi, 4);
    if (mode === "return") {
      return { mode, x, y, ticks, line: linePath(days, x, y, returnPct), zero: y(0) };
    }
    if (mode === "account") {
      const principal = stepPath(days, x, y, (d) => d.principal);
      const totalLine = linePath(days, x, y, total);
      // 계좌총액과 원금 사이 영역: 원금 위쪽은 이익, 아래쪽은 손실 색으로 잘라서 칠한다
      const back = [...days].reverse().map((d, k) => {
        const i = n - 1 - k;
        const next = i + 1 < n ? x(i + 1) : x(i);
        return `L${next.toFixed(1)},${y(d.principal).toFixed(1)}H${x(i).toFixed(1)}`;
      }).join("");
      const gap = `${totalLine}${back}Z`;
      const right = W - PAD.right;
      const above = `${principal}V${PAD.top}H${PAD.left}Z`;
      const below = `${principal}V${H - PAD.bottom}H${PAD.left}Z`;
      return { mode, x, y, ticks, principal, totalLine, gap, above, below, right };
    }
    const top = stepPath(days, x, y, (d) => d.v_max);
    const bottom = [...days].reverse().map((d, k) => {
      const i = n - 1 - k;
      const next = i + 1 < n ? x(i + 1) : x(i);
      return `L${next.toFixed(1)},${y(d.v_min).toFixed(1)}H${x(i).toFixed(1)}`;
    }).join("");
    const index = new Map(days.map((d, i) => [d.date, i]));
    const fills = result.fills.flatMap((f) => {
      const i = index.get(f.date);
      return i == null ? [] : [{ ...f, cx: x(i), cy: y(days[i].stock_value) }];
    });
    return {
      mode, x, y, ticks, band: `${top}${bottom}Z`, top, bottom: stepPath(days, x, y, (d) => d.v_min),
      v: stepPath(days, x, y, (d) => d.v), value: linePath(days, x, y, (d) => d.stock_value), fills,
    };
  }, [days, mode, result.fills, W, H]);

  if (!chart) {
    return (
      <section className="chart-card" ref={boxRef}>
        <header className="panel-header"><span className="eyebrow">VR 차트</span></header>
        <div className="compact-empty">거래일이 2일 이상 쌓이면 V 밴드와 계좌 추이를 그립니다.</div>
      </section>
    );
  }

  const shown = hover != null ? days[hover] : days[days.length - 1];
  const position = (shown.stock_value / shown.v - 1) * 100;
  const shownReturn = returnPct(shown);
  const onMove = (clientX: number) => {
    const box = svgRef.current?.getBoundingClientRect();
    if (!box) return;
    const ratio = ((clientX - box.left) / box.width) * W;
    const t = (ratio - PAD.left) / (W - PAD.left - PAD.right);
    setHover(Math.max(0, Math.min(days.length - 1, Math.round(t * (days.length - 1)))));
  };

  return (
    <section className="chart-card vr-chart-card" ref={boxRef}>
      <header className="panel-header">
        <div className="vr-range vr-mode" role="tablist" aria-label="차트 종류">
          {MODES.map(([key, label]) => (
            <button key={key} type="button" role="tab" aria-selected={mode === key} className={mode === key ? "active" : ""}
              onClick={() => setMode(key)}>{label}</button>
          ))}
        </div>
        <div className="vr-chart-readout">
          <span>{shown.date}</span>
          {mode === "band" ? (
            <>
              <span>평가금 <b>{usd(shown.stock_value)}</b></span>
              <span>V <b>{usd(shown.v)}</b> <em className={position < 0 ? "vr-down" : "vr-up"}>{signedPct(position)}</em></span>
              <span>{usd(shown.v_min)} ~ {usd(shown.v_max)}</span>
            </>
          ) : (
            <>
              <span>계좌총액 <b>{usd(total(shown))}</b></span>
              <span>원금 <b>{usd(shown.principal)}</b></span>
              <span>수익률 <b className={shownReturn < 0 ? "vr-down" : "vr-up"}>{signedPct(shownReturn)}</b></span>
            </>
          )}
        </div>
        <div className="vr-range" role="group" aria-label="기간">
          {RANGES.map(([label, size]) => (
            <button key={label} type="button" className={range === size ? "active" : ""} disabled={size != null && size >= all.length}
              onClick={() => { setRange(size); setHover(null); }}>{label}</button>
          ))}
        </div>
      </header>
      <svg
        ref={svgRef}
        className="vr-chart"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={mode === "band" ? "일별 평가금과 사이클별 V 밴드" : mode === "account" ? "계좌총액과 투자원금" : "투자원금 대비 수익률"}
        onPointerMove={(event) => onMove(event.clientX)}
        onPointerLeave={() => setHover(null)}
      >
        {chart.ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={chart.y(t)} y2={chart.y(t)} className="vr-chart-grid" />
            <text x={W - PAD.right + 8} y={chart.y(t) + 4} className="vr-chart-tick">{tickLabel(t, mode)}</text>
          </g>
        ))}
        {chart.mode === "band" && (
          <>
            <path d={chart.band} className="vr-chart-band" />
            <path d={chart.top} className="vr-chart-edge max" />
            <path d={chart.bottom} className="vr-chart-edge min" />
            <path d={chart.v} className="vr-chart-v" />
            <path d={chart.value} className="vr-chart-value" />
            {chart.fills.map((f, k) => (
              <path key={k} className={`vr-chart-fill ${f.side}`}
                d={f.side === "buy" ? `M${f.cx},${f.cy + 4}l-5,9h10z` : `M${f.cx},${f.cy - 4}l-5,-9h10z`}>
                <title>{`${f.date} ${f.side === "buy" ? "매수" : "매도"} ${f.qty}주 @ ${f.price.toFixed(2)}`}</title>
              </path>
            ))}
          </>
        )}
        {chart.mode === "account" && (
          <>
            <defs>
              <clipPath id={`${clipId}-up`}><path d={chart.above} /></clipPath>
              <clipPath id={`${clipId}-down`}><path d={chart.below} /></clipPath>
            </defs>
            <path d={chart.gap} className="vr-chart-gain" clipPath={`url(#${clipId}-up)`} />
            <path d={chart.gap} className="vr-chart-loss" clipPath={`url(#${clipId}-down)`} />
            <path d={chart.principal} className="vr-chart-principal" />
            <path d={chart.totalLine} className="vr-chart-total" />
          </>
        )}
        {chart.mode === "return" && (
          <>
            <line x1={PAD.left} x2={W - PAD.right} y1={chart.zero} y2={chart.zero} className="vr-chart-zero" />
            <path d={chart.line} className="vr-chart-total" />
          </>
        )}
        {hover != null && <line x1={chart.x(hover)} x2={chart.x(hover)} y1={PAD.top} y2={H - PAD.bottom} className="vr-chart-cursor" />}
        <text x={PAD.left} y={H - 6} className="vr-chart-tick">{days[0].date}</text>
        <text x={W - PAD.right} y={H - 6} textAnchor="end" className="vr-chart-tick">{days[days.length - 1].date}</text>
      </svg>
      <footer className="vr-chart-legend">
        {mode === "band" && (
          <>
            <span><i className="swatch vmax" />V최대 (넘으면 매도)</span>
            <span><i className="swatch vmin" />V최소 (밑돌면 매수)</span>
            <span><i className="swatch v" />V</span>
            <span><i className="swatch value" />평가금 (종가)</span>
            <span><i className="swatch buy" />매수 체결</span>
            <span><i className="swatch sell" />매도 체결</span>
          </>
        )}
        {mode === "account" && (
          <>
            <span><i className="swatch total" />계좌총액 (평가금 + Pool)</span>
            <span><i className="swatch principal" />투자원금 (시작 + 적립)</span>
            <span><i className="swatch gain" />이익</span>
            <span><i className="swatch loss" />손실</span>
          </>
        )}
        {mode === "return" && (
          <>
            <span><i className="swatch total" />수익률 = 계좌총액 ÷ 투자원금 − 1</span>
            <span>적립금은 수익에서 빠집니다</span>
          </>
        )}
      </footer>
    </section>
  );
}
