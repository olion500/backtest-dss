import { type RefObject, useEffect, useMemo, useRef, useState } from "react";

import { niceTicks } from "../chart/engine";
import type { VRDay, VRResult } from "../types";

const PAD = { top: 14, right: 52, bottom: 26, left: 8 };
const RANGES: Array<[string, number | null]> = [["3M", 63], ["6M", 126], ["1Y", 252], ["ALL", null]];
const DEFAULT_RANGE = 126;

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

function tickLabel(value: number) {
  return value >= 1000 ? `${(value / 1000).toLocaleString("en-US", { maximumFractionDigits: 1 })}k` : String(value);
}

/** 계단(step-after) 경로: 하루 안에서는 값이 그대로, 다음 날 x에서 꺾인다 */
function stepPath(days: VRDay[], x: (i: number) => number, y: (v: number) => number, pick: (d: VRDay) => number) {
  return days.map((d, i) => {
    const next = i + 1 < days.length ? x(i + 1) : x(i);
    return `${i ? "L" : "M"}${x(i).toFixed(1)},${y(pick(d)).toFixed(1)}H${next.toFixed(1)}`;
  }).join("");
}

export function VrBandChart({ result }: { result: VRResult }) {
  const all = result.daily;
  const [range, setRange] = useState<number | null>(all.length > DEFAULT_RANGE ? DEFAULT_RANGE : null);
  const [hover, setHover] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const boxRef = useRef<HTMLElement>(null);
  const W = useWidth(boxRef, 960);
  const H = W < 600 ? 240 : 300;
  const days = useMemo(() => (range == null ? all : all.slice(-range)), [all, range]);

  const chart = useMemo(() => {
    if (days.length < 2) return null;
    const values = days.flatMap((d) => [d.v_min, d.v_max, d.stock_value]);
    // 밴드가 차트를 꽉 채우면 상한·하한이 테두리에 묻힌다: 위아래로 여백을 둔다
    const span = Math.max(...values) - Math.min(...values);
    const lo = Math.max(0, Math.min(...values) - span * 0.12);
    const hi = Math.max(...values) + span * 0.12;
    const n = days.length;
    const x = (i: number) => PAD.left + (i / (n - 1)) * (W - PAD.left - PAD.right);
    const y = (v: number) => PAD.top + ((hi - v) / (hi - lo)) * (H - PAD.top - PAD.bottom);
    const top = stepPath(days, x, y, (d) => d.v_max);
    const bottom = [...days].reverse().map((d, k) => {
      const i = n - 1 - k;
      const next = i + 1 < n ? x(i + 1) : x(i);
      return `L${next.toFixed(1)},${y(d.v_min).toFixed(1)}H${x(i).toFixed(1)}`;
    }).join("");
    const value = days.map((d, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(d.stock_value).toFixed(1)}`).join("");
    const index = new Map(days.map((d, i) => [d.date, i]));
    const fills = result.fills.flatMap((f) => {
      const i = index.get(f.date);
      return i == null ? [] : [{ ...f, cx: x(i), cy: y(days[i].stock_value) }];
    });
    const webStart = days.findIndex((d) => d.source === "web");
    return {
      x, y, band: `${top}${bottom}Z`, top, bottom: stepPath(days, x, y, (d) => d.v_min),
      v: stepPath(days, x, y, (d) => d.v), value, fills, ticks: niceTicks(lo, hi, 4),
      webStart: webStart > 0 ? x(webStart) : null,
    };
  }, [days, result.fills, W, H]);

  if (!chart) {
    return (
      <section className="chart-card" ref={boxRef}>
        <header className="panel-header"><span className="eyebrow">V 밴드 // VALUE BAND</span></header>
        <div className="compact-empty">거래일이 2일 이상 쌓이면 V 밴드와 평가금 추이를 그립니다.</div>
      </section>
    );
  }

  const shown = hover != null ? days[hover] : days[days.length - 1];
  const position = (shown.stock_value / shown.v - 1) * 100;
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
        <span className="eyebrow">V 밴드 // VALUE BAND</span>
        <div className="vr-chart-readout">
          <span>{shown.date}</span>
          <span>평가금 <b>{usd(shown.stock_value)}</b></span>
          <span>V <b>{usd(shown.v)}</b> <em className={position < 0 ? "vr-down" : "vr-up"}>{position > 0 ? "+" : ""}{position.toFixed(1)}%</em></span>
          <span>{usd(shown.v_min)} ~ {usd(shown.v_max)}</span>
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
        aria-label="일별 평가금과 사이클별 V 밴드"
        onPointerMove={(event) => onMove(event.clientX)}
        onPointerLeave={() => setHover(null)}
      >
        {chart.ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={chart.y(t)} y2={chart.y(t)} className="vr-chart-grid" />
            <text x={W - PAD.right + 8} y={chart.y(t) + 4} className="vr-chart-tick">{tickLabel(t)}</text>
          </g>
        ))}
        <path d={chart.band} className="vr-chart-band" />
        <path d={chart.top} className="vr-chart-edge max" />
        <path d={chart.bottom} className="vr-chart-edge min" />
        {chart.webStart != null && (
          <g>
            <line x1={chart.webStart} x2={chart.webStart} y1={PAD.top} y2={H - PAD.bottom} className="vr-chart-split" />
            <text x={chart.webStart - 6} y={PAD.top + 12} textAnchor="end" className="vr-chart-tick">시트 기록</text>
            <text x={chart.webStart + 6} y={PAD.top + 12} className="vr-chart-tick">웹 계산</text>
          </g>
        )}
        <path d={chart.v} className="vr-chart-v" />
        <path d={chart.value} className="vr-chart-value" />
        {chart.fills.map((f, k) => (
          <path key={k} className={`vr-chart-fill ${f.side}`}
            d={f.side === "buy" ? `M${f.cx},${f.cy + 4}l-5,9h10z` : `M${f.cx},${f.cy - 4}l-5,-9h10z`}>
            <title>{`${f.date} ${f.side === "buy" ? "매수" : "매도"} ${f.qty}주 @ ${f.price.toFixed(2)}`}</title>
          </path>
        ))}
        {hover != null && <line x1={chart.x(hover)} x2={chart.x(hover)} y1={PAD.top} y2={H - PAD.bottom} className="vr-chart-cursor" />}
        <text x={PAD.left} y={H - 6} className="vr-chart-tick">{days[0].date}</text>
        <text x={W - PAD.right} y={H - 6} textAnchor="end" className="vr-chart-tick">{days[days.length - 1].date}</text>
      </svg>
      <footer className="vr-chart-legend">
        <span><i className="swatch vmax" />V최대 (넘으면 매도)</span>
        <span><i className="swatch vmin" />V최소 (밑돌면 매수)</span>
        <span><i className="swatch v" />V</span>
        <span><i className="swatch value" />평가금 (종가)</span>
        <span><i className="swatch buy" />매수 체결</span>
        <span><i className="swatch sell" />매도 체결</span>
      </footer>
    </section>
  );
}
