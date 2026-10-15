import { useMemo, useRef, useState } from "react";

import { useElementWidth } from "../chart/engine";
import type { BenchmarkSummary } from "../types";

const PAD = { top: 14, right: 56, bottom: 26, left: 8 };
// 처음 켜 둘 곡선: 운용 중인 두 전략, 혼합, 기준선
const DEFAULT_VISIBLE = ["current", "trend_vix", "mix", "hold"];

function pct(value: number, digits = 1) {
  return `${value > 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;
}

function multiple(value: number) {
  return value >= 100 ? `${value.toFixed(0)}×` : value >= 10 ? `${value.toFixed(1)}×` : `${value.toFixed(2)}×`;
}

/** 로그 축 눈금: 1·2·5 배수 중 범위 안에 드는 것 */
function logTicks(lo: number, hi: number) {
  const out: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e += 1) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v >= lo && v <= hi) out.push(v);
    }
  }
  return out.length > 7 ? out.filter((v) => Math.log10(v) % 1 === 0) : out;
}

function tickLabel(value: number) {
  return value >= 1 ? `${value.toLocaleString("en-US")}×` : `${value}×`;
}

export function BenchmarkView({ summary }: { summary: BenchmarkSummary }) {
  const [windowId, setWindowId] = useState(summary.windows[0]?.id);
  const [visible, setVisible] = useState(() => new Set(DEFAULT_VISIBLE.filter((id) => summary.strategies.some((s) => s.id === id))));
  const [hover, setHover] = useState<number | null>(null);
  const [boxRef, width] = useElementWidth<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const win = summary.windows.find((w) => w.id === windowId) ?? summary.windows[0];
  const colorIndex = useMemo(() => new Map(summary.strategies.map((s, i) => [s.id, i])), [summary.strategies]);

  const W = Math.max(280, Math.round(width || 960));
  const H = W < 600 ? 260 : 340;
  const shownIds = summary.strategies.map((s) => s.id).filter((id) => visible.has(id));

  const chart = useMemo(() => {
    const n = win.dates.length;
    if (n < 2 || !shownIds.length) return null;
    const values = shownIds.flatMap((id) => win.curves[id]).filter((v) => v > 0);
    const lo = Math.min(...values) * 0.95;
    const hi = Math.max(...values) * 1.05;
    const [llo, lhi] = [Math.log10(lo), Math.log10(hi)];
    const x = (i: number) => PAD.left + (i / (n - 1)) * (W - PAD.left - PAD.right);
    const y = (v: number) => PAD.top + ((lhi - Math.log10(Math.max(v, lo))) / (lhi - llo)) * (H - PAD.top - PAD.bottom);
    const paths = shownIds.map((id) => ({
      id,
      d: win.curves[id].map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(""),
    }));
    return { x, y, paths, ticks: logTicks(lo, hi) };
  }, [win, shownIds.join(","), W, H]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = (id: string) => setVisible((current) => {
    const next = new Set(current);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  const onMove = (clientX: number) => {
    const box = svgRef.current?.getBoundingClientRect();
    if (!box) return;
    const t = (((clientX - box.left) / box.width) * W - PAD.left) / (W - PAD.left - PAD.right);
    setHover(Math.max(0, Math.min(win.dates.length - 1, Math.round(t * (win.dates.length - 1)))));
  };
  const at = hover ?? win.dates.length - 1;
  const ranked = [...summary.strategies].sort((a, b) => win.stats[b.id].cagr - win.stats[a.id].cagr);

  return (
    <div className="result-stack benchmark">
      <section className="summary-panel">
        <header className="panel-header benchmark-header">
          <span className="eyebrow">BENCHMARK // READ ONLY</span>
          <small>데이터 {summary.data_end}까지 · 생성 {summary.generated_at.slice(0, 10)}</small>
        </header>
        <p className="benchmark-note">
          benchmarks/의 실제 엔진 결과를 그대로 보여 줍니다. 여기서는 계산하지 않습니다.
          갱신은 <code>python -m benchmarks summary</code> 실행 후 커밋·배포.
        </p>
      </section>

      <div className="range-selector benchmark-windows" role="group" aria-label="구간">
        {summary.windows.map((w) => (
          <button key={w.id} type="button" className={w.id === win.id ? "active" : ""} onClick={() => { setWindowId(w.id); setHover(null); }}>{w.label}</button>
        ))}
      </div>

      <section className="table-card">
        <header className="section-heading table-heading">
          <div><h2>{win.label} · {win.start} ~ {win.end}</h2></div>
          <div className="table-actions"><span>{win.description}</span></div>
        </header>
        <div className="table-scroll benchmark-table">
          <table>
            <thead>
              <tr><th>차트</th><th>전략</th><th>{win.total_return ? "수익률" : "CAGR"}</th><th>MDD</th><th>최종</th><th>설명</th></tr>
            </thead>
            <tbody>
              {ranked.map((s) => {
                const stat = win.stats[s.id];
                const curve = win.curves[s.id];
                return (
                  <tr key={s.id} className="copyable" onClick={() => toggle(s.id)} title="클릭하면 차트에 표시/숨김">
                    <td><i className={`bench-swatch c${colorIndex.get(s.id)}${visible.has(s.id) ? "" : " off"}`} /></td>
                    <td>{s.name}</td>
                    <td className={stat.cagr < 0 ? "tone-negative" : "tone-positive"}>{pct(stat.cagr)}</td>
                    <td className="tone-negative">{pct(stat.mdd)}</td>
                    <td>{multiple(curve[curve.length - 1])}</td>
                    <td className="tone-muted">{s.description}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="chart-card vr-chart-card">
        <header className="panel-header">
          <span className="eyebrow">자산곡선 // 시작 = 1×, 로그 축</span>
          <div className="vr-chart-readout">
            <span>{win.dates[at]}</span>
            {shownIds.map((id) => (
              <span key={id}><i className={`bench-swatch c${colorIndex.get(id)}`} /> <b>{multiple(win.curves[id][at])}</b></span>
            ))}
          </div>
        </header>
        <div ref={boxRef}>
          {chart ? (
            <svg
              ref={svgRef}
              className="vr-chart"
              viewBox={`0 0 ${W} ${H}`}
              role="img"
              aria-label={`${win.label} 구간 전략별 자산곡선`}
              onPointerMove={(event) => onMove(event.clientX)}
              onPointerLeave={() => setHover(null)}
            >
              {chart.ticks.map((t) => (
                <g key={t}>
                  <line x1={PAD.left} x2={W - PAD.right} y1={chart.y(t)} y2={chart.y(t)} className="vr-chart-grid" />
                  <text x={W - PAD.right + 8} y={chart.y(t) + 4} className="vr-chart-tick">{tickLabel(t)}</text>
                </g>
              ))}
              {chart.paths.map((p) => <path key={p.id} d={p.d} className={`bench-line c${colorIndex.get(p.id)}${p.id === "hold" ? " dashed" : ""}`} />)}
              {hover != null && <line x1={chart.x(hover)} x2={chart.x(hover)} y1={PAD.top} y2={H - PAD.bottom} className="vr-chart-cursor" />}
              <text x={PAD.left} y={H - 6} className="vr-chart-tick">{win.dates[0]}</text>
              <text x={W - PAD.right} y={H - 6} textAnchor="end" className="vr-chart-tick">{win.dates[win.dates.length - 1]}</text>
            </svg>
          ) : <div className="compact-empty">표에서 전략을 눌러 차트에 표시하세요.</div>}
        </div>
      </section>
    </div>
  );
}
