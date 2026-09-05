import { useMemo, useRef, useState } from "react";

import type { EquityPoint, TablePayload } from "../types";

interface EquityChartProps {
  points: EquityPoint[];
  modeBands: TablePayload;
  logScale: boolean;
  target: string;
}

const WIDTH = 1000;
const HEIGHT = 360;
const PADDING = { top: 24, right: 28, bottom: 36, left: 34 };

function linePath(values: Array<number | null>, scale: (value: number) => number) {
  let path = "";
  values.forEach((value, index) => {
    if (value == null || !Number.isFinite(value)) return;
    const x = PADDING.left + (index / Math.max(values.length - 1, 1)) * (WIDTH - PADDING.left - PADDING.right);
    const y = scale(value);
    path += `${path ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
  });
  return path;
}

function createScale(values: Array<number | null>, logScale: boolean) {
  const transform = (value: number) => logScale ? Math.log10(Math.max(value, 0.0001)) : value;
  const valid = values.filter((value): value is number => value != null && Number.isFinite(value)).map(transform);
  const min = Math.min(...valid);
  const max = Math.max(...valid);
  const span = max - min || 1;
  return (value: number) => PADDING.top + ((max - transform(value)) / span) * (HEIGHT - PADDING.top - PADDING.bottom);
}

export function EquityChart({ points, modeBands, logScale, target }: EquityChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const equityValues = useMemo(() => points.map((point) => point.equity), [points]);
  const priceValues = useMemo(() => points.map((point) => point.price), [points]);
  const equityScale = useMemo(() => createScale(equityValues, logScale), [equityValues, logScale]);
  const priceScale = useMemo(() => createScale(priceValues, logScale), [priceValues, logScale]);
  const dateIndex = useMemo(() => new Map(points.map((point, index) => [point.date.slice(0, 10), index])), [points]);
  const hoverPoint = hoverIndex == null ? null : points[hoverIndex];
  const hoverX = hoverIndex == null
    ? 0
    : PADDING.left + (hoverIndex / Math.max(points.length - 1, 1)) * (WIDTH - PADDING.left - PADDING.right);

  const onPointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const bounds = svgRef.current?.getBoundingClientRect();
    if (!bounds) return;
    const relative = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
    setHoverIndex(Math.round(relative * Math.max(points.length - 1, 0)));
  };

  return (
    <section className="chart-card">
      <header className="section-heading">
        <div>
          <span className="eyebrow">PERFORMANCE / DAILY</span>
          <h2>Equity Curve vs {target}</h2>
        </div>
        <div className="chart-legend">
          <span><i className="lime-dot" /> EQUITY</span>
          <span><i className="magenta-dot" /> PRICE</span>
          <span>{logScale ? "LOG" : "LINEAR"}</span>
        </div>
      </header>
      <div className="chart-wrap">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label="Equity and target price chart"
          onPointerMove={onPointerMove}
          onPointerLeave={() => setHoverIndex(null)}
        >
          <defs>
            <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stopColor="#d4ff00" stopOpacity="0.22" />
              <stop offset="1" stopColor="#d4ff00" stopOpacity="0" />
            </linearGradient>
          </defs>
          {[0, 1, 2, 3, 4].map((line) => {
            const y = PADDING.top + (line / 4) * (HEIGHT - PADDING.top - PADDING.bottom);
            return <line key={line} x1={PADDING.left} x2={WIDTH - PADDING.right} y1={y} y2={y} className="grid-line" />;
          })}
          {modeBands.rows.map((band, index) => {
            if (band.mode !== "공세") return null;
            const start = dateIndex.get(String(band.start).slice(0, 10)) ?? 0;
            const end = dateIndex.get(String(band.end).slice(0, 10)) ?? start;
            const x = PADDING.left + (start / Math.max(points.length - 1, 1)) * (WIDTH - PADDING.left - PADDING.right);
            const width = Math.max(2, ((end - start + 1) / Math.max(points.length, 1)) * (WIDTH - PADDING.left - PADDING.right));
            return <rect key={index} x={x} y={PADDING.top} width={width} height={HEIGHT - PADDING.top - PADDING.bottom} className="mode-band" />;
          })}
          <path
            d={`${linePath(equityValues, equityScale)} L${WIDTH - PADDING.right},${HEIGHT - PADDING.bottom} L${PADDING.left},${HEIGHT - PADDING.bottom} Z`}
            fill="url(#equityFill)"
          />
          <path d={linePath(equityValues, equityScale)} className="equity-line" />
          <path d={linePath(priceValues, priceScale)} className="price-line" />
          {hoverPoint && (
            <>
              <line x1={hoverX} x2={hoverX} y1={PADDING.top} y2={HEIGHT - PADDING.bottom} className="hover-line" />
              {hoverPoint.equity != null && <circle cx={hoverX} cy={equityScale(hoverPoint.equity)} r="4" className="equity-point" />}
              {hoverPoint.price != null && <circle cx={hoverX} cy={priceScale(hoverPoint.price)} r="4" className="price-point" />}
            </>
          )}
          <text x={PADDING.left} y={HEIGHT - 10} className="axis-label">{points[0]?.date.slice(0, 10)}</text>
          <text x={WIDTH - PADDING.right} y={HEIGHT - 10} textAnchor="end" className="axis-label">{points.at(-1)?.date.slice(0, 10)}</text>
        </svg>
        {hoverPoint && (
          <div className="chart-tooltip" style={{ left: `${Math.min(82, Math.max(4, (hoverX / WIDTH) * 100))}%` }}>
            <span>{hoverPoint.date.slice(0, 10)}</span>
            <strong>${hoverPoint.equity?.toLocaleString("en-US", { maximumFractionDigits: 0 }) ?? "—"}</strong>
            <em>{target} ${hoverPoint.price?.toFixed(2) ?? "—"}</em>
          </div>
        )}
      </div>
    </section>
  );
}
