// 차트 공통 유틸. Spectrum UI(arihantcodes/spectrum-ui, Apache-2.0)의
// chart-engine 기법을 참고해 의존성 없이 다시 작성했다.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

interface XY { x: number; y: number }

/** lo~hi 구간을 사람이 읽기 좋은 간격(1·2·2.5·5·10 배수)으로 나눈 눈금 */
export function niceTicks(lo: number, hi: number, target = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return [lo];
  const raw = (hi - lo) / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 2.25 ? 2.5 : norm >= 1.5 ? 2 : 1) * mag;
  const first = Math.ceil(lo / step) * step;
  const out: number[] = [];
  for (let v = first; v <= hi + step * 0.001; v += step) out.push(Math.round(v / step) * step);
  return out;
}

/** Fritsch–Carlson 단조 큐빅 보간. 오버슈트 없이 부드러운 곡선을 만든다 */
export function monotonePath(points: XY[]): string {
  const n = points.length;
  if (n === 0) return "";
  if (n === 1) return `M${points[0].x.toFixed(2)},${points[0].y.toFixed(2)}`;
  const dx: number[] = [];
  const slope: number[] = [];
  for (let i = 0; i < n - 1; i += 1) {
    dx[i] = points[i + 1].x - points[i].x;
    slope[i] = dx[i] === 0 ? 0 : (points[i + 1].y - points[i].y) / dx[i];
  }
  const tangent = new Array<number>(n);
  tangent[0] = slope[0];
  tangent[n - 1] = slope[n - 2];
  for (let i = 1; i < n - 1; i += 1) {
    if (slope[i - 1] * slope[i] <= 0) {
      tangent[i] = 0;
    } else {
      const w1 = 2 * dx[i] + dx[i - 1];
      const w2 = dx[i] + 2 * dx[i - 1];
      tangent[i] = (w1 + w2) / (w1 / slope[i - 1] + w2 / slope[i]);
    }
  }
  let d = `M${points[0].x.toFixed(2)},${points[0].y.toFixed(2)}`;
  for (let i = 0; i < n - 1; i += 1) {
    const c1x = points[i].x + dx[i] / 3;
    const c1y = points[i].y + (tangent[i] * dx[i]) / 3;
    const c2x = points[i + 1].x - dx[i] / 3;
    const c2y = points[i + 1].y - (tangent[i + 1] * dx[i]) / 3;
    d += `C${c1x.toFixed(2)},${c1y.toFixed(2)} ${c2x.toFixed(2)},${c2y.toFixed(2)} ${points[i + 1].x.toFixed(2)},${points[i + 1].y.toFixed(2)}`;
  }
  return d;
}

/** null이 섞인 시계열을 연속 구간별로 나눠 각각 경로를 만든다 */
export function segmentedPath(values: Array<number | null>, cx: (i: number) => number, cy: (v: number) => number): string {
  let d = "";
  let run: XY[] = [];
  const flush = () => {
    if (run.length) d += monotonePath(run);
    run = [];
  };
  values.forEach((value, index) => {
    if (value == null || !Number.isFinite(value)) return flush();
    run.push({ x: cx(index), y: cy(value) });
  });
  flush();
  return d;
}

export function usePrefersReducedMotion() {
  const [reduce, setReduce] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => setReduce(mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);
  return reduce;
}

/** ResizeObserver로 실측 폭을 얻어 svg를 px 단위로 그린다 (preserveAspectRatio 왜곡 없음) */
export function useElementWidth<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const ro = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    ro.observe(node);
    setWidth(node.getBoundingClientRect().width);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

const easeOutCubic = (t: number) => 1 - (1 - t) ** 3;

/** 숫자 벡터를 목표값까지 rAF로 보간. 축 범위 전환·리드아웃 숫자 애니메이션에 쓴다 */
export function useTween(target: number[], { duration = 520, enabled = true } = {}) {
  const [value, setValue] = useState(target);
  const currentRef = useRef(target);
  const fromRef = useRef(target);
  const toRef = useRef(target);
  const startRef = useRef(0);
  const rafRef = useRef(0);

  useEffect(() => {
    if (!enabled) return;
    const to = toRef.current;
    const changed = to.length !== target.length || target.some((v, i) => v !== to[i]);
    if (!changed) return;
    toRef.current = target;
    if (currentRef.current.length !== target.length) {
      currentRef.current = target;
      fromRef.current = target;
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(() => setValue(target));
      return;
    }
    fromRef.current = currentRef.current;
    startRef.current = performance.now();
    cancelAnimationFrame(rafRef.current);
    const tick = (now: number) => {
      const p = Math.min(1, (now - startRef.current) / duration);
      const e = easeOutCubic(p);
      const next = toRef.current.map((v, i) => fromRef.current[i] + (v - fromRef.current[i]) * e);
      currentRef.current = next;
      setValue(next);
      if (p < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  });

  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

  if (!enabled || value.length !== target.length) return target;
  return value;
}

export function useTweenNumber(target: number, options?: { duration?: number; enabled?: boolean }) {
  const vec = useMemo(() => [target], [target]);
  return useTween(vec, options)[0];
}

/** 방향키로 호버 인덱스를 옮길 수 있게 하는 keydown 핸들러 */
export function useHoverIndexKeys(count: number, setIndex: (update: (current: number | null) => number | null) => void) {
  return useCallback((event: React.KeyboardEvent) => {
    if (!count) return;
    const move = (delta: number) => {
      event.preventDefault();
      setIndex((current) => Math.max(0, Math.min(count - 1, (current ?? count - 1) + delta)));
    };
    if (event.key === "ArrowLeft") move(-1);
    else if (event.key === "ArrowRight") move(1);
    else if (event.key === "Home") { event.preventDefault(); setIndex(() => 0); }
    else if (event.key === "End") { event.preventDefault(); setIndex(() => count - 1); }
    else if (event.key === "Escape") setIndex(() => null);
  }, [count, setIndex]);
}

export function formatMoney(value: number, digits = 0) {
  return `$${value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
}

export function formatAxisMoney(value: number) {
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `$${(value / 1_000_000).toFixed(abs >= 10_000_000 ? 0 : 1)}M`;
  if (abs >= 10_000) return `$${(value / 1000).toFixed(0)}K`;
  if (abs >= 1000) return `$${(value / 1000).toFixed(1)}K`;
  return `$${value.toFixed(0)}`;
}

export function formatSignedPct(value: number, digits = 2) {
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}%`;
}
