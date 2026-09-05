interface MetricGridProps {
  summary: Record<string, number | null>;
  realized: Record<string, number | null> | null;
  target: string;
  momentum: string;
  period: string;
}

type MetricStyle = "money" | "ratio" | "percent" | "number" | "days" | "count";

function formatNumber(value: number | null | undefined, style: MetricStyle) {
  if (value == null || !Number.isFinite(value)) return "—";
  if (style === "money") return `$${value.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
  if (style === "ratio") return `${(value * 100).toFixed(2)}%`;
  if (style === "percent") return `${value.toFixed(2)}%`;
  if (style === "days") return `${value.toFixed(1)}일`;
  if (style === "count") return value.toLocaleString("ko-KR", { maximumFractionDigits: 0 });
  return value.toFixed(2);
}

function Metrics({ items }: { items: ReadonlyArray<readonly [string, number | null | undefined, MetricStyle, string?]> }) {
  return (
    <div className="summary-grid">
      {items.map(([label, value, style, tone]) => (
        <div className="summary-metric" key={label}>
          <span>{label}</span>
          <strong className={tone ?? ""}>{formatNumber(value, style)}</strong>
        </div>
      ))}
    </div>
  );
}

export function MetricGrid({ summary, realized, target, momentum, period }: MetricGridProps) {
  const performance = [
    ["FINAL EQUITY", summary["Final Equity"], "money", "lime"],
    ["SHARPE / RF 0", summary["Sharpe (rf=0)"], "number"],
    ["VOLATILITY / ANN", summary["Volatility (ann)"], "ratio"],
    ["MAX DRAWDOWN", summary["Max Drawdown"], "ratio", "negative"],
    ["누적 수익률", summary["Strategy Return"], "percent", "lime"],
    [`${target} 단순보유`, summary["Target Hold Return"], "percent"],
    [`${momentum} 단순보유`, summary["Momentum Hold Return"], "percent"],
    ["CAGR", summary.CAGR, "ratio"],
  ] as const;
  const realizedItems = [
    ["거래횟수", realized?.trade_count, "count"],
    ["MOC 횟수", realized?.moc_count, "count"],
    ["평균 보유일", realized?.avg_hold_days, "days"],
    ["이익금", realized?.net_profit, "money", "lime"],
    ["평균 이익률", realized?.avg_gain_pct, "percent"],
    ["평균 손해률", realized?.avg_loss_pct, "percent", "negative"],
    ["평균 실현이익", realized?.avg_gain, "money"],
    ["평균 실현손해", realized?.avg_loss, "money", "negative"],
  ] as const;

  return (
    <section className="summary-panel">
      <header className="panel-header"><span className="eyebrow">요약 지표 // SUMMARY</span><small>{target} · {period}</small></header>
      <Metrics items={performance} />
      <div className="subsection-label">실현 지표 // REALIZED</div>
      <Metrics items={realizedItems} />
    </section>
  );
}
