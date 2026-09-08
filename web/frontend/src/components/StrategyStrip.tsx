import type { BacktestRequest, OrderBookPayload } from "../types";

export function StrategyStrip({
  request,
  orderBook,
  onSettings,
}: {
  request: BacktestRequest;
  orderBook?: OrderBookPayload;
  onSettings: () => void;
}) {
  const mode = orderBook?.state.current_mode ?? "offense";
  const params = mode === "offense" ? request.strategy.offense : request.strategy.defense;
  const fields = [
    ["시작일", request.start_date],
    ["분할수", `${params.slices}`],
    ["매수조건", `${params.buy_cond_pct}%`],
    ["익절", `${params.tp_pct}%`],
    ["손절", params.stop_loss_pct ? `${params.stop_loss_pct}%` : "OFF"],
    ["최대보유", `${params.max_hold_days}일`],
    ["트랜치 예산", orderBook?.state.tranche_budget ? `$${orderBook.state.tranche_budget.toLocaleString("en-US", { maximumFractionDigits: 0 })}` : "—"],
    ["퉁치기", request.strategy.enable_netting ? "ON" : "OFF"],
  ];
  return (
    <div className="strategy-strip">
      <div className="strategy-mode"><span>현재 전략</span><strong>{mode === "offense" ? "공세" : "안전"}</strong></div>
      {fields.map(([label, value]) => <div className="strategy-cell" key={label}><span>{label}</span><strong>{value}</strong></div>)}
      <button type="button" className="strategy-edit" onClick={onSettings}>설정에서 편집 →</button>
    </div>
  );
}
