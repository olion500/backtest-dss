import type { BacktestRequest } from "../types";

export function RunToolbar({
  request,
  currentMode,
  lastRun,
  loading,
  onChange,
  onRun,
}: {
  request: BacktestRequest;
  currentMode: string;
  lastRun?: Date;
  loading: boolean;
  onChange: (request: BacktestRequest) => void;
  onRun: () => void;
}) {
  return (
    <div className="run-toolbar">
      <label className="toolbar-period">
        <span>기간 // PERIOD</span>
        <input type="date" value={request.start_date} onChange={(event) => onChange({ ...request, start_date: event.target.value })} />
        <b>→</b>
        <input type="date" value={request.end_date} onChange={(event) => onChange({ ...request, end_date: event.target.value })} />
      </label>
      <div className="toolbar-chip"><span>MODE</span><strong className={currentMode === "공세" ? "lime" : ""}>{currentMode}</strong></div>
      <div className="toolbar-chip"><span>프리셋</span><strong>browser_local.json</strong></div>
      <div className="toolbar-actions">
        <span>{lastRun ? `마지막 실행 ${lastRun.toLocaleString("ko-KR")}` : "아직 실행하지 않음"}</span>
        <button className="button primary" type="button" onClick={onRun} disabled={loading}>{loading ? "계산 중..." : "백테스트 실행"}</button>
      </div>
    </div>
  );
}
