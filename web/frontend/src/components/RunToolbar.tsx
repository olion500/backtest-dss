import type { BacktestRequest, Preset } from "../types";
import { matchPreset } from "../presets";

export function RunToolbar({
  request,
  presets,
  currentMode,
  lastRun,
  loading,
  onChange,
  onRun,
}: {
  request: BacktestRequest;
  presets: Preset[];
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
      <PresetSelect request={request} presets={presets} onChange={onChange} />
      <div className="toolbar-actions">
        <span>{lastRun ? `마지막 실행 ${lastRun.toLocaleString("ko-KR")}` : "아직 실행하지 않음"}</span>
        <button className="button primary" type="button" onClick={onRun} disabled={loading}>{loading ? "계산 중..." : "백테스트 실행"}</button>
      </div>
    </div>
  );
}

const CUSTOM = "__custom__";

export function PresetSelect({ request, presets, onChange }: { request: BacktestRequest; presets: Preset[]; onChange: (request: BacktestRequest) => void }) {
  const active = matchPreset(presets, request.strategy);
  return (
    <label className="toolbar-chip toolbar-preset">
      <span>프리셋</span>
      <select
        value={active?.id ?? CUSTOM}
        onChange={(event) => {
          const preset = presets.find((item) => item.id === event.target.value);
          if (preset) onChange({ ...request, strategy: preset.strategy });
        }}
      >
        {!active && <option value={CUSTOM}>사용자 설정</option>}
        {presets.map((preset) => <option key={preset.id} value={preset.id}>{preset.name}</option>)}
      </select>
    </label>
  );
}
