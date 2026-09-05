import type { BacktestRequest, ModeSettings, StrategySettings } from "../types";

interface SettingsPanelProps {
  request: BacktestRequest;
  onChange: (request: BacktestRequest) => void;
  onReset: () => void;
  onApply: () => void;
}

function FieldRow({
  label,
  value,
  type = "number",
  step = 1,
  min,
  disabled = false,
  onChange,
}: {
  label: React.ReactNode;
  value: string | number;
  type?: "number" | "text" | "date";
  step?: number;
  min?: number;
  disabled?: boolean;
  onChange: (value: string | number) => void;
}) {
  return (
    <label className="setting-row">
      <span>{label}</span>
      <input
        type={type}
        value={value}
        step={type === "number" ? step : undefined}
        min={type === "number" ? min : undefined}
        disabled={disabled}
        onChange={(event) => onChange(type === "number" ? Number(event.target.value) : event.target.value)}
      />
    </label>
  );
}

function SettingCard({
  title,
  active = false,
  children,
}: {
  title: React.ReactNode;
  active?: boolean;
  children: React.ReactNode;
}) {
  return (
    <section className={`setting-card${active ? " active" : ""}`}>
      <div className="setting-card-title">{title}</div>
      <div className="setting-card-body">{children}</div>
    </section>
  );
}

function ModeCard({
  title,
  active,
  value,
  onChange,
}: {
  title: React.ReactNode;
  active?: boolean;
  value: ModeSettings;
  onChange: (value: ModeSettings) => void;
}) {
  const update = <K extends keyof ModeSettings>(key: K, next: ModeSettings[K]) => {
    onChange({ ...value, [key]: next });
  };
  return (
    <SettingCard title={title} active={active}>
      <FieldRow label="분할수" value={value.slices} min={1} onChange={(next) => update("slices", Number(next))} />
      <FieldRow label="매수조건 %" value={value.buy_cond_pct} step={0.1} onChange={(next) => update("buy_cond_pct", Number(next))} />
      <FieldRow label="익절 %" value={value.tp_pct} step={0.1} min={0} onChange={(next) => update("tp_pct", Number(next))} />
      <FieldRow label={<>손절 % <small>0=off</small></>} value={value.stop_loss_pct} step={0.1} min={0} onChange={(next) => update("stop_loss_pct", Number(next))} />
      <FieldRow label="최대 보유일" value={value.max_hold_days} min={1} onChange={(next) => update("max_hold_days", Number(next))} />
    </SettingCard>
  );
}

function ToggleRow({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="setting-toggle">
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

export function SettingsPanel({ request, onChange, onReset, onApply }: SettingsPanelProps) {
  const strategy = request.strategy;
  const updateRequest = <K extends keyof BacktestRequest>(key: K, value: BacktestRequest[K]) => {
    onChange({ ...request, [key]: value });
  };
  const updateStrategy = <K extends keyof StrategySettings>(key: K, value: StrategySettings[K]) => {
    onChange({ ...request, strategy: { ...strategy, [key]: value } });
  };

  return (
    <main className="settings-screen">
      <div className="settings-page">
        <header className="settings-titlebar">
          <div>
            <span className="eyebrow">// 설정 · STRATEGY SETTINGS</span>
            <h1>전략 설정 · 뷰어</h1>
          </div>
          <div className="settings-actions">
            <span className="local-status">브라우저 자동 저장</span>
            <button className="button ghost" type="button" onClick={onReset}>되돌리기</button>
            <button className="button primary" type="button" onClick={onApply}>설정 적용</button>
          </div>
        </header>

        <section className="local-settings-panel">
          <div className="local-settings-copy">
            <span className="eyebrow">개인 설정 // LOCAL · 서버 저장 없음</span>
            <p>현재 값은 이 브라우저에만 저장됩니다. 서버 설정 파일과 기존 Streamlit 설정은 변경하지 않습니다.</p>
          </div>
          <div className="local-settings-fields">
            <FieldRow label="백테스트 시작일" value={request.start_date} type="date" onChange={(next) => updateRequest("start_date", String(next))} />
            <FieldRow label="종료일" value={request.end_date} type="date" onChange={(next) => updateRequest("end_date", String(next))} />
            <FieldRow label="초기 가용현금 $" value={request.initial_cash} step={1000} min={1} onChange={(next) => updateRequest("initial_cash", Number(next))} />
          </div>
        </section>

        <section className="settings-card-grid">
          <SettingCard title="종목 // TICKERS">
            <FieldRow label="투자 종목" value={strategy.target_ticker} type="text" onChange={(next) => updateStrategy("target_ticker", String(next).toUpperCase())} />
            <FieldRow label="모멘텀 종목" value={strategy.momentum_ticker} type="text" onChange={(next) => updateStrategy("momentum_ticker", String(next).toUpperCase())} />
            <FieldRow label={<>벤치마크 <small>표시용</small></>} value="SOXX" type="text" disabled onChange={() => undefined} />
          </SettingCard>

          <SettingCard title="모드 전환 // MODE SWITCH">
            <label className="setting-select">
              <span>전환 전략</span>
              <select value={strategy.mode_switch_strategy} onChange={(event) => updateStrategy("mode_switch_strategy", event.target.value as StrategySettings["mode_switch_strategy"])}>
                <option value="rsi">RSI (주봉)</option>
                <option value="ma_cross">Golden Cross</option>
                <option value="roc">Rate of Change</option>
                <option value="btc_overnight">BTC Overnight</option>
              </select>
            </label>
            {strategy.mode_switch_strategy === "rsi" && (
              <>
                <FieldRow label={<>상한 <small>UPPER</small></>} value={strategy.rsi_high_threshold} step={0.1} onChange={(next) => updateStrategy("rsi_high_threshold", Number(next))} />
                <FieldRow label={<>중상 <small>MID-HI</small></>} value={strategy.rsi_mid_high} step={0.1} onChange={(next) => updateStrategy("rsi_mid_high", Number(next))} />
                <FieldRow label={<>중립 <small>NEUTRAL</small></>} value={strategy.rsi_neutral} step={0.1} onChange={(next) => updateStrategy("rsi_neutral", Number(next))} />
                <FieldRow label={<>중하 <small>MID-LO</small></>} value={strategy.rsi_mid_low} step={0.1} onChange={(next) => updateStrategy("rsi_mid_low", Number(next))} />
                <FieldRow label={<>하한 <small>LOWER</small></>} value={strategy.rsi_low_threshold} step={0.1} onChange={(next) => updateStrategy("rsi_low_threshold", Number(next))} />
              </>
            )}
            {strategy.mode_switch_strategy === "ma_cross" && (
              <>
                <FieldRow label="Short MA (주)" value={strategy.ma_short_period} min={1} onChange={(next) => updateStrategy("ma_short_period", Number(next))} />
                <FieldRow label="Long MA (주)" value={strategy.ma_long_period} min={2} onChange={(next) => updateStrategy("ma_long_period", Number(next))} />
              </>
            )}
            {strategy.mode_switch_strategy === "roc" && <FieldRow label="ROC 기간 (주)" value={strategy.roc_period} min={1} onChange={(next) => updateStrategy("roc_period", Number(next))} />}
            {strategy.mode_switch_strategy === "btc_overnight" && (
              <>
                <FieldRow label="BTC 티커" value={strategy.btc_ticker} type="text" onChange={(next) => updateStrategy("btc_ticker", String(next).toUpperCase())} />
                <FieldRow label="Lookback (일)" value={strategy.btc_lookback_days} min={1} onChange={(next) => updateStrategy("btc_lookback_days", Number(next))} />
                <FieldRow label="임계값 %" value={strategy.btc_threshold_pct} step={0.1} min={0} onChange={(next) => updateStrategy("btc_threshold_pct", Number(next))} />
              </>
            )}
          </SettingCard>

          <ModeCard title={<><i className="mode-dot safe" />안전 모드 // SAFE</>} value={strategy.defense} onChange={(value) => updateStrategy("defense", value)} />
          <ModeCard title={<><i className="mode-dot aggressive" />공세 모드 // AGGRESSIVE <b>ACTIVE</b></>} active value={strategy.offense} onChange={(value) => updateStrategy("offense", value)} />

          <SettingCard title="자본 · 옵션 // CAPITAL">
            <ToggleRow label="퉁치기 (동일 종가 상쇄)" checked={strategy.enable_netting} onChange={(value) => updateStrategy("enable_netting", value)} />
            <ToggleRow label="소수점 거래 허용" checked={strategy.allow_fractional_shares} onChange={(value) => updateStrategy("allow_fractional_shares", value)} />
            <ToggleRow label="현금 한도 매수" checked={strategy.cash_limited_buy} onChange={(value) => updateStrategy("cash_limited_buy", value)} />
            <ToggleRow label="Equity 로그 스케일" checked={request.log_scale} onChange={(value) => updateRequest("log_scale", value)} />
          </SettingCard>

          <SettingCard title="스프레드 매수 // SPREAD">
            <FieldRow label="레벨 수" value={request.spread_buy_levels} min={0} onChange={(next) => updateRequest("spread_buy_levels", Number(next))} />
            <FieldRow label="레벨당 수량" value={request.spread_buy_step} min={1} onChange={(next) => updateRequest("spread_buy_step", Number(next))} />
            <p className="setting-help">레벨 간격은 매수조건 구간을 등분합니다. 스프레드는 신규 트랜치 예산 안에서 체결됩니다.</p>
          </SettingCard>
        </section>
      </div>
    </main>
  );
}
