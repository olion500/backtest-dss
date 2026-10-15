export type ViewName = "backtest" | "order-book" | "vr" | "benchmark" | "settings";
export type RunView = Exclude<ViewName, "settings" | "vr" | "benchmark">;
export type ModeSwitchStrategy = "rsi" | "ma_cross" | "roc" | "btc_overnight" | "trend_vix";

export interface ModeSettings {
  slices: number;
  buy_cond_pct: number;
  tp_pct: number;
  stop_loss_pct: number;
  max_hold_days: number;
}

export interface StrategySettings {
  target_ticker: string;
  momentum_ticker: string;
  mode_switch_strategy: ModeSwitchStrategy;
  enable_netting: boolean;
  allow_fractional_shares: boolean;
  cash_limited_buy: boolean;
  rsi_high_threshold: number;
  rsi_mid_high: number;
  rsi_neutral: number;
  rsi_mid_low: number;
  rsi_low_threshold: number;
  ma_short_period: number;
  ma_long_period: number;
  roc_period: number;
  btc_ticker: string;
  btc_lookback_days: number;
  btc_threshold_pct: number;
  trend_sma_days: number;
  trend_band_pct: number;
  vix_ticker: string;
  vix_low: number;
  vix_high: number;
  defense: ModeSettings;
  offense: ModeSettings;
}

export interface BacktestRequest {
  start_date: string;
  end_date: string;
  initial_cash: number;
  log_scale: boolean;
  spread_buy_levels: number;
  spread_buy_step: number;
  strategy: StrategySettings;
}

/** config/strategy.json (live) or config/presets/<id>.json, from GET /api/v1/presets. */
export interface Preset {
  id: string;
  name: string;
  live: boolean;
  strategy: StrategySettings;
}

/** config/accounts/<id>.json — the order book runs the shared strategy with these per-account values. */
export interface Account {
  id: string;
  name: string;
  order: number | null;
  start_date: string;
  initial_cash: number;
  spread_buy_levels: number;
  spread_buy_step: number;
}

export type CellValue = string | number | boolean | null;

export interface TablePayload {
  columns: string[];
  rows: Record<string, CellValue>[];
}

export interface EquityPoint {
  date: string;
  equity: number | null;
  price: number | null;
}

export interface OrderBookPayload {
  state: {
    last_date: string;
    current_mode: "defense" | "offense";
    current_cash: number;
    current_position_qty: number;
    prev_close: number | null;
    tranche_budget: number | null;
    rsi_value: number | null;
  };
  holdings: TablePayload;
  orders: TablePayload;
  netting_message: string;
  netting_details: TablePayload;
  netting_scenarios: TablePayload;
}

export interface ViewerResult {
  meta: {
    target_ticker: string;
    momentum_ticker: string;
    start_date: string;
    end_date: string;
    log_scale: boolean;
  };
  summary: Record<string, number | null>;
  realized_metrics: Record<string, number | null> | null;
  state: {
    cash_end: number;
    open_positions: number;
  };
  equity: EquityPoint[];
  mode_bands: TablePayload;
  journal: TablePayload;
  trade_log: TablePayload;
  order_book?: OrderBookPayload;
}

/** config/vr/<id>.json — 밸류리밸런싱(VR) 계좌. 체결은 일별 가격으로 추정한다. */
export interface VRAccount {
  id: string;
  name: string;
  order: number | null;
  ticker: string;
  start_date: string;
  /** VR 시작일 (시트 시작일). 주차와 G 증가를 여기서부터 센다 */
  origin_date: string | null;
  start_shares: number;
  start_pool: number;
  start_v: number | null;
  principal: number | null;
  deposit: number;
  deposit_changes: Array<{ from: string; deposit: number }>;
  g: number;
  g_step_cycles: number;
  g_step: number;
  advanced: boolean;
  band_pct: number;
  pool_limit_pct: number;
  order_type: "limit" | "loc";
  buy_unit: number;
  sell_unit: number;
  sell_cap_pct: number;
  adjustments: Array<{ date: string; shares: number | null; pool: number | null; v: number | null }>;
}

export interface VRCycle {
  index: number;
  /** VR 시작일부터 지난 주 수 (시트의 주차) */
  week: number;
  start: string;
  end: string;
  /** 시트 기록(source: "sheet")은 상승률 표 방식이라 G가 없다 */
  g: number | null;
  v: number;
  v_min: number;
  v_max: number;
  deposit: number;
  shares_start: number;
  pool_start: number | null;
  bought: number;
  sold: number;
  shares: number;
  close: number | null;
  close_date: string | null;
  stock_value: number | null;
  pool: number;
  total: number | null;
  principal: number;
  return_pct: number | null;
  /** sheet: start_date 이전 구글 시트 기록 · web: 웹에서 계산한 사이클 */
  source: "sheet" | "web";
}

export interface VROrder {
  side: "buy" | "sell";
  price: number;
  qty: number;
  filled_on: string | null;
  fill_price: number | null;
}

export interface VROutlook {
  /** 사이클 며칠째 (달력 기준, 1~cycle_days) */
  day: number;
  cycle_days: number;
  /** 아직 안 걸린 매수 중 가장 높은 주문가 / 매도 중 가장 낮은 주문가 */
  buy_line: number | null;
  sell_line: number | null;
  buy_budget: number;
  buy_budget_left: number;
  next_start: string;
  next_deposit: number;
  next_g: number;
  /** 지금 종가로 계산한 다음 사이클 V */
  next_v: number | null;
}

export interface VRDay {
  date: string;
  close: number;
  shares: number;
  pool: number;
  stock_value: number;
  cycle: number;
  v: number;
  v_min: number;
  v_max: number;
  source: "sheet" | "web";
}

export interface VRResult {
  account: VRAccount;
  today: string;
  current: VRCycle;
  orders: VROrder[];
  outlook: VROutlook;
  daily: VRDay[];
  cycles: VRCycle[];
  fills: Array<{ cycle: number; date: string; side: "buy" | "sell"; qty: number; price: number }>;
}

/** benchmarks/results/summary.json (python -m benchmarks summary) */
export interface BenchmarkSummary {
  generated_at: string;
  data_end: string;
  strategies: Array<{ id: string; name: string; description: string }>;
  windows: Array<{
    id: string;
    label: string;
    description: string;
    start: string;
    end: string;
    total_return: boolean;
    dates: string[];
    curves: Record<string, number[]>;
    stats: Record<string, { cagr: number; mdd: number }>;
  }>;
}
