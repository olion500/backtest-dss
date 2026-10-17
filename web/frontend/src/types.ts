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
  /** >0: ladder runs down to prev close × (1 − this %), replacing spread_buy_levels */
  spread_floor_pct: number;
  /** Broker costs, % of the filled amount: commission both ways, SEC fee on sells. */
  fee_pct: number;
  sec_fee_pct: number;
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
  /** >0: ladder runs down to prev close × (1 − this %), replacing spread_buy_levels */
  spread_floor_pct: number;
  fee_pct: number;
  sec_fee_pct: number;
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

export type TraderRunStatus = "ok" | "halted" | "closed" | "error";

/** One run record of `python -m trader` (var/trader), summarized. */
export interface TraderRun {
  step: string;
  session: string;
  status: TraderRunStatus;
  message: string;
  started_at: string;
  forced: boolean | null;
}

/** One line of the engine-ledger vs broker comparison. ok=null: shown for context, never blocks. */
export interface TraderCheck {
  key: string;
  label: string;
  expected: string;
  actual: string;
  diff: string;
  ok: boolean | null;
  note: string;
}

/** GET /api/v1/trader/status — the order book's auto-trading timeline. */
export interface TraderStatus {
  account: string;
  now: string;
  /** Any run record exists for this account (no records = manual account). */
  configured: boolean;
  auto_trade: boolean | null;
  broker: string | null;
  session: {
    date: string;
    is_today: boolean;
    close: string | null;
    early_close: boolean;
    stop_check_at: string | null;
    order_cutoff: string | null;
  };
  /** current: the check ran for this session (else it is the latest real comparison). */
  check: (TraderRun & { current: boolean; checks: TraderCheck[] }) | null;
  recent: TraderRun[];
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
  g: number;
  v: number;
  v_min: number;
  v_max: number;
  deposit: number;
  shares_start: number;
  pool_start: number;
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
  /** 그 사이클까지 넣은 돈 (시작 원금 + 적립) */
  principal: number;
  cycle: number;
  v: number;
  v_min: number;
  v_max: number;
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
  /** 이동평균 평단: 매도는 평단을 그대로 두고 수량만 줄인다 */
  holding: { cost_basis: number; avg_cost: number | null; unrealized: number | null };
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
