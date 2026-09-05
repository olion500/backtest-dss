export type ViewName = "backtest" | "order-book" | "settings";
export type RunView = Exclude<ViewName, "settings">;
export type ModeSwitchStrategy = "rsi" | "ma_cross" | "roc" | "btc_overnight";

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
  stop_loss_orders: TablePayload;
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
