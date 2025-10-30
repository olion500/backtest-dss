# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dongpa is a Streamlit-based backtesting application for the "동파법" (Dongpa Method) trading strategy. The strategy applies **daily N-tranche LOC (Limit-On-Close) purchases** to leveraged ETFs with **weekly RSI-based mode switching** between safe and aggressive modes.

**Key Characteristics:**
- All orders execute at **LOC (Limit-On-Close)** - no intraday execution
- Daily tranche budget system: available cash divided into N equal parts
- Mode switching driven by weekly RSI (14, Wilder) momentum from QQQ
- Capital refresh cycles with configurable profit/loss contribution rates (PCR/LCR)
- Integer share enforcement with monetary precision to 2 decimal places

## Core Architecture

### Main Modules

**`dongpa_engine.py`** - Backtesting engine and core logic
- `DongpaBacktester` class: Main backtest orchestrator
- `ModeParams`: Mode-specific parameters (buy_cond_pct, tp_pct, max_hold_days, slices, stop_loss_pct)
- `CapitalParams`: Capital management (initial_cash, refresh_cycle_days, PCR, LCR)
- `StrategyParams`: Complete strategy configuration
- Signal helpers: `wilder_rsi()`, `to_weekly_close()`, `cross_up()`, `cross_down()`
- Monetary precision helpers: `money()`, `to_decimal()` for 2-decimal rounding

**`backtest.py`** - Main Streamlit dashboard
- User interface for configuring strategy parameters
- Displays equity curves, trade journals, and summary metrics
- Handles data download from Yahoo Finance via yfinance
- Computes Buy & Hold comparison and trade statistics
- Saves results to `outputs/` directory (gitignored)

**`dongpa_optimizer.py`** - Parameter grid search optimizer
- Runs multi-period backtests across parameter combinations
- Uses 2022+2025 as training set, 2023-2024 as test set
- Scores combinations by: avg_CAGR - (avg_MDD × weight)
- Outputs ranked results to `strategy_performance.md`

**`pages/1_Optimizer.py`** - Streamlit page for running optimizer
- UI wrapper around `dongpa_optimizer.py`
- Configurable grid search parameters
- Visual result display with Altair charts

**`pages/2_orderBook.py`** - LOC order schedule generator
- Real-time order planning based on latest market data
- Persists settings to `config/order_book_settings.json`
- Maintains order history in `config/order_book_history.csv`
- Displays next-day LOC order schedule

### Data Flow

1. **Input**: User specifies tickers (target, momentum, benchmark) and date range
2. **Download**: yfinance fetches daily OHLCV data, saved to `outputs/`
3. **Mode Decision**: Weekly RSI calculated from momentum ticker (QQQ), mode determined by RSI delta and thresholds
4. **Execution Logic**:
   - Daily LOC buy if `close <= prev_close * (1 + buy_cond_pct)`
   - LOC sell if `close >= TP` or max_hold_days exceeded or stop_loss triggered
   - Tranche budget = cash / N slices
5. **Capital Refresh**: Every `refresh_cycle_days`, adjust cash by realized P&L × PCR/LCR
6. **Output**: Trade journal (Korean columns), equity curve, metrics

## Development Commands

### Local Development
```bash
make install          # Install Python dependencies (requirements.txt)
make run-local        # Run Streamlit on localhost:8501 (no Docker)
streamlit run backtest.py --server.address=0.0.0.0 --server.port=8501
```

### Docker Workflows
```bash
make build           # Build production image (dongpa:latest)
make run             # Run containerized app on port 8501
make shell           # Open bash shell inside container
make build-dev       # Build dev image with live-reload (dongpa-dev:latest)
make dev             # Run dev container with source mounted
```

### Testing & Optimization
```bash
python -m pytest                    # Run test suite (if tests/ exists)
make optimize                       # Run parameter optimizer in Docker
python dongpa_optimizer.py          # Run optimizer locally
```

### Port Configuration
Override default port 8501:
```bash
PORT=9000 make run-local
PORT=9000 make run
```

## Strategy Logic (동파법)

### Mode Switching Rules
**Weekly RSI (Wilder, 14) on QQQ determines mode:**

**Safe Mode** (defensive) - triggered when ΔRSI < 0 AND one of:
- RSI ≥ 65 and declining
- 40 < RSI < 50 and declining
- 50-line downward cross

**Aggressive Mode** (offensive) - triggered when ΔRSI > 0 AND one of:
- 50-line upward cross
- 50 < RSI < 60 and rising
- RSI < 35 and rising

**Default**: Maintain previous mode if conditions ambiguous

### LOC Order Execution Rules

**Buy LOC**:
- Limit price = `prev_close × (1 + buy_cond_pct/100)`
- If `today_close ≤ limit_price`, fill at `today_close` for one tranche budget
- Maximum 1 buy per day
- Shares must be integer quantity
- Each tranche sets TP = `fill_price × (1 + tp_pct/100)`

**Sell LOC**:
- If `today_close ≥ TP`: sell at `today_close`
- If days_held ≥ max_hold_days: forced close at `today_close`
- If stop_loss enabled and `today_close ≤ fill_price × (1 - stop_loss_pct/100)`: sell at `today_close`

### Capital Management

**Tranche Budget**: `available_cash / N_slices`
- Recalculated on mode change if `reset_on_mode_change` enabled
- Recalculated at capital refresh cycle

**Capital Refresh Cycle** (every `refresh_cycle_days`):
```python
realized_pnl = sum(closed_trades_in_cycle)
if realized_pnl > 0:
    cash += realized_pnl * PCR
else:
    cash += realized_pnl * LCR  # LCR typically 0.3 (30% of loss)
```

Recommended defaults:
- PCR = 0.8 (80% of profits reinvested)
- LCR = 0.3 (30% of losses deducted)
- refresh_cycle_days = 10

### Recommended Parameter Ranges

| Mode | N_slices | buy_cond_pct | tp_pct | max_hold_days | stop_loss_pct |
|------|----------|--------------|--------|---------------|---------------|
| Safe | 7 | 3.0 | 0.2 | 30 | 0 (disabled) |
| Aggressive | 7 | 5.0 | 2.5 | 7 | 0 (disabled) |

## Code Style & Conventions

- **Language**: Python 3.x with type hints
- **Style**: PEP 8, 4-space indentation, snake_case naming
- **Dataclasses**: Use for configuration objects (ModeParams, CapitalParams, etc.)
- **Korean column names**: Trade journal uses Korean headers (매수일, 매도일, 실현손익, etc.)
- **Monetary precision**: Use `money()` and `to_decimal()` for all trade calculations (2 decimal places)
- **Integer shares**: All share quantities must be integers via `math.floor()`

## File Organization

```
backtest-dss/
├── backtest.py              # Main Streamlit dashboard
├── dongpa_engine.py         # Core backtest engine
├── dongpa_optimizer.py      # Parameter grid search
├── pages/
│   ├── 1_Optimizer.py      # Optimizer UI
│   └── 2_orderBook.py      # LOC order scheduler
├── config/
│   ├── order_book_settings.json    # Persisted order book config
│   └── order_book_history.csv      # Order history log
├── outputs/                 # Downloaded price data & results (gitignored)
├── requirements.txt         # Python dependencies
├── Makefile                # Build & run helpers
├── Dockerfile              # Production container
├── Dockerfile.dev          # Dev container with mount
├── dongpa_strategy.md      # Strategy documentation (Korean)
├── dongpa_visualize.md     # UI layout guide
├── AGENTS.md               # Repository guidelines
└── strategy_performance.md # Optimizer output (generated)
```

## Important Notes

### Data Handling
- Yahoo Finance rate limits apply - avoid excessive downloads
- Downloaded data cached in `outputs/` - delete to regenerate
- All pricing data is daily OHLCV (no intraday)
- Weekly resampling uses Friday close (`'W-FRI'`)

### Testing Strategy Changes
1. Modify parameters in UI or code
2. Run backtest on historical period (2020-2025 typical)
3. Compare equity curve, CAGR, MDD, win rate
4. For systematic testing, use optimizer with custom grid

### Known Constraints
- **No slippage modeling**: Engine assumes exact LOC fills at close price
- **Integer shares only**: May leave small cash remainder unused
- **Leveraged ETF decay**: Long holding periods (>30 days) risk volatility drag
- **Single position per tranche**: No pyramiding within a tranche

### Extending the Engine
- **New indicators**: Add to `dongpa_engine.py` indicator section
- **Alternative modes**: Modify `decide_mode()` logic in backtester
- **Custom exits**: Extend sell logic in main backtest loop
- **Slippage**: Implement via `slippage_pct` parameter (currently unused)

## Documentation References

- **dongpa_strategy.md**: Complete strategy rules, mode logic, parameter definitions (Korean)
- **dongpa_visualize.md**: UI layout, metric descriptions, output column definitions
- **AGENTS.md**: Repository conventions, commit style, testing guidelines
- **strategy_performance.md**: Latest optimizer results (auto-generated)
