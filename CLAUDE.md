# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dongpa is a Streamlit-based backtesting application for the "동파법" (Dongpa Method) trading strategy. The strategy applies **daily N-tranche LOC (Limit-On-Close) purchases** to leveraged ETFs with **weekly RSI-based mode switching** between safe and aggressive modes.

**Key Characteristics:**
- All orders execute at **LOC (Limit-On-Close)** - no intraday execution
- Daily tranche budget system: available cash divided into N equal parts
- Mode switching driven by weekly RSI (14, Wilder) momentum from QQQ
- Tranche budget system with automatic rebalancing on sell
- Integer share enforcement with monetary precision to 2 decimal places

## Core Architecture

### Main Modules

**`main.py`** - Streamlit entry point (landing page)

**`engines/dongpa_engine.py`** - Backtesting engine and core logic
- `run_backtest()`: Main backtest entry point (pure function)
- `ModeParams`: Mode-specific parameters (buy_cond_pct, tp_pct, max_hold_days, slices, stop_loss_pct)
- `CapitalParams`: Capital management (`initial_cash` only)
- `StrategyParams`: Complete strategy configuration
- Signal helpers: `wilder_rsi()`, `to_weekly_close()`, `cross_up()`, `cross_down()`
- Monetary precision helpers: `money()`, `to_decimal()` for 2-decimal rounding

**`engines/order_book_engine.py`** - Pure business logic for the LOC order book
- `extract_state()`, `build_holdings()`, `build_order_sheet()`, `apply_netting()`, `build_spread_orders()`

**`engines/vr_engine.py`** - Value rebalancing (라오어 VR 5.0, 실력공식), separate from 동파법
- `run_vr()`: 2-week cycles from a config + daily bars; buy/sell ladders placed per cycle, fills estimated from Low/High (limit) or Close (LOC)
- Rules mirror the user's VR Google Sheet (`docs/vr_sheet.md`); accounts in `config/vr/<id>.json`, web tab "VR"

**`engines/dongpa_optuna.py`** - Optuna optimizer engine
- `OptunaConfig`, `run_optuna()`, `extract_results()`, `narrow_config()`

**`ui/common.py`** - Shared UI utilities (navigation, settings I/O, strategy params builder)

**`ui/charts.py`** - Equity vs price chart building

**`pages/1_backtest.py`** - Main Streamlit backtest dashboard
- User interface for configuring strategy parameters
- Displays equity curves, trade journals, and summary metrics
- Handles data download from Yahoo Finance via yfinance

**`pages/2_order_book.py`** - LOC order schedule generator
- Real-time order planning based on latest market data
- Persists the strategy to `config/strategy.json` and the selected account to `config/accounts/<id>.json`
- Displays next-day LOC order schedule

**`pages/3_optuna.py`** - Optuna optimizer UI

### Data Flow

1. **Input**: User specifies tickers (target, momentum, benchmark) and date range
2. **Download**: yfinance fetches daily OHLCV data, saved to `outputs/`
3. **Mode Decision**: Weekly RSI calculated from momentum ticker (QQQ), mode determined by RSI delta and thresholds
4. **Execution Logic**:
   - Daily LOC buy if `close <= prev_close * (1 + buy_cond_pct)`
   - LOC sell if `close >= TP` or max_hold_days exceeded or stop_loss triggered
   - Tranche budget = cash / N slices
5. **Tranche Rebalance**: After sells, tranche base updated to reflect available cash
6. **Output**: Trade journal (Korean columns), equity curve, metrics

## Development Commands

### Local Development
```bash
make install          # Install Python dependencies (uv sync)
make run-local        # Run Streamlit on localhost:8501 (no Docker)
streamlit run main.py --server.address=0.0.0.0 --server.port=8501
```

### Docker Workflows
```bash
make build           # Build production image (dongpa:latest)
make run             # Run containerized app on port 8501
make shell           # Open bash shell inside container
make build-dev       # Build dev image with live-reload (dongpa-dev:latest)
make dev             # Run dev container with source mounted
```

### Testing
```bash
python -m pytest                    # Run test suite
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
- With `spread_buy_step` set (web API always sets it from the order book settings), the fill is
  the base qty `budget ÷ limit` plus `spread_buy_step` shares per step price the close reached
  (`spread_fill_qty`, no level cap — the order book's level count is display-only). `None` = legacy `budget ÷ close`

**Sell LOC**:
- If `today_close ≥ TP`: sell at `today_close`
- If days_held ≥ max_hold_days: forced close at `today_close`
- If stop_loss enabled and `today_close ≤ fill_price × (1 - stop_loss_pct/100)`: sell at `today_close`

### Capital Management

**Tranche Budget**: `available_cash / N_slices`
- Recalculated every day based on current tranche base and mode's N_slices
- Tranche base resets to current cash after every sell (no ratchet behavior)

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
├── main.py                  # Streamlit entry point (landing page)
├── engines/
│   ├── __init__.py
│   ├── dongpa_engine.py     # Core backtest engine
│   ├── order_book_engine.py # LOC order book logic
│   ├── vr_engine.py         # Value rebalancing (VR) cycles, ladders, estimated fills
│   ├── config_store.py      # config/ file I/O shared by web and Streamlit
│   └── dongpa_optuna.py     # Optuna optimizer engine
├── ui/
│   ├── __init__.py
│   ├── common.py            # Shared UI utilities
│   └── charts.py            # Chart building
├── pages/
│   ├── 1_backtest.py        # Backtest dashboard
│   ├── 2_order_book.py      # LOC order scheduler
│   └── 3_optuna.py          # Optuna optimizer UI
├── tests/                   # pytest suite
├── config/
│   ├── strategy.json        # Shared strategy (all accounts trade it)
│   ├── accounts/            # Per-account name/start_date/init_cash/spread (*.json gitignored, *.sample committed)
│   └── presets/             # Saved strategy candidates (backups, Optuna results)
├── docs/                    # Documentation
│   ├── dongpa_strategy.md   # Strategy rules (Korean)
│   └── dongpa_visualize.md  # UI layout guide
├── outputs/                 # Downloaded price data & results (gitignored)
├── pyproject.toml           # Dependencies & project metadata
├── Makefile                 # Build & run helpers
├── Dockerfile               # Production container
├── Dockerfile.dev           # Dev container with mount
├── CLAUDE.md                # Claude Code instructions
├── AGENTS.md                # Repository guidelines
└── strategy_performance.md  # Optimizer output (generated)
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
- **New indicators**: Add to `engines/dongpa_engine.py` indicator section
- **Alternative modes**: Modify `decide_mode()` logic in backtester
- **Custom exits**: Extend sell logic in main backtest loop
- **Slippage**: Not modeled; add to `CapitalParams` if needed

## Documentation References

- **docs/dongpa_strategy.md**: Complete strategy rules, mode logic, parameter definitions (Korean)
- **docs/vr_sheet.md**: VR Google Sheet rules and how the web VR tab maps them
- **docs/dongpa_visualize.md**: UI layout, metric descriptions, output column definitions
- **AGENTS.md**: Repository conventions, commit style, testing guidelines
- **strategy_performance.md**: Latest optimizer results (auto-generated)
