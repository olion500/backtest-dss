# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dongpa is a backtester and daily LOC order-book tool (React + FastAPI web app, `web/`) for the "동파법" (Dongpa Method) trading strategy. The strategy applies **daily N-tranche LOC (Limit-On-Close) purchases** to leveraged ETFs with **weekly RSI-based mode switching** between safe and aggressive modes.

**Key Characteristics:**
- All orders execute at **LOC (Limit-On-Close)** - no intraday execution
- Daily tranche budget system: available cash divided into N equal parts
- Mode switching driven by weekly RSI (14, Wilder) momentum from QQQ
- Tranche budget system with automatic rebalancing on sell
- Integer share enforcement with monetary precision to 2 decimal places

## Core Architecture

### Main Modules

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

**`engines/config_store.py`** - Reads `config/` (strategy, accounts, presets, VR accounts). The web never writes config.

**`web/api/`** - FastAPI service (backtest, order book, VR, presets); serves the built frontend
**`web/frontend/`** - React + TypeScript UI (Vite)

**`benchmarks/`** - Strategy research: runs the real engine with injected mode series (see `benchmarks/README.md`)
- `python -m benchmarks` lists suites; one file per suite in `benchmarks/suites/`, shared code in `benchmarks/lib/`
- After engine changes, run `verify_harness` and `verify_trend_vix` (needs `fetch` data first)

### Data Flow

1. **Input**: User specifies tickers (target, momentum, benchmark) and date range
2. **Data**: daily OHLCV from `data/*.csv` (committed; `scripts/update_market_data.py` appends new bars), live yfinance only as a fallback
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
make install          # Python dependencies (uv sync)
make web-install      # Frontend dependencies
make web-api          # FastAPI on :8000 (reload)
make web-frontend     # Vite dev server on :5173
make web              # Build the image and run the app on :8000
make deploy           # Cloud Run (needs .env, see .env.example)
```

### Testing
```bash
python -m pytest                    # Run test suite
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

**Timing**: A weekly bar includes its Friday close, so a new mode applies from the next session
(usually Monday). The journal's `다음모드` column is the next session's mode; the order book trades it.
`btc_overnight` is the exception (BTC's prior-day close is known before the US close).

**Trend + VIX (`trend_vix`, alternative)**: daily signal on the momentum ticker's SMA band and VIX
(`^VIX`, passed as `vix_data`). Momentum above `SMA × (1 + band)` and VIX below `vix_low` → **defense**;
momentum below `SMA × (1 − band)` or VIX above `vix_high` → **offense**; inside either band the
previous state holds. The labels are deliberately inverted vs RSI: "offense" buys weakness with few,
large tranches. Preset: `config/presets/trend_vix_2026-10-12.json`. Research and validation: `benchmarks/README.md`.

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
├── engines/
│   ├── dongpa_engine.py     # Core backtest engine
│   ├── order_book_engine.py # LOC order book logic
│   ├── vr_engine.py         # Value rebalancing (VR) cycles, ladders, estimated fills
│   └── config_store.py      # Reads config/
├── web/
│   ├── api/                 # FastAPI app (app/main.py, app/services/)
│   └── frontend/            # React + TypeScript (Vite)
├── benchmarks/              # Strategy research suites (python -m benchmarks)
│   ├── lib/                 # data, engine injection, windows, signals, Optuna search
│   ├── suites/              # one benchmark per file
│   └── results/             # saved summaries (Optuna *.log gitignored)
├── scripts/update_market_data.py  # Appends new daily bars to data/
├── data/                    # Committed daily OHLCV (data/tickers.txt)
├── tests/                   # pytest suite
├── config/
│   ├── strategy.json        # Shared strategy (all accounts trade it)
│   ├── accounts/            # Per-account name/start_date/init_cash/spread (*.json gitignored, *.sample committed)
│   ├── vr/                  # VR accounts (*.json gitignored)
│   └── presets/             # Saved strategy candidates
├── docs/                    # Strategy, netting, spread, VR sheet notes (Korean)
├── Dockerfile.web           # Production image (frontend build + API)
├── Makefile
└── pyproject.toml           # uv; groups: dev (pytest), bench (optuna)
```

## Important Notes

### Data Handling
- Yahoo Finance rate limits apply - avoid excessive downloads
- All pricing data is daily OHLCV (no intraday)
- Weekly resampling uses Friday close (`'W-FRI'`)

### Testing Strategy Changes
1. Modify parameters in the web settings panel or code
2. Run backtest on historical period (2021~ typical)
3. Compare equity curve, CAGR, MDD, win rate
4. For systematic testing, add a suite under `benchmarks/suites/`

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
- **benchmarks/README.md**: How to run/add benchmarks, research results
- **AGENTS.md**: Repository conventions, commit style, testing guidelines
