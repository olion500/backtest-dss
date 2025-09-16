# backtest-dss

CLI tooling for probing the Algori-C backtest endpoint and a lightweight local
simulator for iterating on strategy ideas without touching the network. The
optimiser walks through shared safe/aggressive parameter grids, mirrors the
browser headers, and grades each result with `score = cagr_pct -
drawdown_weight × max_drawdown_pct` so high-growth ideas with smaller drawdowns
rise to the top.

## Quick start (Docker)

```bash
make help         # discover available targets
make run          # run optimise_params.py with configs/sample_search_space.json
make dry-run      # print the payloads only
make shell        # open an interactive shell inside the container
make test         # run pytest inside Docker
make backtest     # evaluate configs with the local simulator
make fetch-prices # download OHLCV data for a ticker via Yahoo Finance
make exec         # run an arbitrary one-off command inside the container
```

- Pass another config with `CONFIG=configs/focused_search_space.json`.
- Forward extra CLI flags via `RUN_ARGS="--top-k 10 --drawdown-weight 1.5"`.
- All commands mount the repo inside the container, so results land in the
  workspace.

## Configuration model

JSON configs live in `configs/`. Each file specifies:

- `base_payload`: constant fields copied into every request.
- `shared_divisions`: optional list of division counts. Safe and aggressive
  parameter arrays always reuse the same division value from this list.
- `safe_params` / `aggressive_params`: ranges for `max_hold_days`,
  `buy_threshold_pct`, and `sell_threshold_pct`. When `shared_divisions` is
  absent, add a `divisions` array inside each block instead.
- `sleep_seconds`: optional delay to throttle API traffic.

The grids are expanded through a Cartesian product. Keep the lists compact so
requests stay manageable.

## CLI reference

```bash
python optimize_params.py --config configs/sample_search_space.json \
  --top-k 5 --drawdown-weight 1.2 --output runs/latest.jsonl
```

Common flags:

- `--dry-run`: emit payloads without posting.
- `--drawdown-weight`: scale the drawdown penalty in the score calculation.
- `--include-response`: persist raw JSON replies (including max drawdown,
  realized profit, etc.).
- `--sleep`: override the config-level delay.
- `--max-errors`: abort after N consecutive failures.

Install dependencies for local execution with `python -m pip install -r
requirements.txt` or rely on Docker via the `make` targets above.

## Local backtesting workflow

The repository ships a deterministic price sample under `data/sample_prices.csv`
and a pure-Python simulator (`backtest_engine.py`). Run the same optimiser
configs locally to gauge relative performance without sending HTTP requests:

```bash
make backtest BACKTEST_ARGS="--top-k 5 --drawdown-weight 1.0"
```

- Horizons, capital, and fees are pulled from `base_payload` just like the
  remote optimiser.
- `--prices` accepts any CSV with a `close` column if you want to swap in other
  data.
- Pass `BACKTEST_ARGS="--output results/local.jsonl"` to persist every evaluated combination
  (ignored by git via `.gitignore`).
- To explore the RSI-gated strategy, point the config to `configs/rsi_strategy_search_space.json`.

## RSI mode rules

`configs/rsi_strategy_search_space.json` demonstrates how to shape trading
regimes with Wilder's 14-period RSI:

- Safe entries require the RSI to be falling from extreme or mid-zone levels
  (above 65, inside 40–50, or crossing below 50) before applying the usual
  `buy_threshold_pct` drop and `sell_threshold_pct` exit.
- Aggressive entries demand a rising RSI that either bounces from <35, lifts
  through the 50–60 band, or crosses back above 50.
- The optimiser still sweeps the familiar `divisions`/threshold grids while the
  entry gates decide which leg is allowed to add exposure on any given day.

## Price data downloads

Use the bundled Yahoo Finance helper to source OHLCV history before running
local simulations:

```bash
make fetch-prices TICKER=SOXL START=2023-01-01 END=2024-01-01
```

- Files land under `market_data/` by default and stay out of version control.
- Override `FETCH_ARGS="--output data/custom.csv"` for a custom destination or
  `INTERVAL=1wk` to switch aggregation.

## Latest focused run (shared divisions 5 & 7)

```
Top 5 combinations (score=cagr - drawdown×1.0):
  #1: score=86.3000 CAGR=126.5200% drawdown=40.2200% safe=[5, 30, 3.0, 0.3] aggressive=[5, 7, 5.5, 2.5]
  #2: score=85.1700 CAGR=126.1300% drawdown=40.9600% safe=[5, 30, 3.0, 0.3] aggressive=[5, 7, 5.0, 2.5]
  #3: score=80.6500 CAGR=121.1100% drawdown=40.4600% safe=[5, 30, 2.5, 0.3] aggressive=[5, 7, 5.5, 2.5]
  #4: score=80.0400 CAGR=120.5500% drawdown=40.5100% safe=[5, 30, 3.0, 0.2] aggressive=[5, 7, 5.5, 2.5]
  #5: score=79.6900 CAGR=120.7600% drawdown=41.0700% safe=[5, 30, 3.0, 0.3] aggressive=[5, 7, 5.5, 2.0]
```

Use `configs/focused_search_space.json` to reproduce this 128-combination sweep
for SOXL vs QQQ between 2024-01-01 and 2025-09-01. Tweak `--drawdown-weight` to
shift the balance between CAGR and max drawdown, or extend `shared_divisions`
when you want to test additional capital slices.

## Output management

- `--output results/run.jsonl` appends one JSON object per evaluated pair for
  both the remote optimiser and the local simulator.
- Add `--include-response` to the remote optimiser when you need the entire
  payload (daily equity curve, sortino, etc.).
- Generated artefacts should live under a dedicated directory (e.g. `results/`)
  and are ignored by git.

## Networking

The optimiser targets `https://insangai.com/new_backtest` with the same headers
as the hosted UI. Avoid sharing secrets inside configs and respect rate limits
by spacing large grids with `sleep_seconds` or the `--sleep` flag.
