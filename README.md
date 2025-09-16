# backtest-dss

CLI tooling for probing the Algori-C backtest endpoint. The optimiser walks
through shared safe/aggressive parameter grids, mirrors the browser headers, and
now grades each result with `score = cagr_pct - drawdown_weight × max_drawdown_pct`
so high-growth ideas with smaller drawdowns rise to the top.

## Quick start (Docker)

```bash
make help         # discover available targets
make run          # run optimise_params.py with configs/sample_search_space.json
make dry-run      # print the payloads only
make shell        # open an interactive shell inside the container
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
requirements.txt`.

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

- `--output results/run.jsonl` appends one JSON object per evaluated pair.
- Add `--include-response` when you need the entire payload (daily equity curve,
  sortino, etc.).
- Generated artefacts should live under a dedicated directory (e.g. `runs/`) and
  be ignored by git.

## Networking

The optimiser targets `https://insangai.com/new_backtest` with the same headers
as the hosted UI. Avoid sharing secrets inside configs and respect rate limits
by spacing large grids with `sleep_seconds` or the `--sleep` flag.
