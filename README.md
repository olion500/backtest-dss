# backtest-dss

## Purpose

Utility scripts for experimenting with the hidden Algori-C backtest API.

## Optimising safe/aggressive parameters

`optimize_params.py` exhaustively iterates through candidate `safe_params` and
`aggressive_params` combinations and posts them to the production endpoint while
spoofing the headers sent by the official website. The script records the best
`cagr_pct` metric returned by the server.

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Configure the search space

Edit `configs/sample_search_space.json` (or create a copy) to describe the base
payload and the ranges for each of the four safe/aggressive parameters:

- `divisions`
- `max_hold_days`
- `buy_threshold_pct`
- `sell_threshold_pct`

Each list is combined via a Cartesian product, so keep the ranges tight to avoid
an explosion of requests. The file also supports an optional `sleep_seconds`
value that throttles requests.

### 3. Run the optimiser

```bash
python optimize_params.py --config configs/sample_search_space.json --output runs/soxl.jsonl
```

Key options:

- `--dry-run` prints the generated payloads without sending them.
- `--top-k` controls how many of the best combinations are displayed.
- `--include-response` appends the full JSON reply for each attempt to the output
  file (helpful when you need additional metrics beyond `cagr_pct`).
- `--sleep` overrides the default delay between requests.

All requests use the referer and user-agent from the public site so they blend
in with normal traffic.

> **Note:** Direct access to `https://insangai.com/new_backtest` from this
> environment currently fails because the upstream proxy blocks the TLS tunnel.
> Run the script from a network that can reach the endpoint.