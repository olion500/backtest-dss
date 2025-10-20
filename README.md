# Dongpa Backtest App

Streamlit front-end for the Dongpa LOC backtest engine. The app lets you load target and momentum tickers from Yahoo Finance, configure safe and aggressive purchase modes, and inspect equity curves, trade journals, and CSV exports without touching the legacy optimiser tooling.

## Quick Start

- `make install` installs the Python dependencies locally.
- `make run-local` launches `streamlit run app_dongpa.py` on port 8501.
- `make build && make run` builds the Docker image (`dongpa:latest`) and serves the app via Docker.
- `make build-dev && make dev` mounts the working tree into a dev container for live code reloads.

The app writes downloaded pricing data under `outputs/` (ignored by git). Delete files there if you want to free space or regenerate a scenario.

## Project Layout

- `app_dongpa.py` — Streamlit UI for configuring the strategy and rendering metrics.
- `dongpa_engine.py` — vectorised LOC backtest core with weekly RSI mode switching.
- `Dockerfile` / `Dockerfile.dev` — container definitions for production and mounted dev workflows.
- `Makefile` — helper targets for local installs, Docker builds, shells, and dev loops.
- `requirements.txt` — minimal dependency list (Streamlit, pandas, numpy, yfinance).
- `AGENTS.md`, `dongpa_strategy.md`, `dongpa_visualize.md` — strategy notes and visual guides.

## Development Notes

- Use `make shell` after `make build` to drop into a bash shell inside the container.
- Streamlit serves on `http://localhost:8501`; override the host port with `PORT=xxxx make run-local`.
- The backtester enforces integer share sizes, LOC execution, and mode-dependent tranche management. Read `dongpa_strategy.md` for the complete ruleset.

