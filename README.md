# Dongpa Backtest App

Dongpa LOC backtesting engine with a local Streamlit workspace and a publishable read-only web viewer.

## Quick Start

- `make install` installs the Python dependencies locally.
- `make run-local` launches `streamlit run main.py` on port 8501.
- `make build && make run` builds the Docker image (`dongpa:latest`) and serves the app via Docker.
- `make build-dev && make dev` mounts the working tree into a dev container for live code reloads.

The app writes downloaded pricing data under `outputs/` (ignored by git). Delete files there if you want to free space or regenerate a scenario.

## Public Viewer

The React/FastAPI viewer keeps the existing Python engines as its single source of truth. It accepts strategy inputs and returns calculated results, but has no server-side settings or result write endpoints.

- `make web-install` installs frontend dependencies.
- `make web-api` starts FastAPI on `http://localhost:8000`.
- `make web-frontend` starts Vite on `http://localhost:5173`.
- `make web-build && make web-run` builds and serves the complete viewer from one container on `http://localhost:8000`.

The API exposes `/api/v1/config/defaults`, `/api/v1/backtests/run`, `/api/v1/order-book/preview`, and `/api/docs`. Market data is cached for 10 minutes. `DONGPA_RATE_LIMIT` controls the per-IP calculation limit per minute and defaults to `12`.

## Project Layout

- `pages/` — Streamlit UI for backtests, order books, and Optuna.
- `engines/` — reusable backtesting and order-book business logic.
- `web/api/` — FastAPI read-only service layer.
- `web/frontend/` — React and TypeScript viewer.
- `Dockerfile` / `Dockerfile.dev` — container definitions for production and mounted dev workflows.
- `Dockerfile.web` — multi-stage viewer image with the React build and API runtime.
- `Makefile` — helper targets for local installs, Docker builds, shells, and dev loops.
- `pyproject.toml` — dependency list and project metadata.
- `AGENTS.md` — repository conventions and guidelines.
- `docs/dongpa_strategy.md`, `docs/dongpa_visualize.md` — strategy notes and visual guides.

## Development Notes

- Use `make shell` after `make build` to drop into a bash shell inside the container.
- Streamlit serves on `http://localhost:8501`; override the host port with `PORT=xxxx make run-local`.
- The backtester enforces integer share sizes, LOC execution, and mode-dependent tranche management. Read `docs/dongpa_strategy.md` for the complete ruleset.
