# Repository Guidelines

## Project Structure
- `engines/` holds the pure business logic (backtest, order book, VR, config reading). Keep it free of web code.
- `web/api/` (FastAPI) calls the engines and serves `web/frontend/` (React + TypeScript, Vite).
- `benchmarks/` runs the real engine for strategy research. One suite per file in `benchmarks/suites/`; see `benchmarks/README.md`.
- `tests/` holds pytest files named `test_*.py`.

## Build, Test, and Development Commands
- `make install` / `make web-install`: install Python (uv) and frontend dependencies.
- `make web-api` + `make web-frontend`: run the API on :8000 and the Vite dev server on :5173.
- `make web`: build the production image and run it on :8000. `make deploy`: Cloud Run.
- `make test`: run pytest. `uv run --group bench python -m benchmarks`: list benchmark suites.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation, `snake_case` names, and the existing type hints/docstrings.
- Keep helpers pure where possible and reserve brief comments for non-obvious math or data handling.
- Use `money()` / `to_decimal()` for trade amounts and integer shares; keep the Korean journal column names.

## Testing Guidelines
- Use `pytest` with deterministic fixtures (`tests/conftest.py`, `_make_price_df`); expected values come from the documented rules, not from current output.
- After engine changes, also run `python -m benchmarks verify_harness` and `verify_trend_vix`.

## Commit & Pull Request Guidelines
- Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:`) with imperative subjects under 72 characters.
- Explain behavioural changes in the body. Attach screenshots or metrics when the UI changes.

## Security & Configuration Tips
- Keep credentials out of source; deploy settings live in `.env` (see `.env.example`).
- Account files (`config/accounts/*.json`, `config/vr/*.json`) are gitignored; commit only `*.sample`.
- Be mindful of Yahoo Finance rate limits.

## Documentation Map
- `docs/dongpa_strategy.md`: 전략 규칙, 모드 전환, 파라미터.
- `docs/loc_netting.md`, `docs/loc_spread_buy_strategy.md`: 퉁치기, 스프레드 매수.
- `docs/vr_sheet.md`: VR 구글 시트 규칙과 웹 VR 탭.
- `benchmarks/README.md`: 벤치마크 실행·추가 방법과 연구 결과.
