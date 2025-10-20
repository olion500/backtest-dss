# Repository Guidelines

## Project Structure & Module Organization
- `app_dongpa.py` drives the Streamlit UI plus user inputs; keep view-layer helpers beside it and move shared logic into `dongpa_engine.py`.
- `dongpa_engine.py` houses the backtesting engine, dataclasses, and signal helpers. Add future analytics as flat modules in the repo root and import directly.
- Tooling (`requirements.txt`, `Dockerfile`, `Makefile`) sits next to the source for quick builds. Store generated CSVs or notebooks under `outputs/` and ignore that directory in git.
- Put automated checks in `tests/` with filenames such as `tests/test_engine.py` so ownership remains obvious.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt`: install dependencies for local runs.
- `streamlit run app_dongpa.py`: launch the dashboard without Docker for fast iteration.
- `make build`: produce the `dongpa:latest` image; re-run after dependency or system changes.
- `make run`: serve the container on `http://localhost:8501`; use `make bash` for an interactive container shell and `make clean` to remove the image when done.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation, `snake_case` names, and the existing type hints/docstrings.
- Keep helpers pure where possible and reserve brief comments for non-obvious math or data handling.
- Match upstream field names in payloads and align widget labels with the Korean terminology already present.

## Testing Guidelines
- Use `pytest` for new unit coverage. Name files `test_*.py`, prefer deterministic fixtures, and run via `python -m pytest`.
- For manual verification, run `streamlit run app_dongpa.py` with representative tickers and note inputs/outputs.
- Document manual steps in PRs until the automated suite matures.

## Commit & Pull Request Guidelines
- Mirror the Conventional Commit style in history (`feat:`, `docs:`, `chore:`) with imperative subjects under 72 characters.
- Bundle related edits, explain behavioural shifts or migrations in the body, and reference linked issues.
- PRs should summarise intent, list key commands executed (e.g., `make run`), and attach screenshots or metrics when the UI changes.

## Security & Configuration Tips
- Keep credentials out of source and configs; rely on environment variables if secrets become necessary.
- Be mindful of Yahoo Finance rate limits and avoid committing large CSV exports or proprietary datasets.

## Documentation Map
- `dongpa_strategy.md`: 전략 로직과 파라미터, 위험관리 규칙을 정리한 참고 문서.
- `dongpa_visualize.md`: Streamlit 대시보드 구성, 요약 지표 배치, 출력 컬럼 정의 등 UI 관련 지침.
