# Repository Guidelines

## Project Structure & Module Organization
Core logic lives in `optimize_params.py` at the repository root, handling config parsing, payload assembly, and backtest calls. JSON search spaces reside in `configs/`, with `configs/sample_search_space.json` as the reference template. Tooling sits alongside the source: the `Dockerfile` and `Makefile` provide repeatable execution, while `requirements.txt` pins runtime deps. Keep future Python modules near the root (or group them in a small package) so imports stay flat, and store generated data under a new, clearly named directory added to `.gitignore`.

## Build, Test, and Development Commands
`make build` creates the `backtest-dss:latest` image. `make run CONFIG=path/to.json RUN_ARGS="--top-k 10"` executes the optimiser in Docker with overrides. `make dry-run` walks the parameter grid without network calls, and `make shell` launches an interactive container session. Outside Docker, install deps once with `python -m pip install -r requirements.txt`, then run `python optimize_params.py --config configs/sample_search_space.json` plus any extra flags.

## Coding Style & Naming Conventions
Follow PEP 8, four-space indentation, and `snake_case` identifiers. Preserve existing type hints and docstrings, and keep helper functions pure and side-effect free where practical. User messaging should stay concise via `print`. Mirror upstream API field names exactly in JSON payloads, and prefer lowercase filenames without spaces.

## Testing Guidelines
We do not yet ship automated tests, so validate changes with deterministic dry runs: `make dry-run` confirms combination expansion and payload shape without touching the endpoint. When adding non-trivial logic, introduce `pytest`-based tests under `tests/` (create it if absent) and expose them through a `make test` target so they become part of the standard workflow. Document manual verification steps in each PR until the automated suite matures.

## Commit & Pull Request Guidelines
Commit subjects should be short, imperative, and consistent with the current history (`feat: add params optimizer`, `Add Docker tooling and Makefile commands`). Group related edits together and explain breaking changes or config migrations in the body. Pull requests must summarise the intent, call out new configs or CLI flags, and include dry-run or execution snippets that prove the behaviour. Reference linked issues and note expected request volume when altering network usage.

## API & Configuration Notes
The optimiser targets `https://insangai.com/new_backtest` with browser-like headers. Keep secrets out of JSON files, throttle large grids via `sleep_seconds` or `--sleep`, and validate configs with `python -m json.tool configs/<file>.json` before running to avoid runtime parsing failures.
