"""Benchmark results for the BENCHMARK tab: the file `python -m benchmarks summary` writes (read only)."""
from __future__ import annotations

import json
from pathlib import Path

SUMMARY_PATH = Path(__file__).resolve().parents[4] / "benchmarks" / "results" / "summary.json"


def load_benchmark_summary(path: Path = SUMMARY_PATH) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
