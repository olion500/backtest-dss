#!/usr/bin/env python3
"""Run local backtests over the safe/aggressive parameter grid."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence

from backtest_engine import BacktestEngine, BacktestSettings, StrategyParameters

Number = float | int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate strategies using the local backtest engine.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sample_search_space.json"),
        help="Path to the optimiser-style JSON configuration.",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("data/sample_prices.csv"),
        help="CSV containing a 'close' column used as the price series.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of best-scoring combinations to print.",
    )
    parser.add_argument(
        "--drawdown-weight",
        type=float,
        default=1.0,
        help="Penalty multiplier applied to max drawdown when ranking results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSONL file that will receive every evaluated combination.",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file '{path}' does not exist")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_price_series(path: Path) -> List[float]:
    if not path.exists():
        raise FileNotFoundError(f"Price file '{path}' does not exist")
    closes: List[float] = []
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        try:
            close_index = header.index("close")
        except ValueError as exc:
            raise ValueError("Expected a 'close' column in the price CSV") from exc
        for line in fh:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            try:
                closes.append(float(parts[close_index]))
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Could not parse close price from line: {line.strip()}") from exc
    if len(closes) < 2:
        raise ValueError("Price series must contain at least two rows")
    return closes


def expand_param_grid(
    grid: MutableMapping[str, Iterable[Any]],
    *,
    kind: str,
    divisions: Optional[Sequence[Number]] = None,
) -> List[List[Number]]:
    param_order = ("divisions", "max_hold_days", "buy_threshold_pct", "sell_threshold_pct")
    sequences: List[List[Number]] = []
    for field in param_order:
        if field == "divisions" and divisions is not None:
            sequences.append(_ensure_numeric_sequence(divisions, f"{kind}.{field}"))
            continue
        if field not in grid:
            raise KeyError(f"Missing '{field}' inside {kind}_params")
        sequences.append(_ensure_numeric_sequence(grid[field], f"{kind}.{field}"))
    combinations: List[List[Number]] = []
    for values in _product(*sequences):
        combinations.append(list(values))
    return combinations


def _product(*sequences: Iterable[Number]) -> Iterable[Sequence[Number]]:
    if not sequences:
        return [[]]
    import itertools

    return itertools.product(*sequences)


def _ensure_numeric_sequence(values: Iterable[Any], label: str) -> List[Number]:
    result: List[Number] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"'{label}' entries must be numeric, got {item!r}")
        if isinstance(item, int):
            result.append(int(item))
        else:
            result.append(float(item))
    if not result:
        raise ValueError(f"'{label}' must not be empty")
    return result


def _group_by_division(combos: Sequence[Sequence[Number]]) -> Dict[int, List[Sequence[Number]]]:
    grouped: Dict[int, List[Sequence[Number]]] = {}
    for combo in combos:
        division = int(combo[0])
        grouped.setdefault(division, []).append(combo)
    return grouped


def _as_strategy_params(values: Sequence[Number]) -> StrategyParameters:
    return StrategyParameters(
        divisions=int(values[0]),
        max_hold_days=int(values[1]),
        buy_threshold_pct=float(values[2]),
        sell_threshold_pct=float(values[3]),
    )


def _calculate_score(cagr: float, drawdown: float, weight: float) -> float:
    return cagr - (drawdown * weight)


def _ensure_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    base_payload = config.get("base_payload", {})
    settings = BacktestSettings(
        initial_capital=float(base_payload.get("initial_capital", 100000.0)),
        commission_rate=float(base_payload.get("commission_rate", 0.0)),
    )
    prices = read_price_series(args.prices)
    engine = BacktestEngine(prices)

    shared_divisions = config.get("shared_divisions")
    if not shared_divisions:
        raise ValueError("Config must define 'shared_divisions' for local backtests")

    safe_combos = expand_param_grid(config["safe_params"], kind="safe", divisions=shared_divisions)
    aggressive_combos = expand_param_grid(
        config["aggressive_params"], kind="aggressive", divisions=shared_divisions
    )
    safe_grouped = _group_by_division(safe_combos)
    aggressive_grouped = _group_by_division(aggressive_combos)

    divisions = sorted(set(safe_grouped) & set(aggressive_grouped))
    if not divisions:
        raise ValueError("Safe and aggressive grids must share at least one division value")

    results = []
    output_handle = None
    if args.output:
        _ensure_output_path(args.output)
        output_handle = args.output.open("w", encoding="utf-8")

    try:
        for division in divisions:
            for safe_values in safe_grouped[division]:
                safe_params = _as_strategy_params(safe_values)
                for aggressive_values in aggressive_grouped[division]:
                    aggressive_params = _as_strategy_params(aggressive_values)
                    metrics = engine.run(settings, safe_params, aggressive_params)
                    score = _calculate_score(metrics.cagr_pct, metrics.max_drawdown_pct, args.drawdown_weight)
                    results.append(
                        {
                            "score": score,
                            "cagr_pct": metrics.cagr_pct,
                            "max_drawdown_pct": metrics.max_drawdown_pct,
                            "safe_params": list(safe_values),
                            "aggressive_params": list(aggressive_values),
                        }
                    )
                    if output_handle is not None:
                        record = {
                            "score": score,
                            "metrics": asdict(metrics),
                            "safe_params": list(safe_values),
                            "aggressive_params": list(aggressive_values),
                        }
                        output_handle.write(json.dumps(record) + "\n")
    finally:
        if output_handle is not None:
            output_handle.close()

    if not results:
        print("No combinations were evaluated.")
        return 1

    results.sort(key=lambda item: item["score"], reverse=True)
    top_k = max(1, args.top_k)
    print(f"Top {top_k} combinations (score = CAGR - drawdown × {args.drawdown_weight:.2f}):")
    for rank, result in enumerate(results[:top_k], start=1):
        score = result["score"]
        cagr = result["cagr_pct"]
        drawdown = result["max_drawdown_pct"]
        safe = result["safe_params"]
        aggressive = result["aggressive_params"]
        print(
            f"  #{rank}: score={score:.4f} CAGR={cagr:.4f}% drawdown={drawdown:.4f}% "
            f"safe={safe} aggressive={aggressive}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
