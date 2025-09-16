#!/usr/bin/env python3
"""Run local backtests over the safe/aggressive parameter grid."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from backtest_engine import BacktestEngine, BacktestSettings, EntryGatePair, StrategyParameters

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


def _build_entry_gates(prices: Sequence[float], mode_rules: Optional[Dict[str, Any]]) -> Optional[EntryGatePair]:
    if not mode_rules:
        return None
    indicator = mode_rules.get("indicator", "wilder_rsi")
    if indicator != "wilder_rsi":
        raise ValueError(f"Unsupported indicator for mode_rules: {indicator!r}")
    period_raw = mode_rules.get("period", 14)
    try:
        period = int(period_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid RSI period inside mode_rules: {period_raw!r}") from exc
    if period <= 0:
        raise ValueError("RSI period must be a positive integer")

    safe_cfg = _parse_safe_config(mode_rules.get("safe", {}))
    aggressive_cfg = _parse_aggressive_config(mode_rules.get("aggressive", {}))
    rsi = _compute_wilder_rsi(prices, period)
    safe_flags, aggressive_flags = _derive_entry_flags(rsi, safe_cfg, aggressive_cfg)
    return EntryGatePair(safe=safe_flags, aggressive=aggressive_flags)


def _compute_wilder_rsi(prices: Sequence[float], period: int) -> List[Optional[float]]:
    if len(prices) < period + 1:
        return [None for _ in prices]
    gains: List[float] = []
    losses: List[float] = []
    rsi: List[Optional[float]] = [None for _ in prices]
    for i in range(1, period + 1):
        change = prices[i] - prices[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    rsi[period] = _calculate_rsi(avg_gain, avg_loss)
    for i in range(period + 1, len(prices)):
        change = prices[i] - prices[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        rsi[i] = _calculate_rsi(avg_gain, avg_loss)
    return rsi


def _calculate_rsi(avg_gain: float, avg_loss: float) -> Optional[float]:
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else None
    if avg_gain == 0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _parse_safe_config(raw: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    return {
        "falling_above": _coerce_optional_float(raw.get("falling_above")),
        "falling_between": _coerce_optional_range(raw.get("falling_between"), "falling_between"),
        "cross_below": _coerce_optional_float(raw.get("cross_below")),
    }


def _parse_aggressive_config(raw: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    return {
        "rising_below": _coerce_optional_float(raw.get("rising_below")),
        "rising_between": _coerce_optional_range(raw.get("rising_between"), "rising_between"),
        "cross_above": _coerce_optional_float(raw.get("cross_above")),
    }


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a numeric value, got {value!r}") from exc


def _coerce_optional_range(value: Any, label: str) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"'{label}' must be a two-element array")
    low, high = value
    low_f = _coerce_optional_float(low)
    high_f = _coerce_optional_float(high)
    if low_f is None or high_f is None:
        raise ValueError(f"'{label}' values must be numeric")
    return (low_f, high_f)


def _derive_entry_flags(
    rsi: Sequence[Optional[float]],
    safe_cfg: Dict[str, Optional[Any]],
    aggressive_cfg: Dict[str, Optional[Any]],
) -> Tuple[List[bool], List[bool]]:
    safe_flags = [False for _ in rsi]
    aggressive_flags = [False for _ in rsi]
    for index in range(1, len(rsi)):
        previous = rsi[index - 1]
        current = rsi[index]
        if previous is None or current is None:
            continue
        if _safe_allows(previous, current, safe_cfg):
            safe_flags[index] = True
        if _aggressive_allows(previous, current, aggressive_cfg):
            aggressive_flags[index] = True
    return safe_flags, aggressive_flags


def _safe_allows(previous: float, current: float, cfg: Dict[str, Optional[Any]]) -> bool:
    if current >= previous:
        return False
    falling_above = cfg.get("falling_above")
    if falling_above is not None and previous > falling_above:
        return True
    falling_between = cfg.get("falling_between")
    if falling_between is not None:
        low, high = falling_between
        if low < previous <= high:
            return True
    cross_below = cfg.get("cross_below")
    if cross_below is not None and previous >= cross_below and current < cross_below:
        return True
    return False


def _aggressive_allows(previous: float, current: float, cfg: Dict[str, Optional[Any]]) -> bool:
    if current <= previous:
        return False
    cross_above = cfg.get("cross_above")
    if cross_above is not None and previous <= cross_above and current > cross_above:
        return True
    rising_between = cfg.get("rising_between")
    if rising_between is not None:
        low, high = rising_between
        if low < previous <= high:
            return True
    rising_below = cfg.get("rising_below")
    if rising_below is not None and previous < rising_below:
        return True
    return False

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
    entry_gates = _build_entry_gates(prices, config.get("mode_rules"))

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
                    metrics = engine.run(
                        settings,
                        safe_params,
                        aggressive_params,
                        entry_gates=entry_gates,
                    )
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
