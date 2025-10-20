"""Parameter optimizer for Dongpa strategy.

Combines hand-picked parameter candidates for defense/offense modes and
capital management, evaluates them on the requested training and test
windows, and writes a markdown summary.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from itertools import product
from pathlib import Path
from typing import Sequence

import pandas as pd
import yfinance as yf

from dongpa_engine import (
    CapitalParams,
    DongpaBacktester,
    ModeParams,
    StrategyParams,
    summarize,
)

# --------------------------- Date helpers ---------------------------

DateLike = str | date | datetime | pd.Timestamp
DateRange = tuple[DateLike, DateLike]


def _to_timestamp(value: DateLike) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _slice_by_ranges(df: pd.DataFrame, ranges: Sequence[DateRange]) -> pd.DataFrame:
    slices: list[pd.DataFrame] = []
    for start, end in ranges:
        start_ts = _to_timestamp(start)
        end_ts = _to_timestamp(end)
        piece = df.loc[start_ts:end_ts].copy()
        if not piece.empty:
            slices.append(piece)
    if not slices:
        raise ValueError("Requested ranges produced an empty dataframe")
    combined = pd.concat(slices).sort_index()
    idx = pd.DatetimeIndex(combined.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    combined.index = idx
    return combined


def _slice_by_range(df: pd.DataFrame, date_range: DateRange) -> pd.DataFrame:
    start, end = (_to_timestamp(date_range[0]), _to_timestamp(date_range[1]))
    piece = df.loc[start:end].copy()
    if piece.empty:
        raise ValueError("Requested range produced an empty dataframe")
    idx = pd.DatetimeIndex(piece.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    piece.index = idx
    return piece


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Price frame is empty")
    frame = df.copy()
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    frame.index = idx
    return frame


# --------------------------- Optimizer config ---------------------------

TRAIN_RANGES: tuple[DateRange, ...] = (
    ("2022-01-01", "2022-12-31"),
    ("2025-01-01", "2025-10-01"),
)
TEST_RANGE: DateRange = ("2023-01-01", "2024-12-31")


@dataclass
class OptimizerConfig:
    target_ticker: str
    momentum_ticker: str
    benchmark_ticker: str | None = None
    initial_cash: float = 10000.0
    train_ranges: Sequence[DateRange] = TRAIN_RANGES
    test_range: DateRange = TEST_RANGE
    rsi_period: int = 14
    enable_netting: bool = True
    score_penalty: float = 0.6
    top_n: int = 5
    output_path: Path = Path("strategy_performance.md")


# Default parameter candidates (hand curated to balance exploration vs cost)
DEFENSE_CANDIDATES: tuple[dict[str, float | int | None], ...] = (
    {"buy_cond_pct": 2.0, "tp_pct": 0.40, "max_hold_days": 35, "slices": 8, "stop_loss_pct": None},
    {"buy_cond_pct": 2.0, "tp_pct": 0.40, "max_hold_days": 35, "slices": 8, "stop_loss_pct": 4.0},
    {"buy_cond_pct": 2.5, "tp_pct": 0.32, "max_hold_days": 32, "slices": 7, "stop_loss_pct": 6.0},
    {"buy_cond_pct": 2.5, "tp_pct": 0.32, "max_hold_days": 32, "slices": 7, "stop_loss_pct": 8.0},
    {"buy_cond_pct": 3.0, "tp_pct": 0.28, "max_hold_days": 28, "slices": 6, "stop_loss_pct": 10.0},
    {"buy_cond_pct": 3.0, "tp_pct": 0.28, "max_hold_days": 28, "slices": 6, "stop_loss_pct": 12.0},
    {"buy_cond_pct": 3.5, "tp_pct": 0.24, "max_hold_days": 24, "slices": 5, "stop_loss_pct": 14.0},
    {"buy_cond_pct": 3.5, "tp_pct": 0.24, "max_hold_days": 24, "slices": 5, "stop_loss_pct": 16.0},
)

OFFENSE_CANDIDATES: tuple[dict[str, float | int | None], ...] = (
    {"buy_cond_pct": 4.0, "tp_pct": 1.5, "max_hold_days": 15, "slices": 8, "stop_loss_pct": 6.0},
    {"buy_cond_pct": 4.0, "tp_pct": 1.5, "max_hold_days": 15, "slices": 8, "stop_loss_pct": 8.0},
    {"buy_cond_pct": 4.5, "tp_pct": 2.0, "max_hold_days": 12, "slices": 7, "stop_loss_pct": 10.0},
    {"buy_cond_pct": 4.5, "tp_pct": 2.0, "max_hold_days": 12, "slices": 7, "stop_loss_pct": 12.0},
    {"buy_cond_pct": 5.0, "tp_pct": 2.5, "max_hold_days": 10, "slices": 6, "stop_loss_pct": 14.0},
    {"buy_cond_pct": 5.0, "tp_pct": 2.5, "max_hold_days": 10, "slices": 6, "stop_loss_pct": 16.0},
    {"buy_cond_pct": 5.5, "tp_pct": 3.0, "max_hold_days": 8, "slices": 5, "stop_loss_pct": 12.0},
    {"buy_cond_pct": 5.5, "tp_pct": 3.0, "max_hold_days": 8, "slices": 5, "stop_loss_pct": 15.0},
    {"buy_cond_pct": 6.0, "tp_pct": 3.5, "max_hold_days": 7, "slices": 5, "stop_loss_pct": 18.0},
    {"buy_cond_pct": 6.5, "tp_pct": 4.0, "max_hold_days": 6, "slices": 4, "stop_loss_pct": 20.0},
)

CAPITAL_CANDIDATES: tuple[dict[str, float | int], ...] = (
    {"refresh_cycle_days": 5, "profit_compound_rate": 0.75, "loss_compound_rate": 0.35, "slippage_pct": 0.0},
    {"refresh_cycle_days": 7, "profit_compound_rate": 0.80, "loss_compound_rate": 0.30, "slippage_pct": 0.0},
    {"refresh_cycle_days": 10, "profit_compound_rate": 0.70, "loss_compound_rate": 0.40, "slippage_pct": 0.0},
    {"refresh_cycle_days": 15, "profit_compound_rate": 0.90, "loss_compound_rate": 0.25, "slippage_pct": 0.0},
)


@dataclass
class OptimizationResult:
    defense: ModeParams
    offense: ModeParams
    capital: CapitalParams
    score: float
    train_metrics: dict
    test_metrics: dict


# --------------------------- Data loading ---------------------------

def _download_price_history(ticker: str, start: DateLike, end: DateLike) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")
    return _normalize(data)


def _load_price_frames(cfg: OptimizerConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_starts = [cfg.test_range[0], *[rng[0] for rng in cfg.train_ranges]]
    all_ends = [cfg.test_range[1], *[rng[1] for rng in cfg.train_ranges]]
    global_start = min(_to_timestamp(val) for val in all_starts)
    global_end = max(_to_timestamp(val) for val in all_ends)

    target_all = _download_price_history(cfg.target_ticker, global_start, global_end)
    momo_all = _download_price_history(cfg.momentum_ticker, global_start, global_end)

    train_target = _slice_by_ranges(target_all, cfg.train_ranges)
    test_target = _slice_by_range(target_all, cfg.test_range)

    train_momo = _slice_by_ranges(momo_all, cfg.train_ranges)
    test_momo = _slice_by_range(momo_all, cfg.test_range)

    return train_target, train_momo, test_target, test_momo


# --------------------------- Scoring helpers ---------------------------

def _format_mode_params(mp: ModeParams) -> str:
    stop = "없음" if mp.stop_loss_pct is None else f"{mp.stop_loss_pct:.1f}%"
    return (
        f"조건 {mp.buy_cond_pct:.1f}%, TP {mp.tp_pct:.1f}%, 보유 {mp.max_hold_days}일, "
        f"분할 {mp.slices}, SL {stop}"
    )


def _format_capital_params(cp: CapitalParams) -> str:
    return (
        f"주기 {cp.refresh_cycle_days}일, PCR {cp.profit_compound_rate:.2f}, "
        f"LCR {cp.loss_compound_rate:.2f}"
    )


def _percent(value: float) -> float:
    return round(value * 100.0, 2)


def _score(train_metrics: dict, test_metrics: dict, penalty: float) -> float:
    train_cagr = train_metrics.get("CAGR", 0.0)
    train_mdd = abs(train_metrics.get("Max Drawdown", 0.0))
    test_cagr = test_metrics.get("CAGR", 0.0)
    test_mdd = abs(test_metrics.get("Max Drawdown", 0.0))
    avg_cagr = (train_cagr + test_cagr) / 2
    avg_mdd = (train_mdd + test_mdd) / 2
    return avg_cagr - penalty * avg_mdd


def _evaluate(
    train_target: pd.DataFrame,
    train_momo: pd.DataFrame,
    test_target: pd.DataFrame,
    test_momo: pd.DataFrame,
    cfg: OptimizerConfig,
) -> list[OptimizationResult]:
    results: list[OptimizationResult] = []
    for d_cfg, o_cfg, c_cfg in product(DEFENSE_CANDIDATES, OFFENSE_CANDIDATES, CAPITAL_CANDIDATES):
        defense = ModeParams(**d_cfg)
        offense = ModeParams(**o_cfg)
        capital = CapitalParams(initial_cash=cfg.initial_cash, **c_cfg)
        params = StrategyParams(
            target_ticker=cfg.target_ticker,
            momentum_ticker=cfg.momentum_ticker,
            benchmark_ticker=cfg.benchmark_ticker,
            rsi_period=cfg.rsi_period,
            reset_on_mode_change=True,
            enable_netting=cfg.enable_netting,
            defense=defense,
            offense=offense,
        )

        train_backtester = DongpaBacktester(train_target, train_momo, params, capital)
        train_res = train_backtester.run()
        train_metrics = summarize(train_res["equity"])

        test_backtester = DongpaBacktester(test_target, test_momo, params, capital)
        test_res = test_backtester.run()
        test_metrics = summarize(test_res["equity"])

        score_value = _score(train_metrics, test_metrics, cfg.score_penalty)
        results.append(
            OptimizationResult(
                defense=defense,
                offense=offense,
                capital=capital,
                score=score_value,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )
        )
    return sorted(results, key=lambda res: res.score, reverse=True)


# --------------------------- Markdown writer ---------------------------

def _metrics_row(prefix: str, metrics: dict) -> str:
    cagr = _percent(metrics.get("CAGR", 0.0))
    mdd = _percent(metrics.get("Max Drawdown", 0.0))
    sharpe = round(metrics.get("Sharpe (rf=0)", 0.0), 2)
    final_eq = round(metrics.get("Final Equity", 0.0), 2)
    return f"| {prefix} | {cagr:.2f}% | {mdd:.2f}% | {sharpe:.2f} | {final_eq:,.2f} |\n"


def write_markdown(results: Sequence[OptimizationResult], cfg: OptimizerConfig) -> Path:
    output_path = cfg.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Dongpa Parameter Optimization\n\n"
        f"- Target: `{cfg.target_ticker}`\n"
        f"- Momentum: `{cfg.momentum_ticker}`\n"
        f"- Benchmark: `{cfg.benchmark_ticker or '없음'}`\n"
        f"- Training windows: {', '.join(f'{rng[0]}~{rng[1]}' for rng in cfg.train_ranges)}\n"
        f"- Test window: {cfg.test_range[0]}~{cfg.test_range[1]}\n"
        f"- Score penalty (MDD weight): {cfg.score_penalty:.2f}\n"
        "\n"
    )
    table_header = (
        "| Rank | Defense | Offense | Capital | Score | Train CAGR | Train MDD | Test CAGR | Test MDD |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for idx, res in enumerate(results[: cfg.top_n], start=1):
        train_cagr = _percent(res.train_metrics.get("CAGR", 0.0))
        train_mdd = _percent(res.train_metrics.get("Max Drawdown", 0.0))
        test_cagr = _percent(res.test_metrics.get("CAGR", 0.0))
        test_mdd = _percent(res.test_metrics.get("Max Drawdown", 0.0))
        rows.append(
            "| {rank} | {defense} | {offense} | {capital} | {score:.4f} | {train_cagr:.2f}% | {train_mdd:.2f}% | {test_cagr:.2f}% | {test_mdd:.2f}% |".format(
                rank=idx,
                defense=_format_mode_params(res.defense),
                offense=_format_mode_params(res.offense),
                capital=_format_capital_params(res.capital),
                score=res.score,
                train_cagr=train_cagr,
                train_mdd=train_mdd,
                test_cagr=test_cagr,
                test_mdd=test_mdd,
            )
        )

    if rows:
        detail_lines = [
            "\n## Detailed Metrics\n",
            "| Phase | CAGR | Max Drawdown | Sharpe | Final Equity |\n",
            "| --- | --- | --- | --- | --- |\n",
        ]
        for idx, res in enumerate(results[: cfg.top_n], start=1):
            detail_lines.append(f"\n### Rank {idx}\n")
            detail_lines.append(_metrics_row("Train", res.train_metrics))
            detail_lines.append(_metrics_row("Test", res.test_metrics))

        content = header + table_header + "\n".join(rows) + "\n" + "".join(detail_lines)
    else:
        content = header + "평가 가능한 전략 조합이 없습니다. 데이터 범위를 확인하세요.\n"
    output_path.write_text(content, encoding="utf-8")
    return output_path


# --------------------------- Public API ---------------------------

def optimize(cfg: OptimizerConfig) -> tuple[list[OptimizationResult], Path]:
    train_target, train_momo, test_target, test_momo = _load_price_frames(cfg)
    results = _evaluate(train_target, train_momo, test_target, test_momo, cfg)
    md_path = write_markdown(results, cfg)
    return results, md_path


if __name__ == "__main__":  # pragma: no cover
    default_cfg = OptimizerConfig(
        target_ticker="SOXL",
        momentum_ticker="QQQ",
        benchmark_ticker="SOXX",
    )
    try:
        optimize(default_cfg)
        print(f"Markdown report saved to {default_cfg.output_path}")
    except Exception as exc:  # noqa: BLE001 - top-level script guard
        print(f"Optimizer failed: {exc}")
