"""Parameter optimizer for Dongpa strategy.

Uses random sampling from user-defined parameter ranges to explore
the parameter space efficiently. Evaluates sampled combinations on
training and test windows, and writes a markdown summary.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime
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
class ParamRange:
    """Parameter range for random sampling."""
    min_val: float
    max_val: float
    is_int: bool = False

    def sample(self) -> float | int:
        """Generate a random value within the range."""
        if self.is_int:
            return random.randint(int(self.min_val), int(self.max_val))
        return random.uniform(self.min_val, self.max_val)


@dataclass
class ModeParamRanges:
    """Parameter ranges for a mode (defense or offense)."""
    buy_cond_pct: ParamRange
    tp_pct: ParamRange
    max_hold_days: ParamRange
    slices: ParamRange
    stop_loss_pct: ParamRange | None = None  # None means no stop loss
    allow_no_stop_loss: bool = True  # Whether to randomly include no-SL cases

    def sample(self) -> dict[str, float | int | None]:
        """Generate a random parameter set."""
        params = {
            "buy_cond_pct": self.buy_cond_pct.sample(),
            "tp_pct": self.tp_pct.sample(),
            "max_hold_days": self.max_hold_days.sample(),
            "slices": self.slices.sample(),
        }
        # Randomly decide whether to use stop loss
        if self.allow_no_stop_loss and random.random() < 0.3:  # 30% chance of no SL
            params["stop_loss_pct"] = None
        elif self.stop_loss_pct is not None:
            params["stop_loss_pct"] = self.stop_loss_pct.sample()
        else:
            params["stop_loss_pct"] = None
        return params


@dataclass
class CapitalParamRanges:
    """Parameter ranges for capital management."""
    refresh_cycle_days: ParamRange
    profit_compound_rate: ParamRange
    loss_compound_rate: ParamRange

    def sample(self) -> dict[str, float | int]:
        """Generate a random parameter set."""
        return {
            "refresh_cycle_days": self.refresh_cycle_days.sample(),
            "profit_compound_rate": self.profit_compound_rate.sample(),
            "loss_compound_rate": self.loss_compound_rate.sample(),
            "slippage_pct": 0.0,
        }


@dataclass
class RSIThresholdRanges:
    """Parameter ranges for RSI threshold tuning."""
    rsi_high_threshold: ParamRange  # Upper RSI threshold for defense mode
    rsi_mid_low: ParamRange         # Lower bound of middle range
    rsi_mid_high: ParamRange        # Upper bound of middle range
    rsi_low_threshold: ParamRange   # Lower RSI threshold for offense mode
    rsi_neutral: ParamRange         # Neutral line for crossover detection

    def sample(self) -> dict[str, float]:
        """Generate a random RSI threshold set with constraint validation."""
        # Sample values
        rsi_low = self.rsi_low_threshold.sample()
        rsi_mid_low = self.rsi_mid_low.sample()
        rsi_neutral = self.rsi_neutral.sample()
        rsi_mid_high = self.rsi_mid_high.sample()
        rsi_high = self.rsi_high_threshold.sample()

        # Ensure logical ordering: low < mid_low < neutral < mid_high < high
        values = sorted([rsi_low, rsi_mid_low, rsi_neutral, rsi_mid_high, rsi_high])

        return {
            "rsi_low_threshold": values[0],
            "rsi_mid_low": values[1],
            "rsi_neutral": values[2],
            "rsi_mid_high": values[3],
            "rsi_high_threshold": values[4],
        }


@dataclass
class MAPeriodRanges:
    """Parameter ranges for MA period tuning (for ma_cross strategy)."""
    ma_short_period: ParamRange  # Short MA period (weeks)
    ma_long_period: ParamRange   # Long MA period (weeks)

    def sample(self) -> dict[str, int]:
        """Generate a random MA period set with constraint validation."""
        short = self.ma_short_period.sample()
        long = self.ma_long_period.sample()

        # Ensure short < long
        if short >= long:
            short, long = min(short, long), max(short, long)
            # Add minimum gap of 2 periods
            if long - short < 2:
                long = short + 2

        return {
            "ma_short_period": int(short),
            "ma_long_period": int(long),
        }


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
    n_samples: int = 100  # Number of random samples to generate
    output_path: Path = Path("strategy_performance.md")
    # Mode switching strategy
    mode_switch_strategy: str = "rsi"  # "rsi" or "ma_cross"
    # Parameter ranges
    defense_ranges: ModeParamRanges | None = None
    offense_ranges: ModeParamRanges | None = None
    capital_ranges: CapitalParamRanges | None = None
    rsi_threshold_ranges: RSIThresholdRanges | None = None  # RSI threshold optimization
    optimize_rsi_thresholds: bool = False  # Enable RSI threshold optimization
    ma_period_ranges: MAPeriodRanges | None = None  # MA period optimization
    optimize_ma_periods: bool = False  # Enable MA period optimization


# Default parameter ranges (wider ranges for better exploration)
DEFAULT_DEFENSE_RANGES = ModeParamRanges(
    buy_cond_pct=ParamRange(0.5, 10.0, is_int=False),
    tp_pct=ParamRange(0.1, 3.0, is_int=False),
    max_hold_days=ParamRange(5, 90, is_int=True),
    slices=ParamRange(2, 20, is_int=True),
    stop_loss_pct=ParamRange(2.0, 50.0, is_int=False),
    allow_no_stop_loss=True,
)

DEFAULT_OFFENSE_RANGES = ModeParamRanges(
    buy_cond_pct=ParamRange(1.0, 15.0, is_int=False),
    tp_pct=ParamRange(0.5, 10.0, is_int=False),
    max_hold_days=ParamRange(2, 60, is_int=True),
    slices=ParamRange(2, 20, is_int=True),
    stop_loss_pct=ParamRange(2.0, 50.0, is_int=False),
    allow_no_stop_loss=True,
)

DEFAULT_CAPITAL_RANGES = CapitalParamRanges(
    refresh_cycle_days=ParamRange(1, 60, is_int=True),
    profit_compound_rate=ParamRange(0.3, 1.0, is_int=False),
    loss_compound_rate=ParamRange(0.0, 0.8, is_int=False),
)

DEFAULT_RSI_THRESHOLD_RANGES = RSIThresholdRanges(
    rsi_low_threshold=ParamRange(20.0, 40.0, is_int=False),
    rsi_mid_low=ParamRange(35.0, 45.0, is_int=False),
    rsi_neutral=ParamRange(45.0, 55.0, is_int=False),
    rsi_mid_high=ParamRange(55.0, 65.0, is_int=False),
    rsi_high_threshold=ParamRange(60.0, 80.0, is_int=False),
)

DEFAULT_MA_PERIOD_RANGES = MAPeriodRanges(
    ma_short_period=ParamRange(3, 10, is_int=True),   # 3-10 weeks (15-50 trading days)
    ma_long_period=ParamRange(15, 30, is_int=True),   # 15-30 weeks (75-150 trading days)
)


@dataclass
class OptimizationResult:
    defense: ModeParams
    offense: ModeParams
    capital: CapitalParams
    score: float
    train_metrics: dict
    test_metrics: dict
    rsi_thresholds: dict | None = None  # RSI threshold configuration if optimized
    ma_periods: dict | None = None      # MA period configuration if optimized
    mode_switch_strategy: str = "rsi"   # Mode switching strategy used


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


def _format_rsi_thresholds(rsi_cfg: dict | None) -> str:
    if rsi_cfg is None:
        return "기본값 (35/40/50/60/65)"
    return (
        f"Low {rsi_cfg['rsi_low_threshold']:.1f}, "
        f"MidLow {rsi_cfg['rsi_mid_low']:.1f}, "
        f"Neutral {rsi_cfg['rsi_neutral']:.1f}, "
        f"MidHigh {rsi_cfg['rsi_mid_high']:.1f}, "
        f"High {rsi_cfg['rsi_high_threshold']:.1f}"
    )


def _format_ma_periods(ma_cfg: dict | None) -> str:
    if ma_cfg is None:
        return "기본값 (5/20주)"
    return f"Short {ma_cfg['ma_short_period']}주, Long {ma_cfg['ma_long_period']}주"


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
    # Use provided ranges or fall back to defaults
    defense_ranges = cfg.defense_ranges or DEFAULT_DEFENSE_RANGES
    offense_ranges = cfg.offense_ranges or DEFAULT_OFFENSE_RANGES
    capital_ranges = cfg.capital_ranges or DEFAULT_CAPITAL_RANGES
    rsi_threshold_ranges = cfg.rsi_threshold_ranges or DEFAULT_RSI_THRESHOLD_RANGES
    ma_period_ranges = cfg.ma_period_ranges or DEFAULT_MA_PERIOD_RANGES

    results: list[OptimizationResult] = []
    failed_count = 0

    # Generate n_samples random parameter combinations
    for i in range(cfg.n_samples):
        try:
            d_cfg = defense_ranges.sample()
            o_cfg = offense_ranges.sample()
            c_cfg = capital_ranges.sample()

            defense = ModeParams(**d_cfg)
            offense = ModeParams(**o_cfg)
            capital = CapitalParams(initial_cash=cfg.initial_cash, **c_cfg)

            # Build StrategyParams based on mode switching strategy
            rsi_thresholds = None
            ma_periods = None
            params_dict = {
                "target_ticker": cfg.target_ticker,
                "momentum_ticker": cfg.momentum_ticker,
                "benchmark_ticker": cfg.benchmark_ticker,
                "rsi_period": cfg.rsi_period,
                "reset_on_mode_change": True,
                "enable_netting": cfg.enable_netting,
                "defense": defense,
                "offense": offense,
                "mode_switch_strategy": cfg.mode_switch_strategy,
            }

            # Add RSI thresholds if optimizing
            if cfg.optimize_rsi_thresholds and cfg.mode_switch_strategy == "rsi":
                rsi_thresholds = rsi_threshold_ranges.sample()
                params_dict.update(rsi_thresholds)

            # Add MA periods if optimizing
            if cfg.optimize_ma_periods and cfg.mode_switch_strategy == "ma_cross":
                ma_periods = ma_period_ranges.sample()
                params_dict.update(ma_periods)

            params = StrategyParams(**params_dict)

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
                    rsi_thresholds=rsi_thresholds,
                    ma_periods=ma_periods,
                    mode_switch_strategy=cfg.mode_switch_strategy,
                )
            )

            # Progress logging every 100 samples
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{cfg.n_samples} samples evaluated ({len(results)} successful, {failed_count} failed)")
        except Exception as e:  # noqa: BLE001
            failed_count += 1
            # Continue to next sample on error
            continue

    print(f"Completed: {len(results)} successful evaluations, {failed_count} failed")
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
        "# Dongpa Parameter Optimization (Random Search)\n\n"
        f"- Target: `{cfg.target_ticker}`\n"
        f"- Momentum: `{cfg.momentum_ticker}`\n"
        f"- Benchmark: `{cfg.benchmark_ticker or '없음'}`\n"
        f"- Training windows: {', '.join(f'{rng[0]}~{rng[1]}' for rng in cfg.train_ranges)}\n"
        f"- Test window: {cfg.test_range[0]}~{cfg.test_range[1]}\n"
        f"- Random samples: {cfg.n_samples}\n"
        f"- Score penalty (MDD weight): {cfg.score_penalty:.2f}\n"
        f"- Mode switch strategy: {cfg.mode_switch_strategy}\n"
        f"- RSI threshold optimization: {'Enabled' if cfg.optimize_rsi_thresholds else 'Disabled'}\n"
        f"- MA period optimization: {'Enabled' if cfg.optimize_ma_periods else 'Disabled'}\n"
        f"- Total combinations: {len(results)}\n"
        "\n"
    )

    # Conditional table header based on optimization settings
    if cfg.optimize_rsi_thresholds:
        table_header = (
            "| Rank | Defense | Offense | Capital | RSI Thresholds | Score | Train CAGR | Train MDD | Test CAGR | Test MDD |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
    elif cfg.optimize_ma_periods:
        table_header = (
            "| Rank | Defense | Offense | Capital | MA Periods | Score | Train CAGR | Train MDD | Test CAGR | Test MDD |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
    else:
        table_header = (
            "| Rank | Defense | Offense | Capital | Score | Train CAGR | Train MDD | Test CAGR | Test MDD |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )

    rows = []
    for idx, res in enumerate(results, start=1):
        train_cagr = _percent(res.train_metrics.get("CAGR", 0.0))
        train_mdd = _percent(res.train_metrics.get("Max Drawdown", 0.0))
        test_cagr = _percent(res.test_metrics.get("CAGR", 0.0))
        test_mdd = _percent(res.test_metrics.get("Max Drawdown", 0.0))

        if cfg.optimize_rsi_thresholds:
            rows.append(
                "| {rank} | {defense} | {offense} | {capital} | {rsi} | {score:.4f} | {train_cagr:.2f}% | {train_mdd:.2f}% | {test_cagr:.2f}% | {test_mdd:.2f}% |".format(
                    rank=idx,
                    defense=_format_mode_params(res.defense),
                    offense=_format_mode_params(res.offense),
                    capital=_format_capital_params(res.capital),
                    rsi=_format_rsi_thresholds(res.rsi_thresholds),
                    score=res.score,
                    train_cagr=train_cagr,
                    train_mdd=train_mdd,
                    test_cagr=test_cagr,
                    test_mdd=test_mdd,
                )
            )
        elif cfg.optimize_ma_periods:
            rows.append(
                "| {rank} | {defense} | {offense} | {capital} | {ma} | {score:.4f} | {train_cagr:.2f}% | {train_mdd:.2f}% | {test_cagr:.2f}% | {test_mdd:.2f}% |".format(
                    rank=idx,
                    defense=_format_mode_params(res.defense),
                    offense=_format_mode_params(res.offense),
                    capital=_format_capital_params(res.capital),
                    ma=_format_ma_periods(res.ma_periods),
                    score=res.score,
                    train_cagr=train_cagr,
                    train_mdd=train_mdd,
                    test_cagr=test_cagr,
                    test_mdd=test_mdd,
                )
            )
        else:
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
        for idx, res in enumerate(results, start=1):
            detail_lines.append(f"\n### Rank {idx}\n")
            detail_lines.append(_metrics_row("Train", res.train_metrics))
            detail_lines.append(_metrics_row("Test", res.test_metrics))

        content = header + table_header + "\n".join(rows) + "\n" + "".join(detail_lines)
    else:
        content = header + "평가 가능한 전략 조합이 없습니다. 데이터 범위를 확인하세요.\n"
    output_path.write_text(content, encoding="utf-8")
    return output_path


# --------------------------- Public API ---------------------------

def optimize(cfg: OptimizerConfig) -> list[OptimizationResult]:
    train_target, train_momo, test_target, test_momo = _load_price_frames(cfg)
    results = _evaluate(train_target, train_momo, test_target, test_momo, cfg)
    return results


if __name__ == "__main__":  # pragma: no cover
    default_cfg = OptimizerConfig(
        target_ticker="SOXL",
        momentum_ticker="QQQ",
        benchmark_ticker="SOXX",
        n_samples=1500,  # Increased for better optimization
        score_penalty=0.7,  # Higher penalty for MDD (prioritize drawdown reduction)
    )
    try:
        results = optimize(default_cfg)
        print(f"Evaluated {len(results)} parameter combinations")
        if results:
            print(f"\nTop 3 results:")
            for i, res in enumerate(results[:3], 1):
                print(f"  {i}. Score={res.score:.2f}, Test MDD={res.test_metrics.get('Max Drawdown', 0):.2%}, Test CAGR={res.test_metrics.get('CAGR', 0):.2%}")
    except Exception as exc:  # noqa: BLE001 - top-level script guard
        print(f"Optimizer failed: {exc}")
