"""Optuna-based parameter optimizer for Dongpa strategy.

Uses Bayesian Optimization (TPE algorithm) to maximize Calmar ratio
(CAGR / |MDD|) averaged across training and test windows.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import optuna
import pandas as pd
import yfinance as yf

from dongpa_engine import (
    CapitalParams,
    DongpaBacktester,
    ModeParams,
    StrategyParams,
    summarize,
)

# --------------------------- Date/data helpers (moved from dongpa_optimizer) --

DateLike = str | date | datetime | pd.Timestamp
DateRange = tuple[DateLike, DateLike]


def _to_timestamp(value: DateLike) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Price frame is empty")
    frame = df.copy()
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    frame.index = idx
    return frame


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


def _download_price_history(ticker: str, start: DateLike, end: DateLike) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")
    return _normalize(data)


def _score(train_metrics: dict, test_metrics: dict) -> float:
    """Calmar ratio = avg_cagr / avg_mdd.  Higher is better."""
    train_cagr = train_metrics.get("CAGR", 0.0)
    train_mdd = abs(train_metrics.get("Max Drawdown", 0.0))
    test_cagr = test_metrics.get("CAGR", 0.0)
    test_mdd = abs(test_metrics.get("Max Drawdown", 0.0))
    avg_cagr = (train_cagr + test_cagr) / 2
    avg_mdd = (train_mdd + test_mdd) / 2
    if avg_mdd == 0:
        return avg_cagr * 100 if avg_cagr > 0 else 0.0
    return avg_cagr / avg_mdd


@dataclass
class OptimizationResult:
    defense: ModeParams
    offense: ModeParams
    capital: CapitalParams
    score: float
    train_metrics: dict
    test_metrics: dict
    combined_metrics: dict
    rsi_thresholds: dict | None = None
    ma_periods: dict | None = None
    roc_period: dict | None = None
    mode_switch_strategy: str = "rsi"
    cash_limited_buy: bool = False

# Suppress Optuna's verbose logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


# --------------------------- Config ---------------------------

@dataclass
class OptunaConfig:
    """Configuration for Optuna-based optimization."""
    target_ticker: str
    momentum_ticker: str
    initial_cash: float = 10000.0
    train_ranges: Sequence[DateRange] = (
        ("2022-01-01", "2022-12-31"),
        ("2025-01-01", "2025-10-01"),
    )
    test_range: DateRange = ("2023-01-01", "2024-12-31")
    rsi_period: int = 14
    enable_netting: bool = True
    n_trials: int = 300
    top_n: int = 20

    # Mode switching strategy
    mode_switch_strategy: str = "rsi"  # "rsi", "ma_cross", or "both"

    # Defense parameter ranges (min, max)
    def_buy_range: tuple[float, float] = (0.5, 10.0)
    def_tp_range: tuple[float, float] = (0.1, 3.0)
    def_hold_range: tuple[int, int] = (5, 90)
    def_slices_range: tuple[int, int] = (1, 20)
    def_sl_range: tuple[float, float] = (2.0, 50.0)

    # Offense parameter ranges (min, max)
    off_buy_range: tuple[float, float] = (1.0, 15.0)
    off_tp_range: tuple[float, float] = (0.5, 10.0)
    off_hold_range: tuple[int, int] = (2, 60)
    off_slices_range: tuple[int, int] = (1, 20)
    off_sl_range: tuple[float, float] = (2.0, 50.0)

    # RSI threshold ranges (for "rsi" mode strategy)
    rsi_low_range: tuple[float, float] = (20.0, 40.0)
    rsi_mid_low_range: tuple[float, float] = (35.0, 45.0)
    rsi_neutral_range: tuple[float, float] = (45.0, 55.0)
    rsi_mid_high_range: tuple[float, float] = (55.0, 65.0)
    rsi_high_range: tuple[float, float] = (60.0, 80.0)

    # MA period ranges (for "ma_cross" mode strategy)
    ma_short_range: tuple[int, int] = (3, 10)
    ma_long_range: tuple[int, int] = (15, 30)

    # ROC period range (for "roc" mode strategy)
    roc_period_range: tuple[int, int] = (2, 12)

    # Feature toggles
    optimize_rsi_thresholds: bool = False
    optimize_ma_periods: bool = False
    optimize_roc_period: bool = False
    optimize_cash_limited_buy: bool = False

    # Constraints: list of (date_range, max_mdd, min_cagr) tuples
    # e.g. [( ("2022-01-01","2022-12-31"), -0.30, 0.0 )] means
    #   in 2022: MDD must be > -30% AND CAGR must be >= 0%
    # Set max_mdd to None to skip MDD check, min_cagr to None to skip CAGR check
    constraints: list[tuple[DateRange, float | None, float | None]] | None = None

    # Progress callback: (trial_number, n_trials, best_score)
    progress_callback: object = None  # callable | None


# --------------------------- Data loading ---------------------------

def _load_price_frames(cfg: OptunaConfig):
    """Download and slice price data for train/test/combined/constraint periods."""
    all_starts = [cfg.test_range[0], *[rng[0] for rng in cfg.train_ranges]]
    all_ends = [cfg.test_range[1], *[rng[1] for rng in cfg.train_ranges]]
    # Include constraint ranges in the global download window
    if cfg.constraints:
        for dr, _, _ in cfg.constraints:
            all_starts.append(dr[0])
            all_ends.append(dr[1])
    global_start = min(_to_timestamp(val) for val in all_starts)
    global_end = max(_to_timestamp(val) for val in all_ends)

    target_all = _download_price_history(cfg.target_ticker, global_start, global_end)
    momo_all = _download_price_history(cfg.momentum_ticker, global_start, global_end)

    train_target = _slice_by_ranges(target_all, cfg.train_ranges)
    test_target = _slice_by_range(target_all, cfg.test_range)

    combined_ranges = list(cfg.train_ranges) + [cfg.test_range]
    combined_target = _slice_by_ranges(target_all, combined_ranges)

    # Slice constraint periods
    constraint_frames = []
    if cfg.constraints:
        for dr, max_mdd, min_cagr in cfg.constraints:
            ct = _slice_by_range(target_all, dr)
            constraint_frames.append((ct, momo_all, dr, max_mdd, min_cagr))

    return train_target, momo_all, test_target, momo_all, combined_target, momo_all, constraint_frames


# --------------------------- Objective ---------------------------

def create_objective(
    train_target: pd.DataFrame,
    train_momo: pd.DataFrame,
    test_target: pd.DataFrame,
    test_momo: pd.DataFrame,
    cfg: OptunaConfig,
    constraint_frames: list | None = None,
):
    """Create an Optuna objective function for the Dongpa strategy."""

    def objective(trial: optuna.Trial) -> float:
        # --- Defense parameters ---
        d_buy = trial.suggest_float("def_buy_cond_pct", *cfg.def_buy_range)
        d_tp = trial.suggest_float("def_tp_pct", *cfg.def_tp_range)
        d_hold = trial.suggest_int("def_max_hold_days", *cfg.def_hold_range)
        d_slices = trial.suggest_int("def_slices", *cfg.def_slices_range)
        d_use_sl = trial.suggest_categorical("def_use_sl", [True, False])
        d_sl = trial.suggest_float("def_sl_pct", *cfg.def_sl_range) if d_use_sl else None

        # --- Offense parameters ---
        o_buy = trial.suggest_float("off_buy_cond_pct", *cfg.off_buy_range)
        o_tp = trial.suggest_float("off_tp_pct", *cfg.off_tp_range)
        o_hold = trial.suggest_int("off_max_hold_days", *cfg.off_hold_range)
        o_slices = trial.suggest_int("off_slices", *cfg.off_slices_range)
        o_use_sl = trial.suggest_categorical("off_use_sl", [True, False])
        o_sl = trial.suggest_float("off_sl_pct", *cfg.off_sl_range) if o_use_sl else None

        defense = ModeParams(
            buy_cond_pct=d_buy, tp_pct=d_tp,
            max_hold_days=d_hold, slices=d_slices,
            stop_loss_pct=d_sl,
        )
        offense = ModeParams(
            buy_cond_pct=o_buy, tp_pct=o_tp,
            max_hold_days=o_hold, slices=o_slices,
            stop_loss_pct=o_sl,
        )
        capital = CapitalParams(initial_cash=cfg.initial_cash)

        # --- Mode switching strategy ---
        mode_strategy = cfg.mode_switch_strategy
        if mode_strategy == "both":
            mode_strategy = trial.suggest_categorical("mode_strategy", ["rsi", "ma_cross", "roc"])

        # --- Cash-limited buy ---
        if cfg.optimize_cash_limited_buy:
            cash_limited_buy = trial.suggest_categorical("cash_limited_buy", [True, False])
        else:
            cash_limited_buy = False

        params_dict: dict = {
            "target_ticker": cfg.target_ticker,
            "momentum_ticker": cfg.momentum_ticker,
            "rsi_period": cfg.rsi_period,
            "enable_netting": cfg.enable_netting,
            "cash_limited_buy": cash_limited_buy,
            "defense": defense,
            "offense": offense,
            "mode_switch_strategy": mode_strategy,
        }

        # --- RSI threshold optimization ---
        if cfg.optimize_rsi_thresholds and mode_strategy == "rsi":
            rsi_low = trial.suggest_float("rsi_low_threshold", *cfg.rsi_low_range)
            rsi_mid_low = trial.suggest_float("rsi_mid_low", *cfg.rsi_mid_low_range)
            rsi_neutral = trial.suggest_float("rsi_neutral", *cfg.rsi_neutral_range)
            rsi_mid_high = trial.suggest_float("rsi_mid_high", *cfg.rsi_mid_high_range)
            rsi_high = trial.suggest_float("rsi_high_threshold", *cfg.rsi_high_range)

            # Enforce logical ordering: low < mid_low < neutral < mid_high < high
            values = sorted([rsi_low, rsi_mid_low, rsi_neutral, rsi_mid_high, rsi_high])
            params_dict["rsi_low_threshold"] = values[0]
            params_dict["rsi_mid_low"] = values[1]
            params_dict["rsi_neutral"] = values[2]
            params_dict["rsi_mid_high"] = values[3]
            params_dict["rsi_high_threshold"] = values[4]

        # --- MA period optimization ---
        if cfg.optimize_ma_periods and mode_strategy == "ma_cross":
            ma_short = trial.suggest_int("ma_short_period", *cfg.ma_short_range)
            ma_long = trial.suggest_int("ma_long_period", *cfg.ma_long_range)
            # Ensure short < long with minimum gap
            if ma_short >= ma_long:
                ma_short, ma_long = min(ma_short, ma_long), max(ma_short, ma_long)
                if ma_long - ma_short < 2:
                    ma_long = ma_short + 2
            params_dict["ma_short_period"] = ma_short
            params_dict["ma_long_period"] = ma_long

        # --- ROC period optimization ---
        if cfg.optimize_roc_period and mode_strategy == "roc":
            roc_period = trial.suggest_int("roc_period", *cfg.roc_period_range)
            params_dict["roc_period"] = roc_period

        params = StrategyParams(**params_dict)

        # --- Run backtests ---
        try:
            train_bt = DongpaBacktester(train_target, train_momo, params, capital)
            train_res = train_bt.run()
            train_metrics = summarize(train_res["equity"])

            # Early pruning: skip test if train CAGR is very negative
            train_cagr = train_metrics.get("CAGR", 0.0)
            if train_cagr < -0.5:
                raise optuna.TrialPruned()

            test_bt = DongpaBacktester(test_target, test_momo, params, capital)
            test_res = test_bt.run()
            test_metrics = summarize(test_res["equity"])

            # --- Constraint checking ---
            if constraint_frames:
                constraint_results = {}
                for c_target, c_momo, c_range, max_mdd, min_cagr in constraint_frames:
                    c_bt = DongpaBacktester(c_target, c_momo, params, capital)
                    c_res = c_bt.run()
                    c_metrics = summarize(c_res["equity"])
                    c_label = f"{c_range[0]}~{c_range[1]}"
                    constraint_results[c_label] = c_metrics

                    c_actual_mdd = c_metrics.get("Max Drawdown", 0.0)
                    c_actual_cagr = c_metrics.get("CAGR", 0.0)

                    if max_mdd is not None and c_actual_mdd < max_mdd:
                        raise optuna.TrialPruned()
                    if min_cagr is not None and c_actual_cagr < min_cagr:
                        raise optuna.TrialPruned()

                trial.set_user_attr("constraint_metrics", constraint_results)

            score = _score(train_metrics, test_metrics)

            # Store metrics as user attributes for later retrieval
            trial.set_user_attr("train_metrics", train_metrics)
            trial.set_user_attr("test_metrics", test_metrics)
            trial.set_user_attr("score", score)
            trial.set_user_attr("defense", {
                "buy_cond_pct": d_buy, "tp_pct": d_tp,
                "max_hold_days": d_hold, "slices": d_slices,
                "stop_loss_pct": d_sl,
            })
            trial.set_user_attr("offense", {
                "buy_cond_pct": o_buy, "tp_pct": o_tp,
                "max_hold_days": o_hold, "slices": o_slices,
                "stop_loss_pct": o_sl,
            })
            trial.set_user_attr("capital", {})
            trial.set_user_attr("mode_switch_strategy", mode_strategy)

            return score

        except optuna.TrialPruned:
            raise
        except Exception:
            logger.debug("Trial %d failed", trial.number, exc_info=True)
            raise optuna.TrialPruned()

    return objective


# --------------------------- Runner ---------------------------

def run_optuna(cfg: OptunaConfig) -> tuple[optuna.Study, tuple]:
    """Run Optuna optimization and return (study, price_frames).

    The returned price_frames can be passed to extract_results() to avoid
    re-downloading data from Yahoo Finance.
    """
    frames = _load_price_frames(cfg)
    train_target, train_momo, test_target, test_momo, _, _, constraint_frames = frames

    objective = create_objective(
        train_target, train_momo,
        test_target, test_momo,
        cfg,
        constraint_frames=constraint_frames or None,
    )

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="dongpa_optuna",
    )

    # Wrap objective with progress callback
    if cfg.progress_callback:
        original_objective = objective
        _best_score: list[float | None] = [None]

        def objective_with_progress(trial):
            try:
                result = original_objective(trial)
            except optuna.TrialPruned:
                cfg.progress_callback(trial.number + 1, cfg.n_trials, _best_score[0])
                raise
            score = trial.user_attrs.get("score")
            if score is not None and (_best_score[0] is None or score > _best_score[0]):
                _best_score[0] = score
            cfg.progress_callback(trial.number + 1, cfg.n_trials, _best_score[0])
            return result

        study.optimize(objective_with_progress, n_trials=cfg.n_trials, show_progress_bar=False)
    else:
        study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=True)

    return study, frames


# --------------------------- Result extraction ---------------------------

def extract_results(
    study: optuna.Study,
    cfg: OptunaConfig,
    top_n: int | None = None,
    price_frames: tuple | None = None,
) -> list[OptimizationResult]:
    """Extract top results from an Optuna study as OptimizationResult objects.

    Runs combined backtest for top trials to get combined_metrics.
    Pass price_frames from run_optuna() to avoid re-downloading data.
    """
    if top_n is None:
        top_n = cfg.top_n

    # Sort completed trials by score (descending)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.user_attrs.get("score", float("-inf")), reverse=True)
    top_trials = completed[:top_n]

    if not top_trials:
        return []

    # Reuse pre-loaded frames or download fresh
    if price_frames is None:
        price_frames = _load_price_frames(cfg)
    _, _, _, _, combined_target, combined_momo, _ = price_frames

    results: list[OptimizationResult] = []
    for trial in top_trials:
        d_attrs = trial.user_attrs.get("defense", {})
        o_attrs = trial.user_attrs.get("offense", {})
        c_attrs = trial.user_attrs.get("capital", {})
        mode_strategy = trial.user_attrs.get("mode_switch_strategy", cfg.mode_switch_strategy)

        defense = ModeParams(**d_attrs)
        offense = ModeParams(**o_attrs)
        capital = CapitalParams(initial_cash=cfg.initial_cash, **c_attrs)

        # Build StrategyParams for combined backtest
        params_dict: dict = {
            "target_ticker": cfg.target_ticker,
            "momentum_ticker": cfg.momentum_ticker,
            "rsi_period": cfg.rsi_period,
            "enable_netting": cfg.enable_netting,
            "defense": defense,
            "offense": offense,
            "mode_switch_strategy": mode_strategy,
        }

        # Reconstruct RSI/MA params from trial
        rsi_thresholds = None
        ma_periods = None
        if "rsi_low_threshold" in trial.params:
            rsi_thresholds = {
                "rsi_low_threshold": trial.params["rsi_low_threshold"],
                "rsi_mid_low": trial.params["rsi_mid_low"],
                "rsi_neutral": trial.params["rsi_neutral"],
                "rsi_mid_high": trial.params["rsi_mid_high"],
                "rsi_high_threshold": trial.params["rsi_high_threshold"],
            }
            # Note: sorted values were stored in user_attrs via params_dict
            # but trial.params has the raw suggested values. We re-sort.
            sorted_vals = sorted(rsi_thresholds.values())
            rsi_thresholds = {
                "rsi_low_threshold": sorted_vals[0],
                "rsi_mid_low": sorted_vals[1],
                "rsi_neutral": sorted_vals[2],
                "rsi_mid_high": sorted_vals[3],
                "rsi_high_threshold": sorted_vals[4],
            }
            params_dict.update(rsi_thresholds)

        if "ma_short_period" in trial.params:
            ma_short = trial.params["ma_short_period"]
            ma_long = trial.params["ma_long_period"]
            if ma_short >= ma_long:
                ma_short, ma_long = min(ma_short, ma_long), max(ma_short, ma_long)
                if ma_long - ma_short < 2:
                    ma_long = ma_short + 2
            ma_periods = {"ma_short_period": ma_short, "ma_long_period": ma_long}
            params_dict.update(ma_periods)

        roc_period_dict = None
        if "roc_period" in trial.params:
            roc_period_dict = {"roc_period": trial.params["roc_period"]}
            params_dict["roc_period"] = trial.params["roc_period"]

        # Cash-limited buy
        trial_cash_limited = trial.params.get("cash_limited_buy", False)
        params_dict["cash_limited_buy"] = trial_cash_limited

        # Run combined backtest
        try:
            params = StrategyParams(**params_dict)
            combined_bt = DongpaBacktester(combined_target, combined_momo, params, capital)
            combined_res = combined_bt.run()
            combined_metrics = summarize(combined_res["equity"])
        except Exception:
            combined_metrics = {}

        results.append(OptimizationResult(
            defense=defense,
            offense=offense,
            capital=capital,
            score=trial.user_attrs.get("score", 0.0),
            train_metrics=trial.user_attrs.get("train_metrics", {}),
            test_metrics=trial.user_attrs.get("test_metrics", {}),
            combined_metrics=combined_metrics,
            rsi_thresholds=rsi_thresholds,
            ma_periods=ma_periods,
            roc_period=roc_period_dict,
            mode_switch_strategy=mode_strategy,
            cash_limited_buy=trial_cash_limited,
        ))

    return results


def format_results_df(results: list[OptimizationResult]) -> pd.DataFrame:
    """Convert optimization results to a DataFrame for display."""
    rows = []
    for idx, res in enumerate(results, start=1):
        defense_sl = f"{res.defense.stop_loss_pct:.1f}%" if res.defense.stop_loss_pct is not None else "없음"
        offense_sl = f"{res.offense.stop_loss_pct:.1f}%" if res.offense.stop_loss_pct is not None else "없음"

        row = {
            "순위": idx,
            "모드 전환": res.mode_switch_strategy.upper(),
            "Defense 조건": (
                f"조건 {res.defense.buy_cond_pct:.1f}% / TP {res.defense.tp_pct:.1f}% / "
                f"보유 {res.defense.max_hold_days}일 / 분할 {res.defense.slices} / SL {defense_sl}"
            ),
            "Offense 조건": (
                f"조건 {res.offense.buy_cond_pct:.1f}% / TP {res.offense.tp_pct:.1f}% / "
                f"보유 {res.offense.max_hold_days}일 / 분할 {res.offense.slices} / SL {offense_sl}"
            ),
            "자금 관리": f"초기자금 {res.capital.initial_cash:,.0f}",
            "Calmar": round(res.score, 4),
            "Train CAGR(%)": round(res.train_metrics.get("CAGR", 0.0) * 100, 2),
            "Train MDD(%)": round(res.train_metrics.get("Max Drawdown", 0.0) * 100, 2),
            "Test CAGR(%)": round(res.test_metrics.get("CAGR", 0.0) * 100, 2),
            "Test MDD(%)": round(res.test_metrics.get("Max Drawdown", 0.0) * 100, 2),
            "Combined CAGR(%)": round(res.combined_metrics.get("CAGR", 0.0) * 100, 2),
            "Combined MDD(%)": round(res.combined_metrics.get("Max Drawdown", 0.0) * 100, 2),
            "Combined Calmar": round(res.combined_metrics.get("Calmar Ratio", 0.0), 2),
        }

        if res.rsi_thresholds:
            row["RSI Thresholds"] = (
                f"L{res.rsi_thresholds['rsi_low_threshold']:.0f} / "
                f"ML{res.rsi_thresholds['rsi_mid_low']:.0f} / "
                f"N{res.rsi_thresholds['rsi_neutral']:.0f} / "
                f"MH{res.rsi_thresholds['rsi_mid_high']:.0f} / "
                f"H{res.rsi_thresholds['rsi_high_threshold']:.0f}"
            )
        if res.ma_periods:
            row["MA Periods"] = (
                f"Short {res.ma_periods['ma_short_period']}주 / "
                f"Long {res.ma_periods['ma_long_period']}주"
            )
        if res.roc_period:
            row["ROC Period"] = f"{res.roc_period['roc_period']}주"
        if res.cash_limited_buy:
            row["현금한도매수"] = "ON"

        rows.append(row)

    return pd.DataFrame(rows)


def get_history_df(study: optuna.Study) -> pd.DataFrame:
    """Extract optimization history as a DataFrame for charting."""
    rows = []
    best_so_far = float("-inf")
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        score = trial.user_attrs.get("score", float("-inf"))
        if score > best_so_far:
            best_so_far = score
        train_m = trial.user_attrs.get("train_metrics", {})
        test_m = trial.user_attrs.get("test_metrics", {})
        avg_cagr = (train_m.get("CAGR", 0.0) + test_m.get("CAGR", 0.0)) / 2
        avg_mdd = (abs(train_m.get("Max Drawdown", 0.0)) + abs(test_m.get("Max Drawdown", 0.0))) / 2
        rows.append({
            "Trial": trial.number,
            "Calmar": score,
            "CAGR": avg_cagr * 100,
            "MDD": avg_mdd * 100,
            "Best Calmar": best_so_far,
        })
    return pd.DataFrame(rows)


# --------------------------- Phase 2 helper ---------------------------

def _range_float(vals, margin=0.3, clamp_max=None):
    lo, hi = min(vals), max(vals)
    spread = max(hi - lo, 0.5) * margin
    result_lo = round(max(0.0, lo - spread), 2)
    result_hi = round(hi + spread, 2)
    if clamp_max is not None:
        result_hi = min(result_hi, clamp_max)
    return (result_lo, result_hi)


def _range_int(vals, margin=0.3, clamp_min=1):
    lo, hi = min(vals), max(vals)
    spread = max(int((hi - lo) * margin), 2)
    return (max(clamp_min, lo - spread), hi + spread)


def narrow_config(cfg: OptunaConfig, results: list[OptimizationResult], phase2_trials: int | None = None) -> OptunaConfig:
    """Build a focused Phase 2 config by narrowing ranges around top 5 results."""
    top5 = results[:5]
    if not top5:
        return cfg

    # Determine dominant mode strategy
    mode_counts: dict[str, int] = {}
    for r in top5:
        mode_counts[r.mode_switch_strategy] = mode_counts.get(r.mode_switch_strategy, 0) + 1
    dominant_mode = max(mode_counts, key=mode_counts.get)

    narrowed = OptunaConfig(
        target_ticker=cfg.target_ticker,
        momentum_ticker=cfg.momentum_ticker,
        initial_cash=cfg.initial_cash,
        train_ranges=cfg.train_ranges,
        test_range=cfg.test_range,
        rsi_period=cfg.rsi_period,
        enable_netting=cfg.enable_netting,
        n_trials=phase2_trials if phase2_trials else cfg.n_trials * 3,
        top_n=cfg.top_n,
        mode_switch_strategy=dominant_mode,
        optimize_rsi_thresholds=(dominant_mode == "rsi"),
        optimize_ma_periods=(dominant_mode == "ma_cross"),
        optimize_roc_period=(dominant_mode == "roc"),
        optimize_cash_limited_buy=cfg.optimize_cash_limited_buy,
        constraints=cfg.constraints,
        progress_callback=cfg.progress_callback,
        # Narrowed ranges
        def_buy_range=_range_float([r.defense.buy_cond_pct for r in top5]),
        def_tp_range=_range_float([r.defense.tp_pct for r in top5]),
        def_hold_range=_range_int([r.defense.max_hold_days for r in top5]),
        def_slices_range=_range_int([r.defense.slices for r in top5]),
        def_sl_range=_range_float([r.defense.stop_loss_pct or 0 for r in top5]),
        off_buy_range=_range_float([r.offense.buy_cond_pct for r in top5]),
        off_tp_range=_range_float([r.offense.tp_pct for r in top5]),
        off_hold_range=_range_int([r.offense.max_hold_days for r in top5]),
        off_slices_range=_range_int([r.offense.slices for r in top5]),
        off_sl_range=_range_float([r.offense.stop_loss_pct or 0 for r in top5]),
    )

    # Narrow RSI thresholds
    if dominant_mode == "rsi":
        rsi_results = [r for r in top5 if r.rsi_thresholds]
        if rsi_results:
            narrowed.rsi_low_range = _range_float([r.rsi_thresholds["rsi_low_threshold"] for r in rsi_results])
            narrowed.rsi_mid_low_range = _range_float([r.rsi_thresholds["rsi_mid_low"] for r in rsi_results])
            narrowed.rsi_neutral_range = _range_float([r.rsi_thresholds["rsi_neutral"] for r in rsi_results])
            narrowed.rsi_mid_high_range = _range_float([r.rsi_thresholds["rsi_mid_high"] for r in rsi_results])
            narrowed.rsi_high_range = _range_float([r.rsi_thresholds["rsi_high_threshold"] for r in rsi_results])

    # Narrow MA periods
    if dominant_mode == "ma_cross":
        ma_results = [r for r in top5 if r.ma_periods]
        if ma_results:
            narrowed.ma_short_range = _range_int([r.ma_periods["ma_short_period"] for r in ma_results])
            narrowed.ma_long_range = _range_int([r.ma_periods["ma_long_period"] for r in ma_results])

    # Narrow ROC period
    if dominant_mode == "roc":
        roc_results = [r for r in top5 if r.roc_period]
        if roc_results:
            narrowed.roc_period_range = _range_int([r.roc_period["roc_period"] for r in roc_results], clamp_min=2)

    return narrowed


# --------------------------- Config export helpers ---------------------------

def result_to_config_dict(res: OptimizationResult) -> dict:
    """Convert an OptimizationResult to a config dictionary."""
    config = {
        "defense_slices": res.defense.slices,
        "defense_buy": round(res.defense.buy_cond_pct, 2),
        "defense_tp": round(res.defense.tp_pct, 2),
        "defense_sl": round(res.defense.stop_loss_pct, 1) if res.defense.stop_loss_pct else 0.0,
        "defense_hold": res.defense.max_hold_days,
        "offense_slices": res.offense.slices,
        "offense_buy": round(res.offense.buy_cond_pct, 2),
        "offense_tp": round(res.offense.tp_pct, 2),
        "offense_sl": round(res.offense.stop_loss_pct, 1) if res.offense.stop_loss_pct else 0.0,
        "offense_hold": res.offense.max_hold_days,
        "mode_switch_strategy_index": {"rsi": 0, "ma_cross": 1, "roc": 2}.get(res.mode_switch_strategy, 0),
        "cash_limited_buy": res.cash_limited_buy,
    }
    if res.rsi_thresholds:
        config.update({
            "rsi_high_threshold": round(res.rsi_thresholds["rsi_high_threshold"], 1),
            "rsi_mid_high": round(res.rsi_thresholds["rsi_mid_high"], 1),
            "rsi_neutral": round(res.rsi_thresholds["rsi_neutral"], 1),
            "rsi_mid_low": round(res.rsi_thresholds["rsi_mid_low"], 1),
            "rsi_low_threshold": round(res.rsi_thresholds["rsi_low_threshold"], 1),
        })
    if res.ma_periods:
        config.update({
            "ma_short": res.ma_periods.get("ma_short_period", 3),
            "ma_long": res.ma_periods.get("ma_long_period", 7),
        })
    if res.roc_period:
        config["roc_period"] = res.roc_period.get("roc_period", 4)
    return config


def apply_to_config(res: OptimizationResult, config_path: str = "config/order_book_settings.json") -> None:
    """Apply best result to order_book_settings.json."""
    path = Path(config_path)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    else:
        current = {}
    current.update(result_to_config_dict(res))
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_result_as_config(res: OptimizationResult, filename: str) -> Path:
    """Save a single OptimizationResult as a named config JSON file."""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    if not filename.endswith(".json"):
        filename += ".json"
    path = config_dir / filename
    config = result_to_config_dict(res)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# --------------------------- CLI entry point ---------------------------

if __name__ == "__main__":  # pragma: no cover
    default_cfg = OptunaConfig(
        target_ticker="SOXL",
        momentum_ticker="QQQ",
        n_trials=300,
    )
    try:
        study, frames = run_optuna(default_cfg)
        results = extract_results(study, default_cfg, top_n=5, price_frames=frames)
        print(f"Evaluated {len(study.trials)} trials")
        if results:
            print("\nTop 3 results:")
            for i, res in enumerate(results[:3], 1):
                print(
                    f"  {i}. Score={res.score:.4f}, "
                    f"Test MDD={res.test_metrics.get('Max Drawdown', 0):.2%}, "
                    f"Test CAGR={res.test_metrics.get('CAGR', 0):.2%}"
                )
    except Exception as exc:
        print(f"Optuna optimization failed: {exc}")
