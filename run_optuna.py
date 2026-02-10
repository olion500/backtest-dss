"""Run Optuna optimization in two phases: wide exploration then focused search.

Usage:
    python run_optuna.py                    # Default: 300 wide + 1000 focused
    python run_optuna.py --trials 500       # Custom wide trials (focused = wide * 3)
    python run_optuna.py --ticker TQQQ      # Different target ticker
    python run_optuna.py --penalty 0.5      # Lower MDD penalty (more aggressive)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dongpa_optuna import OptunaConfig, run_optuna, extract_results, narrow_config


def parse_args():
    p = argparse.ArgumentParser(description="Dongpa Optuna optimizer (2-phase)")
    p.add_argument("--ticker", default="SOXL", help="Target ticker (default: SOXL)")
    p.add_argument("--momentum", default="QQQ", help="Momentum ticker (default: QQQ)")
    p.add_argument("--benchmark", default="SOXX", help="Benchmark ticker (default: SOXX)")
    p.add_argument("--cash", type=float, default=10000.0, help="Initial cash (default: 10000)")
    p.add_argument("--trials", type=int, default=300, help="Phase 1 trials (default: 300)")
    p.add_argument("--penalty", type=float, default=0.7, help="MDD penalty weight for ranking (default: 0.7)")
    p.add_argument("--top", type=int, default=10, help="Top N results to show (default: 10)")
    p.add_argument("--train-start", default="2020-01-01", help="Train start date")
    p.add_argument("--train-end", default="2022-12-31", help="Train end date")
    p.add_argument("--test-start", default="2023-01-01", help="Test start date")
    p.add_argument("--test-end", default="2025-12-31", help="Test end date")
    p.add_argument("--skip-phase2", action="store_true", help="Skip focused phase 2")
    p.add_argument("--apply", action="store_true", help="Apply best result to config")
    return p.parse_args()


def print_results(results, phase_label):
    print(f"\n{'='*80}")
    print(f"  {phase_label} - Top {len(results)} Results")
    print(f"{'='*80}")

    for i, res in enumerate(results, 1):
        d = res.defense
        o = res.offense
        c = res.capital
        d_sl = f"{d.stop_loss_pct:.1f}%" if d.stop_loss_pct else "없음"
        o_sl = f"{o.stop_loss_pct:.1f}%" if o.stop_loss_pct else "없음"

        print(f"\n--- #{i} | Score: {res.score:.4f} ---")
        print(f"  Defense: buy {d.buy_cond_pct:.2f}%, TP {d.tp_pct:.2f}%, hold {d.max_hold_days}d, slices {d.slices}, SL {d_sl}")
        print(f"  Offense: buy {o.buy_cond_pct:.2f}%, TP {o.tp_pct:.2f}%, hold {o.max_hold_days}d, slices {o.slices}, SL {o_sl}")
        print(f"  Capital: initial_cash {c.initial_cash:,.0f}")
        if res.rsi_thresholds:
            t = res.rsi_thresholds
            print(f"  RSI: L{t['rsi_low_threshold']:.1f}/ML{t['rsi_mid_low']:.1f}/N{t['rsi_neutral']:.1f}/MH{t['rsi_mid_high']:.1f}/H{t['rsi_high_threshold']:.1f}")
        if res.ma_periods:
            print(f"  MA: Short {res.ma_periods['ma_short_period']}w, Long {res.ma_periods['ma_long_period']}w")
        print(f"  Train:    CAGR {res.train_metrics.get('CAGR',0)*100:.2f}%, MDD {res.train_metrics.get('Max Drawdown',0)*100:.2f}%")
        print(f"  Test:     CAGR {res.test_metrics.get('CAGR',0)*100:.2f}%, MDD {res.test_metrics.get('Max Drawdown',0)*100:.2f}%")
        print(f"  Combined: CAGR {res.combined_metrics.get('CAGR',0)*100:.2f}%, MDD {res.combined_metrics.get('Max Drawdown',0)*100:.2f}%, Calmar {res.combined_metrics.get('Calmar Ratio',0):.2f}")


def apply_to_config(res, config_path="config/order_book_settings.json"):
    """Apply best result to order_book_settings.json, backing up first."""
    path = Path(config_path)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    else:
        current = {}

    current.update({
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
        "mode_switch_strategy_index": 0 if res.mode_switch_strategy == "rsi" else 1,
    })

    if res.rsi_thresholds:
        current.update({
            "rsi_high_threshold": round(res.rsi_thresholds["rsi_high_threshold"], 1),
            "rsi_mid_high": round(res.rsi_thresholds["rsi_mid_high"], 1),
            "rsi_neutral": round(res.rsi_thresholds["rsi_neutral"], 1),
            "rsi_mid_low": round(res.rsi_thresholds["rsi_mid_low"], 1),
            "rsi_low_threshold": round(res.rsi_thresholds["rsi_low_threshold"], 1),
        })
    if res.ma_periods:
        current.update({
            "ma_short": res.ma_periods.get("ma_short_period", 3),
            "ma_long": res.ma_periods.get("ma_long_period", 7),
        })

    path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nConfig updated: {config_path}")


def main():
    args = parse_args()
    train_ranges = [(args.train_start, args.train_end)]
    test_range = (args.test_start, args.test_end)

    # ── Phase 1: Wide exploration ──
    print(f"Phase 1: Wide exploration ({args.trials} trials)")
    print(f"  Target: {args.ticker}, Momentum: {args.momentum}")
    print(f"  Train: {args.train_start} ~ {args.train_end}")
    print(f"  Test:  {args.test_start} ~ {args.test_end}")

    cfg1 = OptunaConfig(
        target_ticker=args.ticker,
        momentum_ticker=args.momentum,
        benchmark_ticker=args.benchmark,
        initial_cash=args.cash,
        train_ranges=train_ranges,
        test_range=test_range,
        n_trials=args.trials,
        score_penalty=args.penalty,
        top_n=args.top,
        mode_switch_strategy="both",
        optimize_rsi_thresholds=True,
        optimize_ma_periods=True,
        # Wide ranges
        def_buy_range=(0.5, 10.0),
        def_tp_range=(0.1, 5.0),
        def_hold_range=(5, 90),
        def_slices_range=(2, 15),
        def_sl_range=(2.0, 50.0),
        off_buy_range=(1.0, 15.0),
        off_tp_range=(0.5, 10.0),
        off_hold_range=(2, 70),
        off_slices_range=(2, 10),
        off_sl_range=(2.0, 50.0),
        ma_short_range=(2, 10),
        ma_long_range=(5, 30),
    )

    study1 = run_optuna(cfg1)
    results1 = extract_results(study1, cfg1, top_n=args.top)
    print_results(results1, "Phase 1 (Wide)")

    if args.skip_phase2 or not results1:
        best = results1[0] if results1 else None
        if best and args.apply:
            apply_to_config(best)
        return

    # ── Phase 2: Focused search ──
    phase2_trials = args.trials * 3
    cfg2 = narrow_config(cfg1, results1, phase2_trials=phase2_trials)
    print(f"\nPhase 2: Focused search ({phase2_trials} trials, mode={cfg2.mode_switch_strategy})")

    study2 = run_optuna(cfg2)
    results2 = extract_results(study2, cfg2, top_n=args.top)
    print_results(results2, "Phase 2 (Focused)")

    # Summary
    best1 = results1[0]
    best2 = results2[0] if results2 else None
    print(f"\n{'='*80}")
    print("  Summary")
    print(f"{'='*80}")
    print(f"  Phase 1 Best: Score {best1.score:.4f}")
    if best2:
        print(f"  Phase 2 Best: Score {best2.score:.4f}")
        improvement = ((best2.score - best1.score) / abs(best1.score)) * 100
        print(f"  Improvement:  {improvement:+.1f}%")

    best = best2 if (best2 and best2.score >= best1.score) else best1
    if args.apply:
        apply_to_config(best)


if __name__ == "__main__":
    main()
