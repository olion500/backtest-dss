"""Compare old vs optimized strategy parameters for 2025-01-01 to today."""
from datetime import datetime
import pandas as pd
import yfinance as yf
from dongpa_engine import (
    CapitalParams,
    DongpaBacktester,
    ModeParams,
    StrategyParams,
    summarize,
)


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download price data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


def run_backtest(params: StrategyParams, capital: CapitalParams, target_df: pd.DataFrame, momo_df: pd.DataFrame) -> dict:
    """Run backtest and return results."""
    backtester = DongpaBacktester(target_df, momo_df, params, capital)
    results = backtester.run()
    metrics = summarize(results["equity"])
    return {
        "results": results,
        "metrics": metrics,
        "equity": results["equity"],
        "journal": results["journal"]
    }


def main():
    # Date range
    start_date = "2025-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Comparing strategies from {start_date} to {end_date}")
    print("=" * 70)

    # Download data
    print("\nDownloading data...")
    target_df = download_data("SOXL", start_date, end_date)
    momo_df = download_data("QQQ", start_date, end_date)

    # CURRENT CONFIG PARAMETERS (from config/order_book_settings.json)
    old_defense = ModeParams(
        buy_cond_pct=1.7,
        tp_pct=1.4,
        max_hold_days=70,
        slices=7,
        stop_loss_pct=None  # No stop loss!
    )
    old_offense = ModeParams(
        buy_cond_pct=9.9,
        tp_pct=8.6,
        max_hold_days=9,
        slices=3,
        stop_loss_pct=None  # No stop loss!
    )
    old_capital = CapitalParams(
        initial_cash=10000.0,
        refresh_cycle_days=5,
        profit_compound_rate=0.95,
        loss_compound_rate=0.48,
        slippage_pct=0.0
    )

    # OPTIMIZED PARAMETERS (from config/order_book_settings.json - Rank #1)
    new_defense = ModeParams(
        buy_cond_pct=2.6,
        tp_pct=1.1,
        max_hold_days=73,
        slices=6,
        stop_loss_pct=18.8
    )
    new_offense = ModeParams(
        buy_cond_pct=4.5,
        tp_pct=6.8,
        max_hold_days=51,
        slices=3,
        stop_loss_pct=23.2
    )
    new_capital = CapitalParams(
        initial_cash=10000.0,
        refresh_cycle_days=1,
        profit_compound_rate=0.88,
        loss_compound_rate=0.11,
        slippage_pct=0.0
    )

    # Run backtests
    print("\nRunning backtest with OLD parameters...")
    old_params = StrategyParams(
        target_ticker="SOXL",
        momentum_ticker="QQQ",
        benchmark_ticker="SOXX",
        rsi_period=14,
        reset_on_mode_change=True,
        enable_netting=True,
        defense=old_defense,
        offense=old_offense,
    )
    old_result = run_backtest(old_params, old_capital, target_df, momo_df)

    print("Running backtest with OPTIMIZED parameters...")
    new_params = StrategyParams(
        target_ticker="SOXL",
        momentum_ticker="QQQ",
        benchmark_ticker="SOXX",
        rsi_period=14,
        reset_on_mode_change=True,
        enable_netting=True,
        defense=new_defense,
        offense=new_offense,
    )
    new_result = run_backtest(new_params, new_capital, target_df, momo_df)

    # Display comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS (2025-01-01 to today)")
    print("=" * 70)

    print("\n📊 CURRENT CONFIG (config/order_book_settings.json):")
    old_def_sl = f"{old_defense.stop_loss_pct}%" if old_defense.stop_loss_pct else "없음"
    old_off_sl = f"{old_offense.stop_loss_pct}%" if old_offense.stop_loss_pct else "없음"
    print(f"  Defense: buy={old_defense.buy_cond_pct}%, tp={old_defense.tp_pct}%, hold={old_defense.max_hold_days}d, N={old_defense.slices}, SL={old_def_sl}")
    print(f"  Offense: buy={old_offense.buy_cond_pct}%, tp={old_offense.tp_pct}%, hold={old_offense.max_hold_days}d, N={old_offense.slices}, SL={old_off_sl}")
    print(f"  Capital: cycle={old_capital.refresh_cycle_days}d, PCR={old_capital.profit_compound_rate}, LCR={old_capital.loss_compound_rate}")

    print("\n🚀 OPTIMIZED PARAMETERS:")
    print(f"  Defense: buy={new_defense.buy_cond_pct}%, tp={new_defense.tp_pct}%, hold={new_defense.max_hold_days}d, N={new_defense.slices}, SL={new_defense.stop_loss_pct}%")
    print(f"  Offense: buy={new_offense.buy_cond_pct}%, tp={new_offense.tp_pct}%, hold={new_offense.max_hold_days}d, N={new_offense.slices}, SL={new_offense.stop_loss_pct}%")
    print(f"  Capital: cycle={new_capital.refresh_cycle_days}d, PCR={new_capital.profit_compound_rate}, LCR={new_capital.loss_compound_rate}")

    print("\n" + "=" * 70)
    print(f"{'Metric':<25} {'CURRENT':<20} {'OPTIMIZED':<20} {'Change':<15}")
    print("=" * 70)

    metrics_to_compare = [
        ("Final Equity", "$"),
        ("CAGR", "%"),
        ("Max Drawdown", "%"),
        ("Sharpe (rf=0)", ""),
        ("Win Rate", "%"),
        ("Total Trades", ""),
    ]

    for metric, unit in metrics_to_compare:
        old_val = old_result["metrics"].get(metric, 0)
        new_val = new_result["metrics"].get(metric, 0)

        if unit == "%":
            old_str = f"{old_val:.2f}%"
            new_str = f"{new_val:.2f}%"
            change = new_val - old_val
            change_str = f"{change:+.2f}%"
        elif unit == "$":
            old_str = f"${old_val:,.2f}"
            new_str = f"${new_val:,.2f}"
            change = ((new_val / old_val - 1) * 100) if old_val != 0 else 0
            change_str = f"{change:+.1f}%"
        else:
            old_str = f"{old_val:.2f}"
            new_str = f"{new_val:.2f}"
            change = new_val - old_val
            change_str = f"{change:+.2f}"

        print(f"{metric:<25} {old_str:<20} {new_str:<20} {change_str:<15}")

    print("=" * 70)

    # Summary
    print("\n📈 SUMMARY:")
    old_cagr = old_result["metrics"].get("CAGR", 0)
    new_cagr = new_result["metrics"].get("CAGR", 0)
    old_mdd = old_result["metrics"].get("Max Drawdown", 0)
    new_mdd = new_result["metrics"].get("Max Drawdown", 0)

    print(f"  CAGR: {old_cagr:.2f}% → {new_cagr:.2f}% ({new_cagr - old_cagr:+.2f}%)")
    print(f"  MDD:  {old_mdd:.2f}% → {new_mdd:.2f}% ({new_mdd - old_mdd:+.2f}%)")

    if new_cagr > old_cagr and abs(new_mdd) < abs(old_mdd):
        print("\n✅ OPTIMIZED strategy is BETTER: Higher returns AND lower drawdown!")
    elif new_cagr > old_cagr:
        print("\n⚠️  OPTIMIZED strategy has higher returns but also higher drawdown")
    elif abs(new_mdd) < abs(old_mdd):
        print("\n⚠️  OPTIMIZED strategy has lower drawdown but also lower returns")
    else:
        print("\n❌ CURRENT CONFIG performed better in this period")

    print("\n" + "=" * 70)

    # Save detailed results
    print("\n💾 Trade details:")
    print(f"  CURRENT CONFIG trades: {len(old_result['journal'])}")
    print(f"  OPTIMIZED trades: {len(new_result['journal'])}")

    if len(old_result['journal']) > 0:
        print(f"\n  Last 5 CURRENT CONFIG trades:")
        print(old_result['journal'].tail(5).to_string())

    if len(new_result['journal']) > 0:
        print(f"\n  Last 5 OPTIMIZED trades:")
        print(new_result['journal'].tail(5).to_string())

    # Try to save CSVs
    try:
        import os
        os.makedirs("outputs", exist_ok=True)
        old_result["journal"].to_csv("outputs/comparison_current_journal.csv", index=False)
        new_result["journal"].to_csv("outputs/comparison_optimized_journal.csv", index=False)
        print("\n  Saved to:")
        print("  - outputs/comparison_current_journal.csv")
        print("  - outputs/comparison_optimized_journal.csv")
    except Exception as e:
        print(f"\n  Could not save CSVs: {e}")


if __name__ == "__main__":
    main()
