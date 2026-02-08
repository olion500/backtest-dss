"""Compare current config vs optimized parameters across multiple years."""
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
    years = [
        ("2022", "2022-01-01", "2022-12-31"),
        ("2023", "2023-01-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
    ]

    print("=" * 80)
    print("YEAR-BY-YEAR COMPARISON: CURRENT CONFIG vs OPTIMIZED")
    print("=" * 80)

    # CURRENT CONFIG PARAMETERS (from config/order_book_settings.json)
    current_defense = ModeParams(
        buy_cond_pct=1.7,
        tp_pct=1.4,
        max_hold_days=70,
        slices=7,
        stop_loss_pct=None  # No stop loss!
    )
    current_offense = ModeParams(
        buy_cond_pct=9.9,
        tp_pct=8.6,
        max_hold_days=9,
        slices=3,
        stop_loss_pct=None  # No stop loss!
    )
    current_capital = CapitalParams(
        initial_cash=10000.0,
        slippage_pct=0.0
    )

    # OPTIMIZED PARAMETERS (Rank #1 from optimizer)
    opt_defense = ModeParams(
        buy_cond_pct=2.6,
        tp_pct=1.1,
        max_hold_days=73,
        slices=6,
        stop_loss_pct=18.8
    )
    opt_offense = ModeParams(
        buy_cond_pct=4.5,
        tp_pct=6.8,
        max_hold_days=51,
        slices=3,
        stop_loss_pct=23.2
    )
    opt_capital = CapitalParams(
        initial_cash=10000.0,
        slippage_pct=0.0
    )

    print("\n📊 CURRENT CONFIG:")
    print(f"  Defense: buy={current_defense.buy_cond_pct}%, tp={current_defense.tp_pct}%, hold={current_defense.max_hold_days}d, N={current_defense.slices}, SL=없음")
    print(f"  Offense: buy={current_offense.buy_cond_pct}%, tp={current_offense.tp_pct}%, hold={current_offense.max_hold_days}d, N={current_offense.slices}, SL=없음")
    print(f"  Capital: initial_cash={current_capital.initial_cash}")

    print("\n🚀 OPTIMIZED:")
    print(f"  Defense: buy={opt_defense.buy_cond_pct}%, tp={opt_defense.tp_pct}%, hold={opt_defense.max_hold_days}d, N={opt_defense.slices}, SL={opt_defense.stop_loss_pct}%")
    print(f"  Offense: buy={opt_offense.buy_cond_pct}%, tp={opt_offense.tp_pct}%, hold={opt_offense.max_hold_days}d, N={opt_offense.slices}, SL={opt_offense.stop_loss_pct}%")
    print(f"  Capital: initial_cash={opt_capital.initial_cash}")

    print("\n" + "=" * 80)

    all_results = []

    for year_name, start_date, end_date in years:
        print(f"\n{'=' * 80}")
        print(f"Testing {year_name}: {start_date} to {end_date}")
        print(f"{'=' * 80}")

        # Download data for this year
        print(f"Downloading data for {year_name}...")
        target_df = download_data("SOXL", start_date, end_date)
        momo_df = download_data("QQQ", start_date, end_date)

        if target_df.empty or momo_df.empty:
            print(f"⚠️  No data available for {year_name}, skipping...")
            continue

        # Current config
        print(f"Running CURRENT CONFIG for {year_name}...")
        current_params = StrategyParams(
            target_ticker="SOXL",
            momentum_ticker="QQQ",
            benchmark_ticker="SOXX",
            rsi_period=14,
            reset_on_mode_change=True,
            enable_netting=True,
            defense=current_defense,
            offense=current_offense,
        )
        current_result = run_backtest(current_params, current_capital, target_df, momo_df)

        # Optimized
        print(f"Running OPTIMIZED for {year_name}...")
        opt_params = StrategyParams(
            target_ticker="SOXL",
            momentum_ticker="QQQ",
            benchmark_ticker="SOXX",
            rsi_period=14,
            reset_on_mode_change=True,
            enable_netting=True,
            defense=opt_defense,
            offense=opt_offense,
        )
        opt_result = run_backtest(opt_params, opt_capital, target_df, momo_df)

        # Extract metrics
        curr_metrics = current_result["metrics"]
        opt_metrics = opt_result["metrics"]

        curr_equity = curr_metrics.get("Final Equity", 0)
        opt_equity = opt_metrics.get("Final Equity", 0)
        curr_cagr = curr_metrics.get("CAGR", 0)
        opt_cagr = opt_metrics.get("CAGR", 0)
        curr_mdd = curr_metrics.get("Max Drawdown", 0)
        opt_mdd = opt_metrics.get("Max Drawdown", 0)
        curr_sharpe = curr_metrics.get("Sharpe (rf=0)", 0)
        opt_sharpe = opt_metrics.get("Sharpe (rf=0)", 0)
        curr_trades = len(current_result["journal"])
        opt_trades = len(opt_result["journal"])

        # Print results
        print(f"\n{year_name} Results:")
        print(f"{'Metric':<20} {'CURRENT':<20} {'OPTIMIZED':<20} {'Change':<15}")
        print("-" * 75)
        print(f"{'Final Equity':<20} ${curr_equity:,.2f}       ${opt_equity:,.2f}       {((opt_equity/curr_equity-1)*100 if curr_equity > 0 else 0):+.1f}%")
        print(f"{'CAGR':<20} {curr_cagr:.2f}%            {opt_cagr:.2f}%            {(opt_cagr-curr_cagr):+.2f}%")
        print(f"{'Max Drawdown':<20} {curr_mdd:.2f}%            {opt_mdd:.2f}%            {(opt_mdd-curr_mdd):+.2f}%")
        print(f"{'Sharpe':<20} {curr_sharpe:.2f}              {opt_sharpe:.2f}              {(opt_sharpe-curr_sharpe):+.2f}")
        print(f"{'Total Trades':<20} {curr_trades:<16}  {opt_trades:<16}  {(opt_trades-curr_trades):+d}")

        # Determine winner
        if opt_cagr > curr_cagr and abs(opt_mdd) < abs(curr_mdd):
            verdict = "✅ OPTIMIZED wins (higher CAGR + lower MDD)"
        elif opt_cagr > curr_cagr:
            verdict = "⚠️  OPTIMIZED wins (higher CAGR but higher MDD)"
        elif abs(opt_mdd) < abs(curr_mdd):
            verdict = "⚠️  OPTIMIZED wins (lower MDD but lower CAGR)"
        else:
            verdict = "❌ CURRENT wins"

        print(f"\n{verdict}")

        # Store for summary
        all_results.append({
            "year": year_name,
            "curr_equity": curr_equity,
            "opt_equity": opt_equity,
            "curr_cagr": curr_cagr,
            "opt_cagr": opt_cagr,
            "curr_mdd": curr_mdd,
            "opt_mdd": opt_mdd,
            "curr_sharpe": curr_sharpe,
            "opt_sharpe": opt_sharpe,
            "verdict": verdict,
        })

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: CURRENT CONFIG vs OPTIMIZED (2022-2024)")
    print("=" * 80)
    print(f"\n{'Year':<8} {'CURRENT':<25} {'OPTIMIZED':<25} {'Winner':<30}")
    print("-" * 88)

    for res in all_results:
        curr_str = f"${res['curr_equity']:,.0f} ({res['curr_cagr']:+.1f}%, {res['curr_mdd']:.1f}%)"
        opt_str = f"${res['opt_equity']:,.0f} ({res['opt_cagr']:+.1f}%, {res['opt_mdd']:.1f}%)"
        verdict_short = "✅ OPT" if "✅" in res['verdict'] else ("⚠️ OPT" if "⚠️" in res['verdict'] else "❌ CURR")
        print(f"{res['year']:<8} {curr_str:<25} {opt_str:<25} {verdict_short:<30}")

    # Overall conclusion
    print("\n" + "=" * 80)
    print("🎯 OVERALL CONCLUSION:")
    print("=" * 80)

    opt_wins = sum(1 for res in all_results if "✅" in res['verdict'] or "⚠️" in res['verdict'])
    total_years = len(all_results)

    if opt_wins == total_years:
        print(f"\n✅ OPTIMIZED strategy won in ALL {total_years} years!")
    elif opt_wins > total_years / 2:
        print(f"\n✅ OPTIMIZED strategy won in {opt_wins}/{total_years} years")
    else:
        print(f"\n❌ CURRENT CONFIG won in {total_years - opt_wins}/{total_years} years")

    # Calculate averages
    avg_curr_cagr = sum(r['curr_cagr'] for r in all_results) / len(all_results)
    avg_opt_cagr = sum(r['opt_cagr'] for r in all_results) / len(all_results)
    avg_curr_mdd = sum(r['curr_mdd'] for r in all_results) / len(all_results)
    avg_opt_mdd = sum(r['opt_mdd'] for r in all_results) / len(all_results)

    print(f"\n📊 3-Year Averages:")
    print(f"  CAGR: {avg_curr_cagr:.2f}% (CURRENT) vs {avg_opt_cagr:.2f}% (OPTIMIZED) → {avg_opt_cagr - avg_curr_cagr:+.2f}%")
    print(f"  MDD:  {avg_curr_mdd:.2f}% (CURRENT) vs {avg_opt_mdd:.2f}% (OPTIMIZED) → {avg_opt_mdd - avg_curr_mdd:+.2f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
