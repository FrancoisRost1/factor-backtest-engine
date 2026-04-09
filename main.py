"""
Factor Model + Backtesting Engine — Main Orchestrator

This file orchestrates the full pipeline. No financial logic lives here.
Each step calls a dedicated module in factor_engine/.

Pipeline:
    1. Load universe (S&P 500 tickers + SPY benchmark)
    2. Fetch price and fundamental data via yfinance
    3. Run all backtests (72 combinations)
    4. Compute analytics for each backtest
    5. Export results

Usage:
    python main.py
"""

import numpy as np
import yaml
from pathlib import Path

from factor_engine.universe import get_sp500_tickers, get_benchmark_ticker
from factor_engine.data_loader import (
    fetch_price_history,
    fetch_fundamentals,
    fetch_benchmark_prices,
)
from factor_engine.backtest import run_all_backtests
from factor_engine.analytics import compute_all_metrics
from factor_engine.ic import ic_summary
from factor_engine.regression import regress_vs_benchmark


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(Path(__file__).parent / path, "r") as f:
        return yaml.safe_load(f)


def get_periods_per_year(frequency: str) -> int:
    """Map rebalancing frequency to periods per year."""
    return {"monthly": 12, "quarterly": 4, "annual": 1}[frequency]


def run():
    """Execute the full factor model + backtesting pipeline."""
    config = load_config()

    # Step 1: Load universe
    print("Step 1: Loading S&P 500 universe...")
    tickers = get_sp500_tickers()
    benchmark_ticker = get_benchmark_ticker()
    print(f"  Universe: {len(tickers)} tickers + {benchmark_ticker} benchmark")

    # Step 2: Fetch data
    print("Step 2: Fetching data...")
    prices = fetch_price_history(
        tickers,
        config["data"]["start_date"],
        config["data"]["end_date"],
    )
    fundamentals = fetch_fundamentals(tickers)
    benchmark_prices = fetch_benchmark_prices(
        benchmark_ticker,
        config["data"]["start_date"],
        config["data"]["end_date"],
    )

    # Step 3: Run all backtests
    from factor_engine.backtest import FACTOR_NAMES
    n_combos = (
        len(FACTOR_NAMES)
        * len(config["rebalancing"]["frequencies"])
        * len(config["portfolio"]["weighting_schemes"])
        * 2  # long_only + long_short
    )
    print(f"Step 3: Running backtests ({n_combos} combinations)...")
    all_results = run_all_backtests(prices, fundamentals, benchmark_prices, config)

    # Step 4: Compute analytics for each backtest
    print("Step 4: Computing analytics...")
    risk_free = config["analytics"]["risk_free_rate"]
    summary_rows = []

    for factor in all_results:
        for freq in all_results[factor]:
            ppy = get_periods_per_year(freq)
            for wt in all_results[factor][freq]:
                for pt in all_results[factor][freq][wt]:
                    bt = all_results[factor][freq][wt][pt]

                    if len(bt["returns_gross"]) == 0:
                        continue

                    # Performance metrics (gross)
                    metrics_gross = compute_all_metrics(
                        bt["returns_gross"],
                        bt["benchmark_returns"],
                        bt["turnover"],
                        periods_per_year=ppy,
                        risk_free=risk_free,
                    )
                    # Performance metrics (net)
                    metrics_net = compute_all_metrics(
                        bt["returns_net"],
                        bt["benchmark_returns"],
                        bt["turnover"],
                        periods_per_year=ppy,
                        risk_free=risk_free,
                    )
                    # IC summary
                    ic_stats = ic_summary(bt["ic_series"])

                    # Regression vs benchmark
                    # regress_vs_benchmark returns raw per-period alpha;
                    # multiply by ppy so the stored value is annualised.
                    reg = regress_vs_benchmark(
                        bt["returns_gross"], bt["benchmark_returns"]
                    )
                    reg_alpha_ann = (
                        reg["alpha"] * ppy
                        if not np.isnan(reg["alpha"]) else np.nan
                    )

                    summary_rows.append({
                        "factor": factor,
                        "frequency": freq,
                        "weighting": wt,
                        "portfolio_type": pt,
                        # Gross
                        "return_gross": metrics_gross["annualized_return"],
                        "sharpe_gross": metrics_gross["sharpe"],
                        "sortino_gross": metrics_gross["sortino"],
                        "max_dd": metrics_gross["max_drawdown"],
                        "calmar": metrics_gross["calmar"],
                        "hit_rate": metrics_gross["hit_rate"],
                        # Net
                        "return_net": metrics_net["annualized_return"],
                        "sharpe_net": metrics_net["sharpe"],
                        # Turnover
                        "avg_turnover": metrics_gross["avg_turnover"],
                        # IC
                        "mean_ic": ic_stats["mean_ic"],
                        "ic_ir": ic_stats["ic_ir"],
                        # Regression (alpha is annualised: raw_per_period * ppy)
                        "alpha": reg_alpha_ann,
                        "beta": reg["beta"],
                        "r_squared": reg["r_squared"],
                    })

    # Step 5: Export
    print("Step 5: Exporting results...")
    import pandas as pd

    summary_df = pd.DataFrame(summary_rows)
    output_path = Path(__file__).parent / "outputs" / "backtest_results.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    print(f"  Total backtests: {len(summary_rows)}")
    print("\nDone.")


if __name__ == "__main__":
    run()
