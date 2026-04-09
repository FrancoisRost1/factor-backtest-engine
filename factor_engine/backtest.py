"""
Core backtesting engine: runs all 60 factor × frequency × weighting combinations.

For each combination the engine:
  1. Generates rebalancing dates (monthly / quarterly / annual)
  2. Computes cross-sectional factor scores at each rebalance date (no lookahead)
  3. Normalises scores to percentile ranks
  4. Assigns quintiles and constructs long-only and long-short portfolios
  5. Measures portfolio return over the subsequent holding period
  6. Computes IC (Spearman correlation of factor score vs forward returns)
  7. Applies transaction costs to derive net returns
  8. Records benchmark returns over the same holding period

Result structure:
  {factor: {frequency: {weighting: {portfolio_type: backtest_result}}}}

Where backtest_result is a dict with keys:
  returns_gross  — pd.Series of gross period returns, indexed by period end date
  returns_net    — pd.Series of net-of-cost period returns
  benchmark_returns — pd.Series of SPY period returns for the same dates
  turnover       — pd.Series of portfolio turnover per period
  ic_series      — pd.Series of IC values per period (NaN if too few stocks)

Simplifying assumptions (all documented in the project CLAUDE.md):
  - Survivorship bias: universe is today's S&P 500
  - Fundamentals are today's values, not point-in-time historical
  - Stocks with missing return data for a period contribute 0 return (held flat)
  - Rebalancing dates use BusinessMonthEnd / BusinessQuarterEnd / BusinessYearEnd
    which may not perfectly match actual NYSE trading days
"""

from typing import Tuple

import numpy as np
import pandas as pd

from factor_engine.factors import compute_all_factors
from factor_engine.normalize import percentile_rank
from factor_engine.quintiles import assign_quintiles, compute_quintile_returns
from factor_engine.portfolio import construct_long_only, construct_long_short
from factor_engine.transaction_costs import compute_turnover, apply_transaction_costs
from factor_engine.ic import compute_ic
from factor_engine.rebalance import get_rebalance_dates
from factor_engine.utils import get_period_returns

# ── Factor registry ────────────────────────────────────────────────────────────

FACTOR_NAMES = ["value", "momentum", "quality", "size", "low_volatility", "composite"]

# Maps factor name → column in compute_all_factors output, and whether to invert
# ranking (True = lower raw value is better, e.g. small cap, low volatility).
# composite_score is already an averaged percentile rank in (0,1]; invert=False.
FACTOR_CONFIG = {
    "value":          {"column": "earnings_yield",   "invert": False},
    "momentum":       {"column": "momentum_12_1",    "invert": False},
    "quality":        {"column": "roe",              "invert": False},
    "size":           {"column": "log_market_cap",   "invert": True},
    "low_volatility": {"column": "rolling_vol_60d",  "invert": True},
    "composite":      {"column": "composite_score",  "invert": False},
}


# ── Public entry point ─────────────────────────────────────────────────────────

def run_all_backtests(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    benchmark_prices: pd.Series,
    config: dict,
) -> dict:
    """
    Run all factor × frequency × weighting × portfolio_type backtests.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices.  Index = dates, columns = tickers.
    fundamentals : pd.DataFrame
        Cross-sectional fundamental data.  Index = tickers.
        Columns: pe_ratio, roe, market_cap.
    benchmark_prices : pd.Series
        Daily adjusted close prices for the benchmark (SPY).
    config : dict
        Full configuration dict loaded from config.yaml.

    Returns
    -------
    dict
        Nested dict: {factor: {frequency: {weighting: {portfolio_type: result}}}}
        where result is a dict with keys:
        returns_gross, returns_net, benchmark_returns, turnover, ic_series.
    """
    frequencies = config["rebalancing"]["frequencies"]
    weighting_schemes = config["portfolio"]["weighting_schemes"]
    cost_per_unit = config["transaction_costs"]["cost_per_trade"]
    n_quantiles = config["portfolio"]["n_quantiles"]
    lookback_days = (
        config.get("factors", {})
        .get("low_volatility", {})
        .get("lookback_days", 60)
    )
    min_valid_factors = config["composite"]["min_valid_factors"]
    annualization_factor = config["analytics"]["annualization_factor"]

    results = {}

    for factor in FACTOR_NAMES:
        results[factor] = {}
        for freq in frequencies:
            results[factor][freq] = {}
            # Use eval_start (not start_date) so rebalancing only begins after
            # the lookback warmup period. The 2014-2016 data is for computing
            # momentum and volatility inputs, not for generating trades.
            rebalance_dates = get_rebalance_dates(
                config["data"]["eval_start"],
                config["data"]["end_date"],
                freq,
            )
            if len(rebalance_dates) < 2:
                for wt in weighting_schemes:
                    results[factor][freq][wt] = {
                        "long_only": _empty_result(),
                        "long_short": _empty_result(),
                    }
                continue

            for wt in weighting_schemes:
                lo_result, ls_result = _run_single_factor_backtest(
                    factor_name=factor,
                    prices=prices,
                    fundamentals=fundamentals,
                    benchmark_prices=benchmark_prices,
                    rebalance_dates=rebalance_dates,
                    weighting=wt,
                    n_quantiles=n_quantiles,
                    cost_per_unit=cost_per_unit,
                    lookback_days=lookback_days,
                    min_valid_factors=min_valid_factors,
                    annualization_factor=annualization_factor,
                )
                results[factor][freq][wt] = {
                    "long_only": lo_result,
                    "long_short": ls_result,
                }

    return results


# ── Internal helpers ───────────────────────────────────────────────────────────

def _run_single_factor_backtest(
    factor_name: str,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    benchmark_prices: pd.Series,
    rebalance_dates: list,
    weighting: str,
    n_quantiles: int,
    cost_per_unit: float,
    lookback_days: int,
    min_valid_factors: int = 3,
    annualization_factor: int = 252,
) -> Tuple[dict, dict]:
    """
    Backtest one factor with one frequency and one weighting scheme.

    Returns a tuple (long_only_result, long_short_result) where each result
    is a dict with keys: returns_gross, returns_net, benchmark_returns,
    turnover, ic_series.

    Both portfolio types share the same factor scores, quintile assignments,
    and IC values — only the weight construction differs.

    Parameters
    ----------
    factor_name : str
        Key in FACTOR_CONFIG.
    prices : pd.DataFrame
        Full price history.
    fundamentals : pd.DataFrame
        Cross-sectional fundamentals.
    benchmark_prices : pd.Series
        Benchmark close prices.
    rebalance_dates : list[pd.Timestamp]
        Sorted list of rebalancing dates from get_rebalance_dates.
    weighting : str
        'equal' or 'cap_weight'.
    n_quantiles : int
        Number of quintiles (typically 5).
    cost_per_unit : float
        Transaction cost per unit of turnover (e.g., 0.001 = 10 bps).
    lookback_days : int
        Lookback window for rolling volatility computation.
    min_valid_factors : int
        Minimum valid factor scores to compute a composite (from config).
    annualization_factor : int
        Trading days per year for volatility annualisation (from config).
    """
    col = FACTOR_CONFIG[factor_name]["column"]
    invert = FACTOR_CONFIG[factor_name]["invert"]
    invert_cols = [col] if invert else []

    # Per-period accumulators
    lo_gross_list, lo_net_list, lo_to_list = [], [], []
    ls_gross_list, ls_net_list, ls_to_list = [], [], []
    ic_list = []
    bm_list = []
    qr_list = []   # per-period quintile returns (Q1 through Q5)
    dates = []

    prev_lo_weights = pd.Series(dtype=float)
    prev_ls_weights = pd.Series(dtype=float)

    for i in range(len(rebalance_dates) - 1):
        t_start = rebalance_dates[i]
        t_end = rebalance_dates[i + 1]

        # ── Factor scores at t_start (no lookahead) ────────────────────────
        factors_df = compute_all_factors(
            fundamentals, prices, t_start, lookback_days,
            min_valid_factors, annualization_factor,
        )
        if factors_df.empty or col not in factors_df.columns:
            continue

        factor_col = factors_df[[col]]  # single-column DataFrame

        # ── Normalise ──────────────────────────────────────────────────────
        ranked = percentile_rank(factor_col, invert_columns=invert_cols)
        scores = ranked[col].dropna()

        if len(scores) < n_quantiles:
            continue

        # ── Quintile assignment ────────────────────────────────────────────
        quintiles = assign_quintiles(scores, n_quintiles=n_quantiles)

        # ── Market caps for cap-weighting ──────────────────────────────────
        market_caps = None
        if weighting == "cap_weight":
            market_caps = fundamentals["market_cap"].reindex(scores.index)

        # ── Portfolio construction ─────────────────────────────────────────
        lo_weights = construct_long_only(
            quintiles, market_caps=market_caps, weighting=weighting
        )
        ls_weights = construct_long_short(
            quintiles, market_caps=market_caps, weighting=weighting
        )

        # ── Stock returns over the holding period ──────────────────────────
        stock_rets = get_period_returns(prices, t_start, t_end)

        # Portfolio gross returns.
        # *** MISSING-RETURN / STALE-PRICE WARNING ***
        # Stocks with NaN returns (delisted, M&A exits, data gaps) are treated
        # as 0 return via pandas .sum() NaN-skipping.  This is conservative for
        # bankruptcies (true return is -100%, not 0%) and ambiguous for M&A
        # (the acquirer may deliver a premium that is also excluded).  In a
        # survivorship-biased universe this case is rare, but if yfinance fails
        # to deliver a price for any stock in the holding period, its weight
        # effectively contributes 0 return rather than being excluded from the
        # denominator, which slightly understates positive-period returns.
        lo_gross = _portfolio_return(lo_weights, stock_rets)
        ls_gross = _portfolio_return(ls_weights, stock_rets)

        # ── Quintile returns (Q1–Q5 equal-weighted mean) ──────────────────
        qr = compute_quintile_returns(quintiles, stock_rets, n_quantiles)

        # ── IC: factor score vs forward stock returns ──────────────────────
        ic_val = compute_ic(scores, stock_rets)

        # ── Benchmark return ───────────────────────────────────────────────
        bm_ret = _benchmark_period_return(benchmark_prices, t_start, t_end)

        # ── Turnover (vs previous weights) ────────────────────────────────
        lo_to = compute_turnover(prev_lo_weights, lo_weights)
        ls_to = compute_turnover(prev_ls_weights, ls_weights)

        # ── Net returns after transaction costs ────────────────────────────
        lo_net = apply_transaction_costs(lo_gross, lo_to, cost_per_unit)
        ls_net = apply_transaction_costs(ls_gross, ls_to, cost_per_unit)

        # ── Accumulate ─────────────────────────────────────────────────────
        lo_gross_list.append(lo_gross)
        lo_net_list.append(lo_net)
        lo_to_list.append(lo_to)
        ls_gross_list.append(ls_gross)
        ls_net_list.append(ls_net)
        ls_to_list.append(ls_to)
        ic_list.append(ic_val)
        bm_list.append(bm_ret)
        qr_list.append(qr)
        dates.append(t_end)

        prev_lo_weights = lo_weights
        prev_ls_weights = ls_weights

    # ── Package results ────────────────────────────────────────────────────
    idx = pd.DatetimeIndex(dates)
    bm_series = pd.Series(bm_list, index=idx, dtype=float)
    ic_series = pd.Series(ic_list, index=idx, dtype=float)
    # quintile_returns: DataFrame (dates × quintiles 1–5); shows factor monotonicity
    quintile_df = pd.DataFrame(qr_list, index=idx) if qr_list else pd.DataFrame()

    lo_gross = pd.Series(lo_gross_list, index=idx, dtype=float)
    lo_net = pd.Series(lo_net_list, index=idx, dtype=float)
    ls_gross = pd.Series(ls_gross_list, index=idx, dtype=float)
    ls_net = pd.Series(ls_net_list, index=idx, dtype=float)

    lo_result = {
        "returns_gross":    lo_gross,
        "returns_net":      lo_net,
        "cumulative_gross": (1 + lo_gross).cumprod(),
        "cumulative_net":   (1 + lo_net).cumprod(),
        "benchmark_returns": bm_series,
        "turnover":         pd.Series(lo_to_list, index=idx, dtype=float),
        "quintile_returns": quintile_df,
        "ic_series":        ic_series,
    }
    ls_result = {
        "returns_gross":    ls_gross,
        "returns_net":      ls_net,
        "cumulative_gross": (1 + ls_gross).cumprod(),
        "cumulative_net":   (1 + ls_net).cumprod(),
        "benchmark_returns": bm_series,
        "turnover":         pd.Series(ls_to_list, index=idx, dtype=float),
        "quintile_returns": quintile_df,
        "ic_series":        ic_series,
    }

    return lo_result, ls_result


def _portfolio_return(weights: pd.Series, stock_rets: pd.Series) -> float:
    """
    Compute the weighted average portfolio return.

    Tickers in weights that have no return data contribute 0 to the sum
    (conservative treatment: missing = flat, not missing = delisted).

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights.  Values may be negative (short positions).
    stock_rets : pd.Series
        Individual stock returns.

    Returns
    -------
    float
        Weighted sum of returns.  NaN if weights is empty.
    """
    if weights.empty:
        return np.nan
    aligned = stock_rets.reindex(weights.index).fillna(0.0)
    return float((weights * aligned).sum())


def _benchmark_period_return(
    benchmark_prices: pd.Series,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> float:
    """
    Compute the benchmark return over [t_start, t_end].

    Uses the last available price on or before each date to handle
    weekends and holidays consistently with get_price_at_date.

    Parameters
    ----------
    benchmark_prices : pd.Series
        Daily benchmark prices indexed by date.
    t_start : pd.Timestamp
        Period start date.
    t_end : pd.Timestamp
        Period end date.

    Returns
    -------
    float
        Simple period return (p_end / p_start - 1). NaN if unavailable.
    """
    t_start = pd.Timestamp(t_start)
    t_end = pd.Timestamp(t_end)

    avail_start = benchmark_prices[benchmark_prices.index <= t_start]
    avail_end = benchmark_prices[benchmark_prices.index <= t_end]

    if avail_start.empty or avail_end.empty:
        return np.nan

    p_start = float(avail_start.iloc[-1])
    p_end = float(avail_end.iloc[-1])

    if pd.isna(p_start) or pd.isna(p_end) or p_start <= 0:
        return np.nan

    return p_end / p_start - 1.0


def _empty_result() -> dict:
    """Return a backtest result dict with empty Series (no data periods)."""
    empty = pd.Series(dtype=float)
    return {
        "returns_gross":    empty,
        "returns_net":      empty,
        "cumulative_gross": empty,
        "cumulative_net":   empty,
        "benchmark_returns": empty,
        "turnover":         empty,
        "quintile_returns": pd.DataFrame(),
        "ic_series":        empty,
    }
