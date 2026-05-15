"""
Portfolio performance analytics: return, risk, and risk-adjusted metrics.

All functions accept a pd.Series of period returns (e.g., monthly or daily).
The `periods_per_year` parameter controls annualisation and defaults to 12
(monthly), matching the typical backtesting frequency for factor strategies.

Metric definitions follow standard institutional practice:
  Annualised Return  , geometric compounding (CAGR)
  Sharpe Ratio       , mean excess return / std dev, annualised (Sharpe 1966)
  Sortino Ratio      , mean excess return / downside deviation (Sortino 1994)
  Max Drawdown       , peak-to-trough decline in cumulative wealth
  Calmar Ratio       , annualised return / |max drawdown| (Young 1991)
  Hit Rate           , fraction of periods strategy beats benchmark
"""

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    """
    Compute the compound annual growth rate (CAGR) from period returns.

    CAGR = (∏(1 + r_t))^(periods_per_year / n) - 1

    Parameters
    ----------
    returns : pd.Series
        Period returns (e.g., monthly). Must not be empty.
    periods_per_year : int
        Number of periods in a year (12 for monthly, 252 for daily).

    Returns
    -------
    float
        Annualised return as a fraction. NaN if returns is empty.
    """
    if returns.empty:
        return np.nan
    n = len(returns)
    total_growth = (1.0 + returns).prod()
    return float(total_growth ** (periods_per_year / n) - 1.0)


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    """
    Compute the annualised Sharpe ratio.

    Sharpe = mean(excess_return) / std(excess_return) × sqrt(periods_per_year)

    Returns NaN (not infinity) when volatility is zero, a constant-return
    stream has undefined risk-adjusted performance under this metric.

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    risk_free : float
        Annual risk-free rate. Divided by periods_per_year for the period rate.
    periods_per_year : int
        Annualisation factor.

    Returns
    -------
    float
        Annualised Sharpe ratio. NaN if vol is zero or insufficient data.
    """
    if returns.empty or len(returns) < 2:
        return np.nan
    period_rf = risk_free / periods_per_year
    excess = returns - period_rf
    std = excess.std()
    if std == 0:
        return np.nan
    return float((excess.mean() / std) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    """
    Compute the annualised Sortino ratio.

    Sortino = annualised_mean_excess / annualised_downside_deviation

    Downside deviation is computed as the RMS of min(excess_return, 0) across
    ALL periods (Sortino 1994), not just the negative ones.  Using only
    negative-return periods would shrink the denominator and inflate the ratio
    whenever a strategy has few losing periods.  The full-sample RMS is the
    standard institutional definition.

    Returns NaN when downside deviation is zero (e.g., all returns are positive
    and equal, no harmful volatility to penalise).

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    risk_free : float
        Annual risk-free rate.
    periods_per_year : int
        Annualisation factor.

    Returns
    -------
    float
        Annualised Sortino ratio. NaN if downside deviation is zero.
    """
    if returns.empty:
        return np.nan
    period_rf = risk_free / periods_per_year
    excess = returns - period_rf
    # Full-sample downside deviation: RMS of min(excess, 0), Sortino 1994
    downside = np.minimum(excess.values, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year))
    if downside_dev == 0:
        return np.nan
    annualised_mean = float(excess.mean() * periods_per_year)
    return annualised_mean / downside_dev


def max_drawdown(returns: pd.Series) -> float:
    """
    Compute the maximum peak-to-trough drawdown in cumulative wealth.

    Drawdown at time t = (cum_wealth[t] - peak_before_t) / peak_before_t

    Always returns a value ≤ 0. Returns 0.0 for all-positive return series
    (wealth curve never falls below a prior peak).

    *** FREQUENCY WARNING ***
    Drawdown is computed at rebalance-period granularity (monthly, quarterly,
    or annual), not on daily NAV.  An annual-rebalance backtest samples the
    wealth curve at most 10 times over a 10-year period; large intra-period
    losses that recover before the next rebalance date are invisible.  Annual
    max drawdown figures can dramatically understate true risk.  Use the
    monthly-rebalance drawdown as the most conservative estimate, or switch
    to a daily-mark implementation for risk reporting.  The Calmar ratio
    inherits this limitation.

    Parameters
    ----------
    returns : pd.Series
        Period returns.

    Returns
    -------
    float
        Maximum drawdown as a negative fraction (e.g., -0.30 = -30% drawdown).
    """
    if returns.empty:
        return 0.0
    # Prepend 1.0 to represent initial wealth before any returns.
    # Without this, a single negative return would show drawdown=0 because
    # its own value becomes the "peak", but the investor's loss is real.
    wealth_curve = (1.0 + returns).cumprod()
    wealth_with_start = pd.concat(
        [pd.Series([1.0], index=["start"]), wealth_curve]
    )
    rolling_peak = wealth_with_start.cummax()
    drawdowns = (wealth_with_start - rolling_peak) / rolling_peak
    return float(drawdowns.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """
    Compute the Calmar ratio = annualised_return / |max_drawdown|.

    A higher Calmar indicates better return per unit of worst-case loss.
    Returns NaN if max drawdown is zero (no loss to penalise against).

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    periods_per_year : int
        Annualisation factor.

    Returns
    -------
    float
        Calmar ratio. NaN if max drawdown is zero.
    """
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    ann_ret = annualized_return(returns, periods_per_year)
    return float(ann_ret / abs(mdd))


def hit_rate(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Compute the hit rate = fraction of periods the strategy beats the benchmark.

    A hit rate above 0.5 means the strategy outperforms more often than not.
    Periods where strategy exactly equals benchmark count as NOT outperforming
    (strict inequality).

    Parameters
    ----------
    strategy_returns : pd.Series
        Strategy period returns.
    benchmark_returns : pd.Series
        Benchmark period returns. Must align with strategy_returns by index.

    Returns
    -------
    float
        Hit rate in [0, 1]. NaN if strategy_returns is empty.
    """
    if strategy_returns.empty:
        return np.nan
    # Align by index so mismatched dates produce NaN rather than silently comparing
    # unrelated periods.  reindex + dropna keeps only periods present in both series.
    strategy_aligned, benchmark_aligned = strategy_returns.align(
        benchmark_returns, join="inner"
    )
    if strategy_aligned.empty:
        return np.nan
    outperform = strategy_aligned > benchmark_aligned
    return float(outperform.mean())


def compute_all_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    turnover: pd.Series,
    periods_per_year: int = 12,
    risk_free: float = 0.0,
) -> dict:
    """
    Compute the full suite of performance metrics for one backtest result.

    Convenience wrapper that calls all individual metric functions and bundles
    their outputs into a single dict.  Used by main.py to avoid repetitive
    per-metric calls across the 72 backtest combinations.

    Parameters
    ----------
    returns : pd.Series
        Period returns for the strategy (gross or net).
    benchmark_returns : pd.Series
        Benchmark period returns aligned to the same dates as returns.
    turnover : pd.Series
        Per-period portfolio turnover (fraction of portfolio traded).
    periods_per_year : int
        Annualisation factor (12 = monthly, 4 = quarterly, 1 = annual).
    risk_free : float
        Annual risk-free rate used in Sharpe and Sortino calculations.

    Returns
    -------
    dict
        Keys: annualized_return, sharpe, sortino, max_drawdown,
              calmar, hit_rate, avg_turnover.
    """
    avg_to = float(turnover.mean()) if not turnover.empty else np.nan

    return {
        "annualized_return": annualized_return(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
        "hit_rate": hit_rate(returns, benchmark_returns),
        "avg_turnover": avg_to,
    }
