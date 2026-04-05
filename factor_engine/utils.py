"""
Shared utility functions for the factor backtest engine.

Provides safe arithmetic helpers and date-aligned price lookups used
across backtest.py, data_loader.py, and factor computations.

All functions are pure (no side effects, no I/O) to keep them easily testable.
"""

import numpy as np
import pandas as pd


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Divide two numbers, returning NaN instead of raising on zero denominator.

    Used throughout factor computations to guard against earnings-yield
    calculations on firms with zero P/E and debt-coverage edge cases.

    Parameters
    ----------
    numerator : float
    denominator : float

    Returns
    -------
    float
        numerator / denominator, or NaN if denominator is 0, NaN, or None.
    """
    if pd.isna(denominator) or denominator == 0 or pd.isna(numerator):
        return np.nan
    return float(numerator) / float(denominator)


def get_price_at_date(prices_df: pd.DataFrame, date) -> pd.Series:
    """
    Return the last available closing price for all tickers on or before date.

    Uses 'last price on or before date' to handle weekends, holidays, and
    corporate events where a given stock may not have traded on the exact date.
    If no prices exist before date, returns NaN for all tickers.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Daily adjusted close prices; index = dates, columns = tickers.
    date : date-like
        Target date (inclusive upper bound).

    Returns
    -------
    pd.Series
        Most recent price per ticker on or before date; NaN if unavailable.
    """
    date = pd.Timestamp(date)
    available = prices_df[prices_df.index <= date]
    if available.empty:
        return pd.Series(np.nan, index=prices_df.columns)
    return available.iloc[-1]


def get_period_returns(
    prices_df: pd.DataFrame,
    date_start,
    date_end,
) -> pd.Series:
    """
    Compute the simple return for each ticker over [date_start, date_end].

    Return = price(t_end) / price(t_start) - 1

    Uses last-available price on or before each date, consistent with how
    the backtester measures performance between rebalance dates.
    Tickers with a missing or non-positive price at either date return NaN.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Daily adjusted close prices.
    date_start : date-like
        Period start date (cost basis).
    date_end : date-like
        Period end date (exit price).

    Returns
    -------
    pd.Series
        Simple return per ticker; NaN if price unavailable at either date.
    """
    p_start = get_price_at_date(prices_df, date_start)
    p_end = get_price_at_date(prices_df, date_end)

    rets = p_end / p_start - 1.0

    # Zero or negative prices indicate data errors — suppress to NaN
    rets[p_start <= 0] = np.nan
    rets[p_end <= 0] = np.nan

    return rets
