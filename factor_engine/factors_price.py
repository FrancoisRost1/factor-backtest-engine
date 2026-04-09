"""
Price-based factor computation: momentum and rolling volatility.

These factors are derived purely from historical adjusted close prices,
with strict no-lookahead enforcement: all computations filter data to
<= as_of_date before touching any price series.

Momentum (12-1): The return from 12 months ago to 1 month ago.
  Skipping the most recent month avoids the well-documented short-term
  reversal effect (Jegadeesh 1990). This is the standard academic signal.

Rolling Volatility: Annualized standard deviation of daily returns over
  a rolling lookback window. Used as both a standalone factor (low-vol
  anomaly, Ang et al. 2006) and a position-sizing input.
"""

import numpy as np
import pandas as pd


def compute_momentum_12_1(prices_df: pd.DataFrame, as_of_date) -> pd.Series:
    """
    Compute cross-sectional 12-1 momentum for all tickers at as_of_date.

    Momentum = price(t - 1 month) / price(t - 12 months) - 1

    The 1-month skip avoids short-term reversal. Tickers with insufficient
    price history return NaN. No data beyond as_of_date is ever accessed.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Daily adjusted close prices; index = dates, columns = tickers.
    as_of_date : date-like
        Computation date. Data after this date is ignored.

    Returns
    -------
    pd.Series
        Momentum return per ticker; NaN if insufficient history.
    """
    as_of_date = pd.Timestamp(as_of_date)
    # Strict no-lookahead: drop all dates after as_of_date
    prices = prices_df[prices_df.index <= as_of_date]

    date_1m_ago = as_of_date - pd.DateOffset(months=1)
    date_12m_ago = as_of_date - pd.DateOffset(months=12)

    def _price_on_or_before(target_date):
        """Return the last available row on or before target_date."""
        available = prices[prices.index <= pd.Timestamp(target_date)]
        if available.empty:
            return pd.Series(np.nan, index=prices.columns)
        return available.iloc[-1]

    price_1m = _price_on_or_before(date_1m_ago)
    price_12m = _price_on_or_before(date_12m_ago)

    momentum = price_1m / price_12m - 1.0

    # Zero or negative prices indicate bad data — set to NaN
    bad_12m = price_12m <= 0
    bad_1m = price_1m <= 0
    momentum[bad_12m | bad_1m] = np.nan

    momentum.name = "momentum_12_1"
    return momentum


def compute_rolling_volatility(
    prices_df: pd.DataFrame,
    as_of_date,
    lookback_days: int = 60,
    annualization_factor: int = 252,
) -> pd.Series:
    """
    Compute annualized rolling volatility of daily returns at as_of_date.

    Volatility = std(daily_returns[-lookback_days:]) * sqrt(252)

    Uses the last `lookback_days` trading days of return data ending on
    as_of_date. Returns NaN for all tickers if fewer than lookback_days + 1
    price observations are available (insufficient history for one full window).

    Parameters
    ----------
    prices_df : pd.DataFrame
        Daily adjusted close prices; index = dates, columns = tickers.
    as_of_date : date-like
        Computation date. Data after this date is ignored.
    lookback_days : int
        Number of trading days in the volatility window (default 60 ≈ 3 months).
    annualization_factor : int
        Trading days per year for annualising daily volatility.
        Read from config['analytics']['annualization_factor'].

    Returns
    -------
    pd.Series
        Annualized volatility per ticker; NaN if insufficient history.
    """
    as_of_date = pd.Timestamp(as_of_date)
    # Strict no-lookahead: drop all dates after as_of_date
    prices = prices_df[prices_df.index <= as_of_date]

    # Need lookback_days returns = lookback_days + 1 price observations
    if len(prices) < lookback_days + 1:
        return pd.Series(np.nan, index=prices.columns)

    recent_prices = prices.iloc[-(lookback_days + 1):]
    daily_returns = recent_prices.pct_change().dropna()

    # Annualize: daily std × sqrt(trading days/year)
    vol = daily_returns.std() * np.sqrt(annualization_factor)

    vol.name = "rolling_vol_60d"
    return vol
