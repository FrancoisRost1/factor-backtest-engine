"""
Fundamental factor computation: earnings yield, ROE, and log market cap.

These factors are derived from cross-sectional fundamental data (P/E ratio,
return on equity, market capitalisation). Combined with price-based factors
from factors_price.py, they form the full five-factor model.

Factor definitions:
  Earnings Yield (EY) = 1 / P/E ratio
    Higher EY = cheaper stock = better value signal.
    NaN for negative, zero, or missing P/E (avoids sign-flip distortion).

  Return on Equity (ROE) = Net Income / Shareholders' Equity
    Higher ROE = more profitable capital allocation.
    Negative ROE is valid data (loss-making firm) and is preserved as-is.

  Log Market Cap = ln(market_cap)
    Size factor. Smaller firms have historically outperformed (Fama-French 1992).
    Logged to compress the wide distribution of market caps.
    NaN for zero or negative market cap (data error).
"""

import numpy as np
import pandas as pd

from factor_engine.factors_price import compute_momentum_12_1, compute_rolling_volatility
from factor_engine.normalize import percentile_rank


# Factors where lower raw value = better signal = should rank highest
_INVERT_FOR_COMPOSITE = ["log_market_cap", "rolling_vol_60d"]
_COMPOSITE_MIN_VALID = 3   # min valid factor scores required to assign a composite


def compute_earnings_yield(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Compute earnings yield = 1 / P/E ratio.

    Negative P/E (loss-making firms) and zero P/E are set to NaN to avoid
    the sign-flip problem: a company with PE=-10 would appear to have a
    "high" positive yield of -10%, which would incorrectly signal cheapness.

    *** POINT-IN-TIME WARNING ***
    This factor uses TODAY's trailing P/E ratio fetched from yfinance, applied
    uniformly across the ENTIRE backtest history.  It is NOT point-in-time.
    A company's current P/E is used for 2014 rebalances as well as 2025 ones.
    This introduces look-ahead bias into the value factor.  In a production
    system, quarterly SEC filings would be used for each rebalance date.
    This simplification is documented in CLAUDE.md under 'Simplifying assumptions'.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        Must contain column 'pe_ratio'. Index = ticker symbols.

    Returns
    -------
    pd.Series
        Earnings yield per ticker; NaN for invalid/missing P/E.
    """
    pe = fundamentals["pe_ratio"].copy()
    # Zero and negative P/E are economically meaningless for yield ranking
    pe = pe.where(pe > 0, other=np.nan)
    result = 1.0 / pe
    result.name = "earnings_yield"
    return result


def compute_roe(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Extract return on equity (ROE) directly from fundamentals.

    ROE is passed through as-is. Negative ROE is valid (net loss) and must
    be preserved so that deeply unprofitable firms rank at the bottom.
    NaN propagates naturally.

    *** POINT-IN-TIME WARNING ***
    This factor uses TODAY's trailing ROE fetched from yfinance, applied
    uniformly across the ENTIRE backtest history.  It is NOT point-in-time.
    A company's current ROE is used for 2014 rebalances as well as 2025 ones.
    This introduces look-ahead bias into the quality factor.  In a production
    system, quarterly SEC filings would be used for each rebalance date.
    This simplification is documented in CLAUDE.md under 'Simplifying assumptions'.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        Must contain column 'roe'. Index = ticker symbols.

    Returns
    -------
    pd.Series
        ROE per ticker; NaN where data is missing.
    """
    result = fundamentals["roe"].copy()
    result.name = "roe"
    return result


def compute_log_market_cap(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Compute natural log of market capitalisation for the size factor.

    Log transformation compresses the right-skewed distribution of market caps,
    making the factor behave better in linear models. Zero and negative values
    indicate data errors and are set to NaN.

    *** POINT-IN-TIME WARNING ***
    This factor uses TODAY's market cap fetched from yfinance, applied
    uniformly across the ENTIRE backtest history.  It is NOT point-in-time.
    A company's current market cap is used for 2014 rebalances as well as 2025 ones.
    This introduces look-ahead bias into the size factor — a company that is large
    today may have been small-cap in 2014, yet is treated as large throughout.
    In a production system, historical market cap (price × shares outstanding at date)
    would be used for each rebalance date.
    This simplification is documented in CLAUDE.md under 'Simplifying assumptions'.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        Must contain column 'market_cap' in USD. Index = ticker symbols.

    Returns
    -------
    pd.Series
        ln(market_cap) per ticker; NaN for invalid market cap.
    """
    mcap = fundamentals["market_cap"].copy()
    mcap = mcap.where(mcap > 0, other=np.nan)
    result = np.log(mcap)
    result.name = "log_market_cap"
    return result


def compute_all_factors(
    fundamentals: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """
    Compute all five factors and combine into a single cross-sectional DataFrame.

    Only tickers with price data available at as_of_date are included in the output.
    Tickers that appear in fundamentals but have no price at as_of_date are silently
    excluded, as they are not tradeable at that date.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        Cross-sectional fundamental data. Must contain columns:
        'pe_ratio', 'roe', 'market_cap'. Index = ticker symbols.
    prices_df : pd.DataFrame
        Daily adjusted close prices. Index = dates, columns = tickers.
    as_of_date : date-like
        The date at which to compute all factors.
    lookback_days : int
        Volatility lookback window in trading days.

    Returns
    -------
    pd.DataFrame
        Columns: earnings_yield, roe, log_market_cap, momentum_12_1, rolling_vol_60d.
        Index = tickers with valid prices at as_of_date.
    """
    as_of_date = pd.Timestamp(as_of_date)

    # Determine which tickers have a price on or before as_of_date
    prices_up_to = prices_df[prices_df.index <= as_of_date]
    if prices_up_to.empty:
        return pd.DataFrame()

    # Tickers where the most recent price is not NaN
    valid_tickers = prices_up_to.iloc[-1].dropna().index

    # Fundamental factors (NaN for tickers not in fundamentals)
    fund = fundamentals.reindex(valid_tickers)
    ey = compute_earnings_yield(fund)
    roe = compute_roe(fund)
    size = compute_log_market_cap(fund)

    # Price-based factors (only for valid tickers, no lookahead)
    price_subset = prices_df[valid_tickers]
    momentum = compute_momentum_12_1(price_subset, as_of_date)
    vol = compute_rolling_volatility(price_subset, as_of_date, lookback_days)

    result = pd.DataFrame(
        {
            "earnings_yield": ey,
            "roe": roe,
            "log_market_cap": size,
            "momentum_12_1": momentum,
            "rolling_vol_60d": vol,
        },
        index=valid_tickers,
    )

    # Composite factor: percentile-rank each of the 5 factors, then average.
    # Stocks with fewer than _COMPOSITE_MIN_VALID valid factor scores are excluded
    # (NaN composite) — the signal would be too noisy with only 1-2 inputs.
    # Ranking is repeated here (not deferred to backtest.py) so that the composite
    # reflects a true equal-weighted blend of all five signals.
    #
    # *** COMPOSITE CONTAMINATION WARNING ***
    # Three of the five inputs (earnings_yield, roe, log_market_cap) use today's
    # fundamental data applied uniformly across all historical rebalance dates.
    # The composite inherits this look-ahead bias — it is not point-in-time and
    # should be interpreted with the same caveats as the individual fundamental
    # factors.  Momentum and low-volatility sub-scores are clean.
    raw_5 = result[["earnings_yield", "roe", "log_market_cap", "momentum_12_1", "rolling_vol_60d"]]
    ranked_5 = percentile_rank(raw_5, invert_columns=_INVERT_FOR_COMPOSITE)
    valid_count = ranked_5.notna().sum(axis=1)
    composite = ranked_5.mean(axis=1, skipna=True)
    composite[valid_count < _COMPOSITE_MIN_VALID] = np.nan
    result["composite_score"] = composite

    return result
