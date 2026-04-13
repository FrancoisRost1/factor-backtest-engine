"""
Portfolio construction from quintile assignments.

Two standard long-only and long-short portfolio styles are provided.
Both styles guarantee the weight constraints that institutional portfolios
require: non-negative weights for long-only, and exact zero net exposure
for long-short.

Long-Only (Q5):
  Invests only in the top quintile. The simplest implementation of a factor
  strategy, suitable for mutual funds and other constrained mandates.

Long-Short (Q5 long, Q1 short):
  The canonical academic factor portfolio. Long-short isolates the pure factor
  return by cancelling out market beta (approximately). Equal gross exposure
  on each side (+0.5, -0.5) is the standard academic convention (see
  Fama-French, AQR, and most empirical asset pricing papers).

*** CAP-WEIGHT CONTAMINATION WARNING ***
The market_caps passed here come from fetch_fundamentals(), which returns
today's market cap for every ticker.  At each historical rebalance date the
same current-day caps are used to weight positions.  This introduces look-ahead
bias into ALL cap-weighted portfolios, even momentum and low-volatility, which
otherwise derive entirely from historical prices.  The equal-weight variants
are unaffected.  Cap-weighted results should be interpreted with the same
caution as the fundamental-factor results (value, quality, size).
"""

import numpy as np
import pandas as pd


def construct_long_only(
    quintile_assignments: pd.Series,
    market_caps: pd.Series = None,
    weighting: str = "equal",
    long_quintile: int = 5,
) -> pd.Series:
    """
    Build a long-only portfolio from the top-quintile stocks.

    Weights sum to exactly 1.0. Only stocks in `long_quintile` receive nonzero
    weight. No negative weights are ever produced.

    Parameters
    ----------
    quintile_assignments : pd.Series
        Quintile label (1–n_quantiles) per ticker.
    market_caps : pd.Series or None
        Market capitalisation per ticker. Required for cap_weight.
    weighting : str
        'equal'    , equal weight across all top-bucket stocks.
        'cap_weight', weight proportional to market cap within the bucket.
    long_quintile : int
        Quintile label representing the top bucket (highest factor score).
        Read from config['portfolio']['long_quintile']. Defaults to 5.

    Returns
    -------
    pd.Series
        Portfolio weights. Index = top-bucket tickers only. Values sum to 1.0.
    """
    top_tickers = quintile_assignments[quintile_assignments == long_quintile].index

    if len(top_tickers) == 0:
        return pd.Series(dtype=float)

    if weighting == "equal":
        w = 1.0 / len(top_tickers)
        return pd.Series(w, index=top_tickers)

    if weighting == "cap_weight":
        if market_caps is None:
            raise ValueError("market_caps must be provided for cap_weight")
        caps = market_caps.reindex(top_tickers).clip(lower=0)
        total = caps.sum()  # NaN caps pass through clip; sum(skipna=True) treats them as 0
        if total == 0 or pd.isna(total):
            # All market caps missing or zero, fall back to equal weight so the
            # portfolio remains investable rather than producing zero allocations.
            w = 1.0 / len(top_tickers)
            return pd.Series(w, index=top_tickers)
        return caps / total

    raise ValueError(f"Unknown weighting scheme: '{weighting}'. Use 'equal' or 'cap_weight'.")


def construct_long_short(
    quintile_assignments: pd.Series,
    market_caps: pd.Series = None,
    weighting: str = "equal",
    long_quintile: int = 5,
    short_quintile: int = 1,
) -> pd.Series:
    """
    Build a long/short portfolio: long top bucket (+0.5 gross), short bottom (-0.5 gross).

    Weights sum to exactly 0.0 (dollar-neutral).
    Long side sums to +0.5; short side sums to -0.5.
    Middle quintiles receive zero weight (not included in output index).

    Parameters
    ----------
    quintile_assignments : pd.Series
        Quintile label (1–n_quantiles) per ticker.
    market_caps : pd.Series or None
        Market capitalisation per ticker. Required for cap_weight.
    weighting : str
        'equal'    , equal weight within each leg.
        'cap_weight', weight proportional to market cap within each leg.
    long_quintile : int
        Quintile label for the long leg (top bucket).
        Read from config['portfolio']['long_quintile']. Defaults to 5.
    short_quintile : int
        Quintile label for the short leg (bottom bucket).
        Read from config['portfolio']['short_quintile']. Defaults to 1.

    Returns
    -------
    pd.Series
        Portfolio weights. Index = long ∪ short tickers. Values sum to 0.0.
        Positive weights = long, negative weights = short.
    """
    top_tickers = quintile_assignments[quintile_assignments == long_quintile].index
    bottom_tickers = quintile_assignments[quintile_assignments == short_quintile].index

    def _leg_weights(tickers, target_sum: float) -> pd.Series:
        """Compute normalised weights for one leg, scaled to target_sum."""
        if len(tickers) == 0:
            return pd.Series(dtype=float)

        if weighting == "equal":
            w = target_sum / len(tickers)
            return pd.Series(w, index=tickers)

        if weighting == "cap_weight":
            if market_caps is None:
                raise ValueError("market_caps must be provided for cap_weight")
            caps = market_caps.reindex(tickers).clip(lower=0)
            total = caps.sum()  # NaN caps pass through clip; sum(skipna=True) treats them as 0
            if total == 0 or pd.isna(total):
                # All market caps missing or zero, fall back to equal weight so the
                # portfolio leg remains investable rather than producing zero allocations.
                w = target_sum / len(tickers)
                return pd.Series(w, index=tickers)
            return (caps / total) * target_sum

        raise ValueError(f"Unknown weighting scheme: '{weighting}'.")

    long_weights = _leg_weights(top_tickers, +0.5)
    short_weights = _leg_weights(bottom_tickers, -0.5)

    return pd.concat([long_weights, short_weights])
