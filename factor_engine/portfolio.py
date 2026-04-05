"""
Portfolio construction from quintile assignments.

Two standard long-only and long-short portfolio styles are provided.
Both styles guarantee the weight constraints that institutional portfolios
require: non-negative weights for long-only, and exact zero net exposure
for long-short.

Long-Only (Q5):
  Invests only in the top quintile. The simplest implementation of a factor
  strategy — suitable for mutual funds and other constrained mandates.

Long-Short (Q5 long, Q1 short):
  The canonical academic factor portfolio. Long-short isolates the pure factor
  return by cancelling out market beta (approximately). Equal gross exposure
  on each side (+0.5, -0.5) is the standard academic convention (see
  Fama-French, AQR, and most empirical asset pricing papers).

*** CAP-WEIGHT CONTAMINATION WARNING ***
The market_caps passed here come from fetch_fundamentals(), which returns
today's market cap for every ticker.  At each historical rebalance date the
same current-day caps are used to weight positions.  This introduces look-ahead
bias into ALL cap-weighted portfolios — even momentum and low-volatility, which
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
) -> pd.Series:
    """
    Build a long-only portfolio from Q5 (top-quintile) stocks.

    Weights sum to exactly 1.0. Only stocks in quintile 5 receive nonzero weight.
    No negative weights are ever produced.

    Parameters
    ----------
    quintile_assignments : pd.Series
        Quintile label (1–5) per ticker. Only label 5 is included.
    market_caps : pd.Series or None
        Market capitalisation per ticker. Required for cap_weight.
    weighting : str
        'equal'     — equal weight across all Q5 stocks.
        'cap_weight' — weight proportional to market cap within Q5.

    Returns
    -------
    pd.Series
        Portfolio weights. Index = Q5 tickers only. Values sum to 1.0.
    """
    q5_tickers = quintile_assignments[quintile_assignments == 5].index

    if len(q5_tickers) == 0:
        return pd.Series(dtype=float)

    if weighting == "equal":
        w = 1.0 / len(q5_tickers)
        return pd.Series(w, index=q5_tickers)

    if weighting == "cap_weight":
        if market_caps is None:
            raise ValueError("market_caps must be provided for cap_weight")
        caps = market_caps.reindex(q5_tickers).clip(lower=0)
        total = caps.sum()  # NaN caps pass through clip; sum(skipna=True) treats them as 0
        if total == 0 or pd.isna(total):
            # All market caps missing or zero — fall back to equal weight so the
            # portfolio remains investable rather than producing zero allocations.
            w = 1.0 / len(q5_tickers)
            return pd.Series(w, index=q5_tickers)
        return caps / total

    raise ValueError(f"Unknown weighting scheme: '{weighting}'. Use 'equal' or 'cap_weight'.")


def construct_long_short(
    quintile_assignments: pd.Series,
    market_caps: pd.Series = None,
    weighting: str = "equal",
) -> pd.Series:
    """
    Build a long/short portfolio: long Q5 (+0.5 gross), short Q1 (-0.5 gross).

    Weights sum to exactly 0.0 (dollar-neutral).
    Long side sums to +0.5; short side sums to -0.5.
    Q2, Q3, Q4 receive zero weight (not included in output index).

    Parameters
    ----------
    quintile_assignments : pd.Series
        Quintile label (1–5) per ticker.
    market_caps : pd.Series or None
        Market capitalisation per ticker. Required for cap_weight.
    weighting : str
        'equal'     — equal weight within each leg.
        'cap_weight' — weight proportional to market cap within each leg.

    Returns
    -------
    pd.Series
        Portfolio weights. Index = Q5 ∪ Q1 tickers. Values sum to 0.0.
        Positive weights = long (Q5), negative weights = short (Q1).
    """
    q5_tickers = quintile_assignments[quintile_assignments == 5].index
    q1_tickers = quintile_assignments[quintile_assignments == 1].index

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
                # All market caps missing or zero — fall back to equal weight so the
                # portfolio leg remains investable rather than producing zero allocations.
                w = target_sum / len(tickers)
                return pd.Series(w, index=tickers)
            return (caps / total) * target_sum

        raise ValueError(f"Unknown weighting scheme: '{weighting}'.")

    long_weights = _leg_weights(q5_tickers, +0.5)
    short_weights = _leg_weights(q1_tickers, -0.5)

    return pd.concat([long_weights, short_weights])
