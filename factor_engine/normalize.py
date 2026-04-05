"""
Factor normalization via cross-sectional percentile ranking.

Raw factor values are converted to percentile ranks in (0, 1] so that all
factors are expressed on the same scale before compositing. This avoids
factors with large magnitudes (e.g., market cap in billions) dominating
factors with small magnitudes (e.g., earnings yield of 0.05).

Percentile rank = rank / n_valid, where rank is the position within the
cross-section after sorting. The minimum achievable rank is 1/n (not 0),
and the maximum is always 1.0.

For factors where LOWER raw values are better (size, volatility), the ranking
is inverted: the ticker with the smallest raw value receives rank 1.0.
This ensures that "better" always means "higher rank" across all factors,
enabling simple addition or weighting for composite scores.

References:
  Fama & French (1992): Size and B/M as cross-sectional predictors.
  Ang et al. (2006): Idiosyncratic volatility and expected returns.
"""

from typing import List, Optional

import pandas as pd

# Default columns where lower raw value = better signal = should rank highest
DEFAULT_INVERT_COLUMNS: List[str] = ["log_market_cap", "rolling_vol_60d"]


def percentile_rank(
    factors_df: pd.DataFrame,
    invert_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Assign each ticker a cross-sectional percentile rank for every factor.

    Rank = position / n_valid_tickers, so the output lies in (0, 1].
    NaN inputs are preserved as NaN in the output (not ranked).

    For columns in invert_columns, ranking is descending: the ticker with the
    LOWEST raw factor value receives rank 1.0. This is appropriate for:
      - log_market_cap: smaller firms have the size premium
      - rolling_vol_60d: lower volatility = more attractive (low-vol anomaly)

    For all other columns, ranking is ascending: highest raw value = rank 1.0.

    Parameters
    ----------
    factors_df : pd.DataFrame
        Raw factor values. Rows = tickers, columns = factor names.
    invert_columns : list[str] or None
        Columns to rank in descending order (lower raw = higher rank).
        Defaults to DEFAULT_INVERT_COLUMNS = ['log_market_cap', 'rolling_vol_60d'].
        Pass an empty list [] to rank everything ascending.

    Returns
    -------
    pd.DataFrame
        Same shape as input. Values in (0, 1]; NaN where input was NaN.
    """
    if invert_columns is None:
        invert_columns = DEFAULT_INVERT_COLUMNS

    result = pd.DataFrame(index=factors_df.index, columns=factors_df.columns, dtype=float)

    for col in factors_df.columns:
        series = factors_df[col]
        # ascending=True  → highest raw value gets rank 1.0 (normal factors)
        # ascending=False → lowest raw value gets rank 1.0 (inverted factors)
        ascending = col not in invert_columns
        result[col] = series.rank(pct=True, ascending=ascending, na_option="keep")

    return result
