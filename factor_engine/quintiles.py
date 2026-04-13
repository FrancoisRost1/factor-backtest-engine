"""
Quintile assignment and quintile-return computation.

The universe is divided into five buckets (Q1 = worst, Q5 = best) based on
composite factor scores. This is the standard academic and practitioner
approach for evaluating factor efficacy: a factor "works" if Q5 consistently
outperforms Q1 over time.

Assignment logic uses rank-based partitioning rather than pd.qcut to avoid
edge cases with tied values and to produce exactly equal-sized quintiles
whenever the universe size is divisible by 5.
"""

import numpy as np
import pandas as pd


def assign_quintiles(
    composite_score: pd.Series,
    n_quintiles: int = 5,
) -> pd.Series:
    """
    Assign each ticker to a quintile (1 = lowest score, n_quintiles = highest).

    Uses rank-based partitioning: tickers are ranked by composite score, then
    divided into n_quintiles equal buckets.  Ties are broken by order of
    appearance in the Series (rank method='first'), so two tickers with
    identical scores are assigned consecutive unique ranks.  This makes quintile
    boundaries fully deterministic, no ticker ever straddles two buckets.

    Returns all NaN if fewer than n_quintiles tickers have valid scores, the
    quintile spread is meaningless with only a handful of stocks.

    Parameters
    ----------
    composite_score : pd.Series
        Composite factor scores. Index = tickers. NaN = excluded from ranking.
    n_quintiles : int
        Number of quantile buckets (default 5).

    Returns
    -------
    pd.Series
        Quintile label (1 to n_quintiles) per ticker; NaN for missing scores
        or if fewer than n_quintiles valid scores exist.
    """
    valid = composite_score.dropna()

    if len(valid) < n_quintiles:
        return pd.Series(np.nan, index=composite_score.index, name=composite_score.name)

    # rank(method='first') assigns unique ranks with no ties, preserving order
    ranks = valid.rank(method="first")
    n = len(valid)

    # Map rank → quintile: floor((rank - 1) / n * n_quintiles) + 1
    # Clip to [1, n_quintiles] to handle the edge case where rank == n exactly
    quintile_raw = np.floor((ranks - 1) / n * n_quintiles).astype(int) + 1
    quintiles = quintile_raw.clip(1, n_quintiles).astype(float)

    # Reindex to full original index; tickers not in valid get NaN
    return quintiles.reindex(composite_score.index)


def compute_quintile_returns(
    quintile_assignments: pd.Series,
    returns: pd.Series,
    n_quantiles: int = 5,
) -> pd.Series:
    """
    Compute the equal-weighted mean return for each quintile bucket.

    This is the core diagnostic for factor evaluation: a factor with strong
    predictive power shows a monotonic return spread from Q1 to Q5.

    Parameters
    ----------
    quintile_assignments : pd.Series
        Quintile label (1–n_quantiles) per ticker. NaN tickers are excluded.
    returns : pd.Series
        Realised returns for the same period. Index = tickers.
    n_quantiles : int
        Number of quantile buckets. Read from config['portfolio']['n_quantiles'].

    Returns
    -------
    pd.Series
        Mean return per quintile; index = [1, ..., n_quantiles].
        NaN for quintiles with no members.
    """
    result = {}
    for q in range(1, n_quantiles + 1):
        tickers_in_q = quintile_assignments[quintile_assignments == q].index
        if len(tickers_in_q) == 0:
            result[q] = np.nan
        else:
            result[q] = returns.reindex(tickers_in_q).mean()
    return pd.Series(result, name="quintile_returns")
