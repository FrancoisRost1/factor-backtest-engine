"""
Information Coefficient (IC) computation.

The IC measures how well a factor score predicts subsequent returns in
cross-section. It is defined as the Spearman rank correlation between
factor scores and forward returns at a given point in time.

IC = 1.0 → perfect positive prediction (highest scores → highest returns)
IC = 0.0 → no predictive power
IC = -1.0 → perfect inverse prediction

A factor is considered "good" if it generates a mean IC (ICIR = mean(IC) / std(IC))
above ~0.3 in backtesting (Grinold & Kahn 2000). Individual IC observations
are noisy; the time-series mean and ICIR (signal-to-noise ratio) are what matter.

Spearman correlation is used instead of Pearson because:
1. Returns have fat tails; rank-based measures are more robust.
2. We care about relative ordering, not absolute magnitude.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_ic(
    scores: pd.Series,
    forward_returns: pd.Series,
    min_obs: int = 10,
) -> float:
    """
    Compute the Information Coefficient for one cross-section.

    Drops tickers where either score or forward_return is NaN before computing.
    Returns NaN if fewer than min_obs valid pairs remain after dropping NaN.

    Parameters
    ----------
    scores : pd.Series
        Factor scores at the beginning of the period. Index = tickers.
    forward_returns : pd.Series
        Realised returns over the subsequent period. Index = tickers.
    min_obs : int
        Minimum number of valid ticker pairs required to compute IC.
        Below this threshold the estimate is too noisy to be meaningful.

    Returns
    -------
    float
        Spearman rank correlation in [-1, 1]. NaN if insufficient data.
    """
    combined = pd.DataFrame(
        {"scores": scores, "returns": forward_returns}
    ).dropna()

    if len(combined) < min_obs:
        return np.nan

    ic, _ = spearmanr(combined["scores"], combined["returns"])
    return float(ic)


def ic_summary(ic_series: pd.Series) -> dict:
    """
    Summarise a time-series of Information Coefficient values.

    Computes the mean IC and the IC Information Ratio (ICIR).  The ICIR is
    mean(IC) / std(IC), a signal-to-noise ratio measuring how consistently
    the factor predicts returns over time.  It is NOT a t-statistic; a
    t-statistic would divide by std(IC) / sqrt(n).  ICIR and the t-statistic
    differ by a factor of sqrt(n).

    Grinold & Kahn (2000) suggest ICIR > 0.5 indicates a useful factor.
    Mean IC alone is noisy; ICIR rewards both high average IC and low IC
    variability (i.e., consistent predictive power across regimes).

    Parameters
    ----------
    ic_series : pd.Series
        Time-series of per-period IC values (NaN periods are dropped).

    Returns
    -------
    dict
        Keys:
          mean_ic, mean of the non-NaN IC values.
          ic_ir  , mean_ic / std(ic).  NaN if fewer than 2 valid observations
                    or if std is zero.
    """
    clean = ic_series.dropna()

    if clean.empty:
        return {"mean_ic": np.nan, "ic_ir": np.nan}

    mean_ic = float(clean.mean())
    std_ic = float(clean.std())

    if len(clean) < 2 or std_ic == 0:
        ic_ir = np.nan
    else:
        ic_ir = float(mean_ic / std_ic)

    return {"mean_ic": mean_ic, "ic_ir": ic_ir}
