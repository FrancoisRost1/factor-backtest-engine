"""
Factor strategy regression vs benchmark.

Regresses strategy returns against benchmark returns using OLS:

    strategy_return = alpha + beta × benchmark_return + epsilon

Alpha: raw per-period OLS intercept.  This is NOT Jensen's alpha.
  Jensen's alpha requires regressing EXCESS returns (strategy - rf) against
  EXCESS benchmark returns (benchmark - rf); with rf = 0 the two coincide,
  but the label is misleading.  The intercept here is the per-period constant
  return unexplained by benchmark exposure.  Multiply by periods_per_year to
  get an annualised figure before displaying or comparing across frequencies.

Beta: strategy's sensitivity to the benchmark.
  β = 1 → moves 1-for-1 with benchmark.
  β < 1 → defensive (lower market exposure).
  β > 1 → aggressive (leveraged market exposure).

R²: fraction of strategy variance explained by the benchmark.
  R² near 1 → strategy is mostly a benchmark tracker.
  R² near 0 → strategy return is independent of the benchmark.

Uses numpy.linalg.lstsq to avoid sklearn dependency. All outputs are
returned as a dict for easy DataFrame assembly across multiple time windows.
"""

import numpy as np
import pandas as pd


def regress_vs_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    min_obs: int = 5,
) -> dict:
    """
    OLS regression: strategy = alpha + beta × benchmark + epsilon.

    Aligns strategy and benchmark by index, drops NaN pairs, then fits.
    Returns all NaN if fewer than min_obs valid pairs exist.

    Parameters
    ----------
    strategy_returns : pd.Series
        Period returns of the strategy being evaluated.
    benchmark_returns : pd.Series
        Period returns of the benchmark (e.g., S&P 500).
    min_obs : int
        Minimum valid observations required to compute a meaningful regression.

    Returns
    -------
    dict with keys:
        'alpha'     : float — raw per-period OLS intercept (not annualised)
        'beta'      : float — slope coefficient
        'r_squared' : float — coefficient of determination R²
    All values are NaN if insufficient data.

    Note: multiply alpha by periods_per_year to annualise before display.
    """
    combined = pd.DataFrame(
        {"strategy": strategy_returns, "benchmark": benchmark_returns}
    ).dropna()

    nan_result = {"alpha": np.nan, "beta": np.nan, "r_squared": np.nan}

    if len(combined) < min_obs:
        return nan_result

    x = combined["benchmark"].values
    y = combined["strategy"].values
    n = len(x)

    # Design matrix: [1, benchmark_return] for each observation
    X = np.column_stack([np.ones(n), x])

    # OLS via least squares: coefficients = (X'X)^-1 X'y
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coeffs[0]), float(coeffs[1])

    # R-squared = 1 - SS_res / SS_tot
    y_pred = X @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    if ss_tot == 0:
        r_squared = np.nan
    else:
        r_squared = float(1.0 - ss_res / ss_tot)

    return {"alpha": alpha, "beta": beta, "r_squared": r_squared}
