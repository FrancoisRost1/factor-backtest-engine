"""
Tests for factor_engine/ic.py and factor_engine/regression.py.

compute_ic: Information Coefficient (Spearman rank correlation).
regress_vs_benchmark: OLS alpha, beta, R² decomposition.

All expected values are derived analytically or from seeded random data.
"""

import numpy as np
import pandas as pd
import pytest

from factor_engine.ic import compute_ic
from factor_engine.regression import regress_vs_benchmark


# ---------------------------------------------------------------------------
# compute_ic
# ---------------------------------------------------------------------------

class TestComputeIC:
    def test_perfectly_correlated_ic_1(self):
        """Same rank order for scores and returns → IC = 1.0."""
        scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        ic = compute_ic(scores, returns, min_obs=3)
        assert abs(ic - 1.0) < 1e-10

    def test_perfectly_inverse_ic_minus_1(self):
        """Perfectly reversed rank order → IC = -1.0."""
        scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = pd.Series([0.05, 0.04, 0.03, 0.02, 0.01])
        ic = compute_ic(scores, returns, min_obs=3)
        assert abs(ic - (-1.0)) < 1e-10

    def test_uncorrelated_data_ic_near_zero(self):
        """Independent random series: IC should be near 0 (no signal)."""
        np.random.seed(42)
        n = 200
        scores = pd.Series(np.random.normal(0, 1, n))
        returns = pd.Series(np.random.normal(0, 1, n))
        ic = compute_ic(scores, returns, min_obs=10)
        assert abs(ic) < 0.15

    def test_fewer_than_min_obs_returns_nan(self):
        scores = pd.Series([1.0, 2.0])
        returns = pd.Series([0.01, 0.02])
        ic = compute_ic(scores, returns, min_obs=3)
        assert pd.isna(ic)

    def test_exactly_min_obs_returns_value(self):
        """Exactly min_obs valid pairs → should compute (not NaN)."""
        scores = pd.Series([1.0, 2.0, 3.0])
        returns = pd.Series([0.01, 0.02, 0.03])
        ic = compute_ic(scores, returns, min_obs=3)
        assert not pd.isna(ic)

    def test_nan_in_scores_dropped_not_crash(self):
        """NaN scores should be silently dropped before computing IC."""
        scores = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        ic = compute_ic(scores, returns, min_obs=3)
        assert not pd.isna(ic)

    def test_nan_in_returns_dropped_not_crash(self):
        scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = pd.Series([0.01, np.nan, 0.03, 0.04, 0.05])
        ic = compute_ic(scores, returns, min_obs=3)
        assert not pd.isna(ic)

    def test_ic_range_minus_1_to_1(self):
        np.random.seed(7)
        scores = pd.Series(np.random.normal(0, 1, 50))
        returns = pd.Series(np.random.normal(0, 0.02, 50))
        ic = compute_ic(scores, returns, min_obs=10)
        if not pd.isna(ic):
            assert -1.0 <= ic <= 1.0

    def test_stronger_signal_higher_absolute_ic(self):
        """A stronger signal (more correlated) should produce higher |IC|."""
        n = 100
        noise_scale = 0.0001  # very little noise → strong signal
        scores = pd.Series(np.arange(n, dtype=float))
        returns_strong = pd.Series(scores + np.random.normal(0, noise_scale, n))
        returns_weak = pd.Series(np.random.normal(0, 1, n))
        ic_strong = compute_ic(scores, returns_strong, min_obs=10)
        ic_weak = compute_ic(scores, returns_weak, min_obs=10)
        assert abs(ic_strong) > abs(ic_weak)


# ---------------------------------------------------------------------------
# regress_vs_benchmark
# ---------------------------------------------------------------------------

class TestRegressVsBenchmark:
    def test_strategy_equals_benchmark_alpha_0_beta_1_r2_1(self):
        """
        y = x exactly → OLS gives alpha=0, beta=1, R²=1.
        """
        np.random.seed(42)
        bench = pd.Series(np.random.normal(0.01, 0.05, 100))
        result = regress_vs_benchmark(bench, bench)
        assert abs(result["alpha"]) < 1e-8
        assert abs(result["beta"] - 1.0) < 1e-8
        assert abs(result["r_squared"] - 1.0) < 1e-6

    def test_strategy_equals_benchmark_plus_constant_alpha(self):
        """
        y = x + 0.005 → alpha ≈ 0.005, beta ≈ 1.
        """
        np.random.seed(42)
        bench = pd.Series(np.random.normal(0.01, 0.05, 100))
        strategy = bench + 0.005
        result = regress_vs_benchmark(strategy, bench)
        assert abs(result["alpha"] - 0.005) < 1e-8
        assert abs(result["beta"] - 1.0) < 1e-8

    def test_strategy_equals_2x_benchmark_beta_2(self):
        """
        y = 2x → alpha=0, beta=2, R²=1.
        """
        np.random.seed(1)
        bench = pd.Series(np.random.normal(0.01, 0.05, 100))
        strategy = 2.0 * bench
        result = regress_vs_benchmark(strategy, bench)
        assert abs(result["alpha"]) < 1e-8
        assert abs(result["beta"] - 2.0) < 1e-8

    def test_uncorrelated_strategy_beta_near_zero(self):
        """Independent strategy and benchmark → beta near 0."""
        np.random.seed(42)
        bench = pd.Series(np.random.normal(0.01, 0.05, 100))
        strategy = pd.Series(np.random.normal(0.01, 0.05, 100))
        result = regress_vs_benchmark(strategy, bench)
        # With n=100 and truly independent series, |beta| should be small
        assert abs(result["beta"]) < 0.3

    def test_fewer_than_min_obs_all_nan(self):
        bench = pd.Series([0.01, 0.02, 0.03, 0.04])  # only 4 observations
        strategy = pd.Series([0.01, 0.02, 0.03, 0.04])
        result = regress_vs_benchmark(strategy, bench, min_obs=5)
        assert pd.isna(result["alpha"])
        assert pd.isna(result["beta"])
        assert pd.isna(result["r_squared"])

    def test_exactly_min_obs_returns_values(self):
        bench = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        strategy = bench + 0.001
        result = regress_vs_benchmark(strategy, bench, min_obs=5)
        assert not pd.isna(result["alpha"])
        assert not pd.isna(result["beta"])

    def test_r_squared_between_0_and_1(self):
        np.random.seed(10)
        bench = pd.Series(np.random.normal(0, 0.05, 50))
        strategy = pd.Series(np.random.normal(0, 0.05, 50))
        result = regress_vs_benchmark(strategy, bench)
        if not pd.isna(result["r_squared"]):
            assert 0.0 <= result["r_squared"] <= 1.0

    def test_nan_pairs_dropped_gracefully(self):
        bench = pd.Series([0.01, 0.02, np.nan, 0.04, 0.05,
                           0.01, 0.02, 0.03, 0.04, 0.05])
        strategy = pd.Series([0.02, 0.03, 0.04, np.nan, 0.06,
                               0.02, 0.03, 0.04, 0.05, 0.06])
        result = regress_vs_benchmark(strategy, bench, min_obs=5)
        # Should not crash; should return finite values
        assert not pd.isna(result["alpha"])
