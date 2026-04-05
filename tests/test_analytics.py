"""
Tests for factor_engine/analytics.py.

All expected values are hand-calculated from first principles.
"""

import numpy as np
import pandas as pd
import pytest

from factor_engine.analytics import (
    annualized_return,
    calmar_ratio,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


# ---------------------------------------------------------------------------
# annualized_return
# ---------------------------------------------------------------------------

class TestAnnualizedReturn:
    def test_12_months_1pct_monthly(self):
        """12 months of 1% → CAGR = 1.01^12 - 1."""
        returns = pd.Series([0.01] * 12)
        expected = (1.01 ** 12) - 1
        result = annualized_return(returns, periods_per_year=12)
        assert abs(result - expected) < 1e-10

    def test_all_zero_returns_gives_zero(self):
        returns = pd.Series([0.0, 0.0, 0.0, 0.0])
        result = annualized_return(returns, periods_per_year=12)
        assert result == 0.0

    def test_single_period_equals_itself(self):
        """1 annual period = 1 year → annualised = itself."""
        returns = pd.Series([0.10])
        result = annualized_return(returns, periods_per_year=1)
        assert abs(result - 0.10) < 1e-10

    def test_empty_series_returns_nan(self):
        result = annualized_return(pd.Series([], dtype=float))
        assert pd.isna(result)

    def test_positive_returns_positive_cagr(self):
        returns = pd.Series([0.02, 0.03, 0.01, 0.04])
        result = annualized_return(returns, periods_per_year=12)
        assert result > 0

    def test_geometric_compounding_vs_arithmetic(self):
        """Geometric CAGR should differ from arithmetic mean when returns vary."""
        returns = pd.Series([0.10, -0.10, 0.10, -0.10])
        cagr = annualized_return(returns, periods_per_year=4)
        arith = returns.mean() * 4
        # CAGR < arithmetic mean due to volatility drag
        assert cagr < arith


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_constant_returns_nan_not_infinity(self):
        """Zero volatility → NaN, never infinity."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        result = sharpe_ratio(returns)
        assert pd.isna(result)

    def test_positive_mean_with_noise_positive_sharpe(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.01, 0.03, 100))
        result = sharpe_ratio(returns, periods_per_year=12)
        assert result > 0

    def test_higher_mean_same_vol_higher_sharpe(self):
        np.random.seed(0)
        noise = np.random.normal(0, 0.02, 60)
        low_mean = pd.Series(noise + 0.005)
        high_mean = pd.Series(noise + 0.020)
        sharpe_low = sharpe_ratio(low_mean, periods_per_year=12)
        sharpe_high = sharpe_ratio(high_mean, periods_per_year=12)
        assert sharpe_high > sharpe_low

    def test_empty_series_returns_nan(self):
        result = sharpe_ratio(pd.Series([], dtype=float))
        assert pd.isna(result)

    def test_single_observation_returns_nan(self):
        result = sharpe_ratio(pd.Series([0.05]))
        assert pd.isna(result)


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    def test_all_positive_returns_nan(self):
        """No downside returns → downside deviation undefined → NaN."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.04])
        result = sortino_ratio(returns)
        assert pd.isna(result)

    def test_mix_positive_negative_finite_value(self):
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.04, -0.01])
        result = sortino_ratio(returns, periods_per_year=12)
        assert not pd.isna(result)
        assert np.isfinite(result)

    def test_empty_series_returns_nan(self):
        result = sortino_ratio(pd.Series([], dtype=float))
        assert pd.isna(result)

    def test_only_negative_returns_negative_sortino(self):
        returns = pd.Series([-0.01, -0.02, -0.03])
        result = sortino_ratio(returns, periods_per_year=12)
        assert result < 0


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_only_positive_returns_zero_drawdown(self):
        returns = pd.Series([0.01, 0.02, 0.03, 0.01])
        result = max_drawdown(returns)
        assert result == 0.0

    def test_known_sequence_exact_value(self):
        """
        Returns: [+10%, +5%, -30%, +15%, +10%, +5%]
        Cumulative wealth: 1.10, 1.155, 0.8085, 0.929775, 1.022752, 1.073890
        Peak before trough: 1.155 (after +5%)
        Trough: 0.8085 (after -30%)
        Max drawdown = (0.8085 - 1.155) / 1.155
        """
        returns = pd.Series([0.10, 0.05, -0.30, 0.15, 0.10, 0.05])
        cum = (1.0 + returns).cumprod()
        # Verify cumulative path
        assert abs(cum.iloc[2] - 0.8085) < 1e-6

        expected_mdd = (0.8085 - 1.155) / 1.155
        result = max_drawdown(returns)
        assert abs(result - expected_mdd) < 1e-10

    def test_always_negative_or_zero(self):
        """Max drawdown must never be positive."""
        returns = pd.Series([0.05, -0.10, 0.03, -0.05, 0.08])
        result = max_drawdown(returns)
        assert result <= 0.0

    def test_empty_series_returns_zero(self):
        result = max_drawdown(pd.Series([], dtype=float))
        assert result == 0.0

    def test_single_positive_return_zero_drawdown(self):
        result = max_drawdown(pd.Series([0.10]))
        assert result == 0.0

    def test_single_negative_return_is_that_return(self):
        """Single -30% period → drawdown = -30%."""
        result = max_drawdown(pd.Series([-0.30]))
        assert abs(result - (-0.30)) < 1e-10


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    def test_calmar_equals_ann_return_over_abs_mdd(self):
        returns = pd.Series([0.10, 0.05, -0.30, 0.15, 0.10, 0.05])
        expected = annualized_return(returns, 12) / abs(max_drawdown(returns))
        result = calmar_ratio(returns, periods_per_year=12)
        assert abs(result - expected) < 1e-10

    def test_zero_drawdown_returns_nan(self):
        """All positive returns → MDD=0 → Calmar undefined (divide by zero)."""
        returns = pd.Series([0.01, 0.02, 0.03])
        result = calmar_ratio(returns)
        assert pd.isna(result)

    def test_positive_return_negative_mdd_positive_calmar(self):
        returns = pd.Series([0.02, -0.01, 0.03, -0.01, 0.02])
        result = calmar_ratio(returns, periods_per_year=12)
        # If annualized return > 0 and MDD < 0, Calmar should be positive
        ann = annualized_return(returns, 12)
        mdd = max_drawdown(returns)
        if ann > 0 and mdd < 0:
            assert result > 0


# ---------------------------------------------------------------------------
# hit_rate
# ---------------------------------------------------------------------------

class TestHitRate:
    def test_always_outperform_hit_rate_1(self):
        strategy = pd.Series([0.02, 0.03, 0.04])
        benchmark = pd.Series([0.01, 0.01, 0.01])
        assert hit_rate(strategy, benchmark) == 1.0

    def test_never_outperform_hit_rate_0(self):
        strategy = pd.Series([0.00, 0.00, 0.00])
        benchmark = pd.Series([0.01, 0.01, 0.01])
        assert hit_rate(strategy, benchmark) == 0.0

    def test_two_out_of_four_hit_rate_05(self):
        strategy = pd.Series([0.02, 0.00, 0.02, 0.00])
        benchmark = pd.Series([0.01, 0.01, 0.01, 0.01])
        assert abs(hit_rate(strategy, benchmark) - 0.5) < 1e-10

    def test_empty_strategy_returns_nan(self):
        result = hit_rate(pd.Series([], dtype=float), pd.Series([], dtype=float))
        assert pd.isna(result)

    def test_tie_not_counted_as_outperformance(self):
        """Strategy == Benchmark for all periods → hit rate = 0."""
        returns = pd.Series([0.01, 0.02, 0.03])
        assert hit_rate(returns, returns) == 0.0

    def test_three_out_of_five(self):
        strategy = pd.Series([0.02, 0.00, 0.03, 0.00, 0.04])
        benchmark = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
        assert abs(hit_rate(strategy, benchmark) - 0.6) < 1e-10
