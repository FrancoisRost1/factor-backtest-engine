"""
Tests for factor_engine/quintiles.py and factor_engine/portfolio.py.

Covers assign_quintiles, construct_long_only, construct_long_short,
and compute_quintile_returns with synthetic, hand-calculable inputs.
"""

import numpy as np
import pandas as pd
import pytest

from factor_engine.portfolio import construct_long_only, construct_long_short
from factor_engine.quintiles import assign_quintiles, compute_quintile_returns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evenly_spaced_scores(n=20):
    """Scores 1..n evenly distributed across n tickers."""
    return pd.Series(
        range(1, n + 1),
        index=[f"T{i}" for i in range(n)],
        dtype=float,
    )


def _direct_quintiles(n=20):
    """Pre-assigned quintile labels cycling Q1–Q5 (4 tickers each for n=20)."""
    return pd.Series(
        [float(i % 5 + 1) for i in range(n)],
        index=[f"T{i}" for i in range(n)],
    )


# ---------------------------------------------------------------------------
# assign_quintiles
# ---------------------------------------------------------------------------

class TestAssignQuintiles:
    def test_20_stocks_4_per_quintile(self):
        scores = _evenly_spaced_scores(20)
        q = assign_quintiles(scores)
        for quintile in range(1, 6):
            count = (q == quintile).sum()
            assert count == 4, f"Quintile {quintile} has {count} stocks, expected 4"

    def test_q5_contains_highest_scores(self):
        scores = _evenly_spaced_scores(20)
        q = assign_quintiles(scores)
        q5_tickers = q[q == 5].index
        q1_tickers = q[q == 1].index
        # Every Q5 score > every Q1 score
        assert scores[q5_tickers].min() > scores[q1_tickers].max()

    def test_q1_contains_lowest_scores(self):
        scores = _evenly_spaced_scores(20)
        q = assign_quintiles(scores)
        q1_tickers = q[q == 1].index
        q5_tickers = q[q == 5].index
        assert scores[q1_tickers].max() < scores[q5_tickers].min()

    def test_nan_scores_get_nan_quintile(self):
        scores = pd.Series(
            [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            index=[f"T{i}" for i in range(10)],
        )
        q = assign_quintiles(scores)
        assert pd.isna(q["T2"])
        # Non-NaN tickers should have valid quintile labels
        assert not pd.isna(q["T0"])

    def test_fewer_than_5_stocks_all_nan(self):
        scores = pd.Series([1.0, 2.0, 3.0, 4.0], index=["A", "B", "C", "D"])
        q = assign_quintiles(scores)
        assert q.isna().all()

    def test_exactly_5_stocks_one_per_quintile(self):
        scores = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0],
                           index=["A", "B", "C", "D", "E"])
        q = assign_quintiles(scores)
        for quintile in range(1, 6):
            assert (q == quintile).sum() == 1

    def test_output_index_matches_input_index(self):
        scores = _evenly_spaced_scores(20)
        q = assign_quintiles(scores)
        assert list(q.index) == list(scores.index)


# ---------------------------------------------------------------------------
# construct_long_only
# ---------------------------------------------------------------------------

class TestConstructLongOnly:
    def test_equal_weight_sums_to_1(self):
        q = _direct_quintiles(20)
        weights = construct_long_only(q, weighting="equal")
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_only_q5_tickers_in_output(self):
        q = _direct_quintiles(20)
        weights = construct_long_only(q, weighting="equal")
        q5_tickers = set(q[q == 5].index)
        assert set(weights.index) == q5_tickers

    def test_equal_weight_all_identical(self):
        q = pd.Series({"A": 5.0, "B": 5.0, "C": 5.0, "D": 1.0, "E": 1.0})
        weights = construct_long_only(q, weighting="equal")
        # All Q5 stocks get weight 1/3
        assert abs(weights["A"] - weights["B"]) < 1e-10
        assert abs(weights["B"] - weights["C"]) < 1e-10
        assert abs(weights["A"] - 1.0 / 3) < 1e-10

    def test_cap_weight_sums_to_1(self):
        q = pd.Series({"A": 5.0, "B": 5.0, "C": 1.0})
        mcaps = pd.Series({"A": 1e9, "B": 2e9, "C": 3e9})
        weights = construct_long_only(q, market_caps=mcaps, weighting="cap_weight")
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_cap_weight_larger_mcap_higher_weight(self):
        q = pd.Series({"A": 5.0, "B": 5.0, "C": 1.0})
        mcaps = pd.Series({"A": 1e9, "B": 3e9, "C": 2e9})
        weights = construct_long_only(q, market_caps=mcaps, weighting="cap_weight")
        assert weights["B"] > weights["A"]

    def test_cap_weight_proportional_to_mcap(self):
        """A with 1/3 of total cap should get weight 1/3."""
        q = pd.Series({"A": 5.0, "B": 5.0, "C": 5.0})
        mcaps = pd.Series({"A": 1e9, "B": 2e9, "C": 3e9})
        weights = construct_long_only(q, market_caps=mcaps, weighting="cap_weight")
        total_cap = 6e9
        assert abs(weights["A"] - 1e9 / total_cap) < 1e-10
        assert abs(weights["B"] - 2e9 / total_cap) < 1e-10
        assert abs(weights["C"] - 3e9 / total_cap) < 1e-10

    def test_no_negative_weights(self):
        q = _direct_quintiles(20)
        weights = construct_long_only(q, weighting="equal")
        assert (weights >= 0).all()

    def test_empty_q5_returns_empty_series(self):
        q = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
        weights = construct_long_only(q, weighting="equal")
        assert len(weights) == 0

    def test_cap_weight_all_nan_falls_back_to_equal(self):
        """All market caps NaN → must fall back to equal weight, not zero allocation."""
        q = pd.Series({"A": 5.0, "B": 5.0, "C": 5.0, "D": 1.0})
        mcaps = pd.Series({"A": np.nan, "B": np.nan, "C": np.nan, "D": np.nan})
        weights = construct_long_only(q, market_caps=mcaps, weighting="cap_weight")
        assert abs(weights.sum() - 1.0) < 1e-10, "Weights must sum to 1 even with NaN caps"
        # Each Q5 ticker should get equal weight 1/3
        for ticker in ["A", "B", "C"]:
            assert abs(weights[ticker] - 1.0 / 3) < 1e-10

    def test_cap_weight_all_zero_falls_back_to_equal(self):
        """All market caps zero → must fall back to equal weight."""
        q = pd.Series({"A": 5.0, "B": 5.0, "D": 1.0})
        mcaps = pd.Series({"A": 0.0, "B": 0.0, "D": 0.0})
        weights = construct_long_only(q, market_caps=mcaps, weighting="cap_weight")
        assert abs(weights.sum() - 1.0) < 1e-10
        assert abs(weights["A"] - 0.5) < 1e-10
        assert abs(weights["B"] - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# construct_long_short
# ---------------------------------------------------------------------------

class TestConstructLongShort:
    def test_weights_sum_to_zero_equal(self):
        q = _direct_quintiles(20)
        weights = construct_long_short(q, weighting="equal")
        assert abs(weights.sum()) < 1e-10

    def test_weights_sum_to_zero_cap_weight(self):
        q = _direct_quintiles(20)
        mcaps = pd.Series({f"T{i}": float(i + 1) * 1e9 for i in range(20)})
        weights = construct_long_short(q, market_caps=mcaps, weighting="cap_weight")
        assert abs(weights.sum()) < 1e-10

    def test_q5_positive_weights(self):
        q = _direct_quintiles(20)
        weights = construct_long_short(q, weighting="equal")
        q5_tickers = q[q == 5].index
        assert (weights[q5_tickers] > 0).all()

    def test_q1_negative_weights(self):
        q = _direct_quintiles(20)
        weights = construct_long_short(q, weighting="equal")
        q1_tickers = q[q == 1].index
        assert (weights[q1_tickers] < 0).all()

    def test_long_side_sums_to_plus_05(self):
        q = _direct_quintiles(20)
        weights = construct_long_short(q, weighting="equal")
        long_sum = weights[weights > 0].sum()
        assert abs(long_sum - 0.5) < 1e-10

    def test_short_side_sums_to_minus_05(self):
        q = _direct_quintiles(20)
        weights = construct_long_short(q, weighting="equal")
        short_sum = weights[weights < 0].sum()
        assert abs(short_sum - (-0.5)) < 1e-10

    def test_q2_q3_q4_not_in_output(self):
        """Middle quintiles are not traded — must not appear in the weight index."""
        q = _direct_quintiles(20)
        weights = construct_long_short(q, weighting="equal")
        for quintile in [2, 3, 4]:
            qx_tickers = q[q == quintile].index
            for ticker in qx_tickers:
                assert ticker not in weights.index

    def test_only_q5_and_q1_in_output(self):
        q = _direct_quintiles(20)
        weights = construct_long_short(q, weighting="equal")
        q5 = set(q[q == 5].index)
        q1 = set(q[q == 1].index)
        assert set(weights.index) == q5 | q1

    def test_cap_weight_all_nan_falls_back_to_equal(self):
        """All market caps NaN → both legs must fall back to equal weight."""
        q = pd.Series({"A": 5.0, "B": 5.0, "C": 1.0, "D": 1.0})
        mcaps = pd.Series({"A": np.nan, "B": np.nan, "C": np.nan, "D": np.nan})
        weights = construct_long_short(q, market_caps=mcaps, weighting="cap_weight")
        # Net sum must still be zero
        assert abs(weights.sum()) < 1e-10
        # Long leg: A and B each +0.25 (equal weight within +0.5 target)
        assert abs(weights["A"] - 0.25) < 1e-10
        assert abs(weights["B"] - 0.25) < 1e-10
        # Short leg: C and D each -0.25 (equal weight within -0.5 target)
        assert abs(weights["C"] - (-0.25)) < 1e-10
        assert abs(weights["D"] - (-0.25)) < 1e-10

    def test_cap_weight_all_zero_falls_back_to_equal(self):
        """All market caps zero → both legs must fall back to equal weight."""
        q = pd.Series({"A": 5.0, "B": 5.0, "C": 1.0, "D": 1.0})
        mcaps = pd.Series({"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0})
        weights = construct_long_short(q, market_caps=mcaps, weighting="cap_weight")
        assert abs(weights.sum()) < 1e-10
        assert abs(weights["A"] - 0.25) < 1e-10
        assert abs(weights["B"] - 0.25) < 1e-10


# ---------------------------------------------------------------------------
# compute_quintile_returns
# ---------------------------------------------------------------------------

class TestComputeQuintileReturns:
    def test_returns_value_for_each_quintile(self):
        q = _direct_quintiles(20)
        returns = pd.Series(
            np.ones(20) * 0.01, index=[f"T{i}" for i in range(20)]
        )
        qr = compute_quintile_returns(q, returns)
        for quintile in range(1, 6):
            assert quintile in qr.index
            assert not pd.isna(qr[quintile])

    def test_equal_weight_return_equals_mean(self):
        """Q1 return = mean(A, B), Q5 return = mean(C, D). Hand-calculated."""
        q = pd.Series({"A": 1.0, "B": 1.0, "C": 5.0, "D": 5.0})
        returns = pd.Series({"A": 0.01, "B": 0.03, "C": 0.05, "D": 0.07})
        qr = compute_quintile_returns(q, returns)
        # Q1: (0.01 + 0.03) / 2 = 0.02
        assert abs(qr[1] - 0.02) < 1e-10
        # Q5: (0.05 + 0.07) / 2 = 0.06
        assert abs(qr[5] - 0.06) < 1e-10

    def test_empty_quintile_returns_nan(self):
        """Q3 and Q4 are empty → NaN."""
        q = pd.Series({"A": 1.0, "B": 5.0})
        returns = pd.Series({"A": 0.01, "B": 0.05})
        qr = compute_quintile_returns(q, returns)
        assert pd.isna(qr[3])
        assert pd.isna(qr[4])

    def test_uniform_returns_all_quintiles_equal(self):
        q = _direct_quintiles(20)
        returns = pd.Series(np.full(20, 0.02), index=[f"T{i}" for i in range(20)])
        qr = compute_quintile_returns(q, returns)
        for quintile in range(1, 6):
            assert abs(qr[quintile] - 0.02) < 1e-10
