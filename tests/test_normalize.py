"""
Tests for factor_engine/normalize.py — percentile_rank function.

All tests use synthetic data with known expected rank outcomes.
"""

import numpy as np
import pandas as pd
import pytest

from factor_engine.normalize import percentile_rank


class TestOutputRange:
    def test_all_values_between_0_and_1(self):
        df = pd.DataFrame({"ey": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = percentile_rank(df, invert_columns=[])
        assert (result["ey"] >= 0.0).all()
        assert (result["ey"] <= 1.0).all()

    def test_max_rank_is_1(self):
        df = pd.DataFrame({"ey": [10.0, 20.0, 30.0]})
        result = percentile_rank(df, invert_columns=[])
        assert result["ey"].max() == 1.0

    def test_min_rank_is_1_over_n(self):
        n = 5
        df = pd.DataFrame({"ey": list(range(n))})
        result = percentile_rank(df, invert_columns=[])
        assert abs(result["ey"].min() - 1.0 / n) < 1e-10


class TestNaNPreservation:
    def test_nan_in_gives_nan_out(self):
        df = pd.DataFrame({"ey": [1.0, np.nan, 3.0]})
        result = percentile_rank(df, invert_columns=[])
        assert pd.isna(result["ey"].iloc[1])

    def test_non_nan_still_ranked(self):
        df = pd.DataFrame({"ey": [1.0, np.nan, 3.0]})
        result = percentile_rank(df, invert_columns=[])
        assert not pd.isna(result["ey"].iloc[0])
        assert not pd.isna(result["ey"].iloc[2])

    def test_all_nan_all_nan_out(self):
        df = pd.DataFrame({"ey": [np.nan, np.nan, np.nan]})
        result = percentile_rank(df, invert_columns=[])
        assert result["ey"].isna().all()


class TestNormalOrdering:
    def test_earnings_yield_higher_value_higher_rank(self):
        """For a normal factor: highest raw value → rank 1.0."""
        df = pd.DataFrame(
            {"earnings_yield": [0.05, 0.10, 0.15]},
            index=["low", "mid", "high"],
        )
        result = percentile_rank(df, invert_columns=[])
        ranks = result["earnings_yield"]
        assert ranks["high"] > ranks["mid"] > ranks["low"]

    def test_highest_value_gets_rank_1(self):
        df = pd.DataFrame({"ey": [10.0, 20.0, 30.0]}, index=["A", "B", "C"])
        result = percentile_rank(df, invert_columns=[])
        assert result["ey"]["C"] == 1.0

    def test_lowest_value_gets_rank_1_over_n(self):
        df = pd.DataFrame({"ey": [10.0, 20.0, 30.0]}, index=["A", "B", "C"])
        result = percentile_rank(df, invert_columns=[])
        assert abs(result["ey"]["A"] - 1.0 / 3) < 1e-10


class TestInvertedOrdering:
    def test_log_market_cap_inverted_by_default(self):
        """log_market_cap: smallest raw value should get the highest rank."""
        df = pd.DataFrame(
            {"log_market_cap": [1.0, 2.0, 3.0]},
            index=["small", "mid", "large"],
        )
        result = percentile_rank(df)  # uses default invert_columns
        ranks = result["log_market_cap"]
        assert ranks["small"] > ranks["mid"] > ranks["large"]

    def test_log_market_cap_smallest_gets_rank_1(self):
        df = pd.DataFrame({"log_market_cap": [1.0, 2.0, 3.0]}, index=["A", "B", "C"])
        result = percentile_rank(df)
        assert result["log_market_cap"]["A"] == 1.0

    def test_rolling_vol_inverted_by_default(self):
        """rolling_vol_60d: lowest volatility should get the highest rank."""
        df = pd.DataFrame(
            {"rolling_vol_60d": [0.10, 0.20, 0.30]},
            index=["low", "mid", "high"],
        )
        result = percentile_rank(df)
        ranks = result["rolling_vol_60d"]
        assert ranks["low"] > ranks["mid"] > ranks["high"]

    def test_rolling_vol_lowest_gets_rank_1(self):
        df = pd.DataFrame({"rolling_vol_60d": [0.10, 0.20, 0.30]}, index=["A", "B", "C"])
        result = percentile_rank(df)
        assert result["rolling_vol_60d"]["A"] == 1.0


class TestEdgeCases:
    def test_all_same_values_get_same_rank(self):
        df = pd.DataFrame({"ey": [5.0, 5.0, 5.0]})
        result = percentile_rank(df, invert_columns=[])
        # All values identical → all get the same rank
        assert result["ey"].nunique() == 1

    def test_single_stock_gets_rank_1(self):
        df = pd.DataFrame({"ey": [0.10]}, index=["A"])
        result = percentile_rank(df, invert_columns=[])
        assert result["ey"]["A"] == 1.0

    def test_single_stock_inverted_still_gets_rank_1(self):
        df = pd.DataFrame({"log_market_cap": [20.0]}, index=["A"])
        result = percentile_rank(df)  # default invert
        assert result["log_market_cap"]["A"] == 1.0


class TestCustomInvertColumns:
    def test_empty_invert_columns_makes_all_ascending(self):
        """With invert_columns=[], log_market_cap ranks ascending (larger = higher rank)."""
        df = pd.DataFrame(
            {"log_market_cap": [1.0, 2.0, 3.0]},
            index=["small", "mid", "large"],
        )
        result = percentile_rank(df, invert_columns=[])
        ranks = result["log_market_cap"]
        # Without inversion: larger raw value gets higher rank
        assert ranks["large"] > ranks["mid"] > ranks["small"]

    def test_custom_invert_overrides_default(self):
        """Explicitly inverting 'ey' (normally ascending) should flip it."""
        df = pd.DataFrame({"ey": [0.05, 0.10, 0.15]}, index=["low", "mid", "high"])
        result = percentile_rank(df, invert_columns=["ey"])
        ranks = result["ey"]
        # Now lowest raw value gets highest rank
        assert ranks["low"] > ranks["mid"] > ranks["high"]

    def test_multiple_columns_independently_ranked(self):
        """Each column gets its own independent rank, not across columns."""
        df = pd.DataFrame(
            {
                "earnings_yield": [0.05, 0.10, 0.15],
                "roe": [0.30, 0.20, 0.10],
            }
        )
        result = percentile_rank(df, invert_columns=[])
        # earnings_yield: 0.15 should rank highest
        assert result["earnings_yield"].iloc[2] == 1.0
        # roe: 0.30 should rank highest
        assert result["roe"].iloc[0] == 1.0
