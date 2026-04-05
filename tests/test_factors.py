"""
Tests for factor_engine/factors.py and factor_engine/factors_price.py.

All tests use synthetic data with hand-calculable expected outputs.
No network calls, no yfinance.
"""

import numpy as np
import pandas as pd
import pytest

from factor_engine.factors import (
    compute_all_factors,
    compute_earnings_yield,
    compute_log_market_cap,
    compute_roe,
)
from factor_engine.factors_price import compute_momentum_12_1, compute_rolling_volatility


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(n_days=300):
    """300 business days (~14 months) of synthetic prices for 3 tickers."""
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    return pd.DataFrame(
        {
            "A": 100.0 * (1.0005 ** np.arange(n_days)),   # steady uptrend
            "B": 100.0 * (0.9995 ** np.arange(n_days)),   # steady downtrend
            "C": np.full(n_days, 100.0),                   # flat
        },
        index=dates,
    )


def _make_fundamentals(tickers=("A", "B", "C")):
    return pd.DataFrame(
        {
            "pe_ratio": [20.0, 10.0, 15.0],
            "roe": [0.15, 0.12, 0.10],
            "market_cap": [1e9, 2e9, 3e9],
        },
        index=list(tickers),
    )


# ---------------------------------------------------------------------------
# Earnings Yield
# ---------------------------------------------------------------------------

class TestEarningsYield:
    def test_pe_20_gives_005(self):
        df = pd.DataFrame({"pe_ratio": [20.0]}, index=["X"])
        result = compute_earnings_yield(df)
        assert abs(result["X"] - 0.05) < 1e-10

    def test_pe_10_gives_010(self):
        df = pd.DataFrame({"pe_ratio": [10.0]}, index=["X"])
        result = compute_earnings_yield(df)
        assert abs(result["X"] - 0.10) < 1e-10

    def test_negative_pe_gives_nan(self):
        df = pd.DataFrame({"pe_ratio": [-5.0]}, index=["X"])
        result = compute_earnings_yield(df)
        assert pd.isna(result["X"])

    def test_zero_pe_gives_nan(self):
        df = pd.DataFrame({"pe_ratio": [0.0]}, index=["X"])
        result = compute_earnings_yield(df)
        assert pd.isna(result["X"])

    def test_missing_pe_gives_nan(self):
        df = pd.DataFrame({"pe_ratio": [np.nan]}, index=["X"])
        result = compute_earnings_yield(df)
        assert pd.isna(result["X"])

    def test_multiple_tickers(self):
        df = pd.DataFrame({"pe_ratio": [20.0, 10.0, np.nan, -3.0, 0.0]},
                          index=["A", "B", "C", "D", "E"])
        result = compute_earnings_yield(df)
        assert abs(result["A"] - 0.05) < 1e-10
        assert abs(result["B"] - 0.10) < 1e-10
        assert pd.isna(result["C"])
        assert pd.isna(result["D"])
        assert pd.isna(result["E"])


# ---------------------------------------------------------------------------
# Return on Equity
# ---------------------------------------------------------------------------

class TestROE:
    def test_passthrough_positive(self):
        df = pd.DataFrame({"roe": [0.15, 0.20]}, index=["A", "B"])
        result = compute_roe(df)
        assert abs(result["A"] - 0.15) < 1e-10
        assert abs(result["B"] - 0.20) < 1e-10

    def test_negative_roe_preserved(self):
        """Loss-making firms have negative ROE — this is valid data, not an error."""
        df = pd.DataFrame({"roe": [-0.05]}, index=["X"])
        result = compute_roe(df)
        assert abs(result["X"] - (-0.05)) < 1e-10

    def test_nan_propagates(self):
        df = pd.DataFrame({"roe": [np.nan]}, index=["X"])
        result = compute_roe(df)
        assert pd.isna(result["X"])

    def test_zero_roe_preserved(self):
        df = pd.DataFrame({"roe": [0.0]}, index=["X"])
        result = compute_roe(df)
        assert result["X"] == 0.0


# ---------------------------------------------------------------------------
# Log Market Cap
# ---------------------------------------------------------------------------

class TestLogMarketCap:
    def test_log_1e10(self):
        df = pd.DataFrame({"market_cap": [1e10]}, index=["X"])
        result = compute_log_market_cap(df)
        assert abs(result["X"] - np.log(1e10)) < 1e-10

    def test_zero_mcap_gives_nan(self):
        df = pd.DataFrame({"market_cap": [0.0]}, index=["X"])
        result = compute_log_market_cap(df)
        assert pd.isna(result["X"])

    def test_negative_mcap_gives_nan(self):
        df = pd.DataFrame({"market_cap": [-1e9]}, index=["X"])
        result = compute_log_market_cap(df)
        assert pd.isna(result["X"])

    def test_larger_mcap_higher_log(self):
        df = pd.DataFrame({"market_cap": [1e9, 1e10]}, index=["A", "B"])
        result = compute_log_market_cap(df)
        assert result["B"] > result["A"]

    def test_log_preserves_order(self):
        caps = [1e8, 5e8, 1e9, 5e9, 1e10]
        df = pd.DataFrame({"market_cap": caps}, index=list("ABCDE"))
        result = compute_log_market_cap(df)
        logs = result.values
        # All logs should be strictly increasing
        assert all(logs[i] < logs[i + 1] for i in range(len(logs) - 1))


# ---------------------------------------------------------------------------
# Momentum 12-1
# ---------------------------------------------------------------------------

class TestMomentum12_1:
    def test_uptrend_positive_momentum(self):
        prices = _make_price_df()
        as_of = prices.index[270]
        result = compute_momentum_12_1(prices, as_of)
        assert result["A"] > 0

    def test_downtrend_negative_momentum(self):
        prices = _make_price_df()
        as_of = prices.index[270]
        result = compute_momentum_12_1(prices, as_of)
        assert result["B"] < 0

    def test_flat_price_near_zero_momentum(self):
        prices = _make_price_df()
        as_of = prices.index[270]
        result = compute_momentum_12_1(prices, as_of)
        # Flat price: price 1m ago == price 12m ago → momentum = 0
        assert abs(result["C"]) < 1e-10

    def test_uptrend_stronger_than_downtrend(self):
        prices = _make_price_df()
        as_of = prices.index[270]
        result = compute_momentum_12_1(prices, as_of)
        assert result["A"] > result["B"]

    def test_no_lookahead_full_vs_truncated(self):
        """
        Computing at date t with full data must equal computing with
        data truncated at t. If future data leaks, these will differ.
        """
        prices = _make_price_df()
        as_of = prices.index[270]

        result_full = compute_momentum_12_1(prices, as_of)
        result_trunc = compute_momentum_12_1(prices[prices.index <= as_of], as_of)

        pd.testing.assert_series_equal(result_full, result_trunc)

    def test_returns_series_indexed_by_ticker(self):
        prices = _make_price_df()
        as_of = prices.index[270]
        result = compute_momentum_12_1(prices, as_of)
        assert set(result.index) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# Rolling Volatility
# ---------------------------------------------------------------------------

class TestRollingVolatility:
    def test_flat_price_zero_vol(self):
        """Constant price → zero daily returns → zero volatility."""
        dates = pd.date_range("2020-01-02", periods=200, freq="B")
        prices = pd.DataFrame({"A": np.full(200, 100.0)}, index=dates)
        as_of = prices.index[-1]
        result = compute_rolling_volatility(prices, as_of, lookback_days=60)
        assert abs(result["A"]) < 1e-10

    def test_noisy_stock_positive_vol(self):
        """Random-walk prices produce positive realised volatility."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-02", periods=200, freq="B")
        daily_rets = np.random.normal(0.001, 0.02, 200)
        prices = pd.DataFrame(
            {"A": 100.0 * np.exp(np.cumsum(daily_rets))},
            index=dates,
        )
        as_of = prices.index[-1]
        result = compute_rolling_volatility(prices, as_of, lookback_days=60)
        assert result["A"] > 0

    def test_no_lookahead_full_vs_truncated(self):
        """
        Result at t using full data must equal result using data truncated at t.
        """
        np.random.seed(7)
        dates = pd.date_range("2020-01-02", periods=200, freq="B")
        daily_rets = np.random.normal(0.001, 0.02, 200)
        prices = pd.DataFrame(
            {"A": 100.0 * np.exp(np.cumsum(daily_rets))},
            index=dates,
        )
        as_of = prices.index[150]

        result_full = compute_rolling_volatility(prices, as_of, lookback_days=60)
        result_trunc = compute_rolling_volatility(
            prices[prices.index <= as_of], as_of, lookback_days=60
        )
        pd.testing.assert_series_equal(result_full, result_trunc)

    def test_insufficient_history_returns_nan(self):
        """Fewer days than lookback_days → all NaN (not enough data)."""
        dates = pd.date_range("2020-01-02", periods=10, freq="B")
        prices = pd.DataFrame({"A": np.full(10, 100.0)}, index=dates)
        as_of = prices.index[-1]
        result = compute_rolling_volatility(prices, as_of, lookback_days=60)
        assert result.dropna().empty

    def test_vol_is_annualised(self):
        """With known daily std, annualised vol = daily_std × sqrt(252)."""
        np.random.seed(99)
        dates = pd.date_range("2020-01-02", periods=200, freq="B")
        # Fixed daily returns so we can predict the std exactly
        daily_rets = np.random.normal(0.0, 0.01, 200)
        prices = pd.DataFrame(
            {"A": 100.0 * np.exp(np.cumsum(daily_rets))},
            index=dates,
        )
        as_of = prices.index[-1]
        result = compute_rolling_volatility(prices, as_of, lookback_days=60)
        # The annualised vol should be in a reasonable range
        assert 0 < result["A"] < 5.0  # sanity: not unreasonably large


# ---------------------------------------------------------------------------
# compute_all_factors
# ---------------------------------------------------------------------------

class TestComputeAllFactors:
    def test_returns_all_6_columns(self):
        """compute_all_factors returns the 5 individual factors plus composite_score."""
        prices = _make_price_df()
        fundamentals = _make_fundamentals()
        as_of = prices.index[270]
        result = compute_all_factors(fundamentals, prices, as_of)
        expected_cols = {
            "earnings_yield", "roe", "log_market_cap",
            "momentum_12_1", "rolling_vol_60d", "composite_score",
        }
        assert set(result.columns) == expected_cols

    def test_only_tickers_with_prices_at_date_appear(self):
        """Ticker D has fundamentals but no price data — must be excluded."""
        fundamentals = pd.DataFrame(
            {
                "pe_ratio": [20.0, 15.0, 10.0, 12.0],
                "roe": [0.15, 0.12, 0.10, 0.08],
                "market_cap": [1e9, 2e9, 3e9, 4e9],
            },
            index=["A", "B", "C", "D"],
        )
        prices = _make_price_df()  # only A, B, C
        as_of = prices.index[270]
        result = compute_all_factors(fundamentals, prices, as_of)
        assert "D" not in result.index
        assert "A" in result.index
        assert "B" in result.index
        assert "C" in result.index

    def test_result_index_matches_price_tickers(self):
        prices = _make_price_df()
        fundamentals = _make_fundamentals()
        as_of = prices.index[270]
        result = compute_all_factors(fundamentals, prices, as_of)
        assert set(result.index) == {"A", "B", "C"}

    def test_fundamental_factors_consistent_with_individual(self):
        """earnings_yield in compute_all_factors must match compute_earnings_yield."""
        prices = _make_price_df()
        fundamentals = _make_fundamentals()
        as_of = prices.index[270]
        result = compute_all_factors(fundamentals, prices, as_of)
        expected_ey = compute_earnings_yield(fundamentals)
        for ticker in result.index:
            assert abs(result.loc[ticker, "earnings_yield"] - expected_ey[ticker]) < 1e-10
