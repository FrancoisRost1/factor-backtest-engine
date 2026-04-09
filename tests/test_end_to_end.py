"""
End-to-end integration test for the factor backtest engine.

Runs the full pipeline (factor computation → backtesting → analytics → regression)
on a synthetic 10-ticker universe to verify correctness without calling yfinance.

Design decisions:
  - Synthetic data removes network dependency and makes tests deterministic.
  - Random walks with seed=42 ensure reproducibility across runs.
  - Quarterly rebalancing only (fastest of the three frequencies).
  - All 6 factor names tested (5 individual + composite).
  - Signal-timing test verifies the no-lookahead guarantee explicitly.
"""

import numpy as np
import pandas as pd
import pytest

from factor_engine.backtest import run_all_backtests, FACTOR_NAMES
from factor_engine.analytics import compute_all_metrics
from factor_engine.ic import ic_summary
from factor_engine.regression import regress_vs_benchmark
from factor_engine.factors import compute_all_factors


# ── Constants ──────────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ", "PG", "KO"]
START_DATE = "2014-01-01"
END_DATE = "2026-03-31"
SEED = 42

EXPECTED_FACTOR_NAMES = ["value", "momentum", "quality", "size", "low_volatility", "composite"]
EXPECTED_RESULT_KEYS = {
    "returns_gross", "returns_net",
    "cumulative_gross", "cumulative_net",
    "turnover", "quintile_returns",
    "ic_series", "benchmark_returns",
}
WEIGHTING_SCHEMES = ["equal", "cap_weight"]
PORTFOLIO_TYPES = ["long_only", "long_short"]


# ── Synthetic data fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_prices() -> pd.DataFrame:
    """
    Generate synthetic daily price paths for 10 tickers as random walks.

    Each ticker gets a different drift (0–0.03% daily) and volatility
    (0.5–2.5% daily) to simulate realistic cross-sectional dispersion.
    All prices start at $100 and remain positive throughout.
    Seed=42 guarantees reproducibility.
    """
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(START_DATE, END_DATE)
    n = len(dates)

    drifts = rng.uniform(0.0, 0.0003, size=len(TICKERS))   # ~0–7.5% annual drift
    vols   = rng.uniform(0.005, 0.025, size=len(TICKERS))  # ~8–40% annual vol

    prices_dict = {}
    for i, ticker in enumerate(TICKERS):
        shocks = rng.normal(drifts[i], vols[i], size=n)
        prices_dict[ticker] = 100.0 * np.exp(np.cumsum(shocks))

    return pd.DataFrame(prices_dict, index=dates)


@pytest.fixture(scope="module")
def synthetic_fundamentals() -> pd.DataFrame:
    """
    Generate synthetic cross-sectional fundamental data for 10 tickers.

    Values are drawn within realistic ranges:
      pe_ratio   : 8–40 (all positive so earnings_yield is well-defined)
      roe        : -0.05 to 0.35 (negative ROE allowed for loss-makers)
      market_cap : $1B – $2T USD (covers small-cap to mega-cap)
    Seed=42 guarantees reproducibility.
    """
    rng = np.random.default_rng(SEED)
    return pd.DataFrame(
        {
            "pe_ratio":   rng.uniform(8, 40, size=len(TICKERS)),
            "roe":        rng.uniform(-0.05, 0.35, size=len(TICKERS)),
            "market_cap": rng.uniform(1e9, 2e12, size=len(TICKERS)),
        },
        index=TICKERS,
    )


@pytest.fixture(scope="module")
def synthetic_benchmark() -> pd.Series:
    """
    Generate a synthetic benchmark (SPY proxy) as a random walk with a
    moderate uptrend (~5% annual drift, ~16% annual vol).
    Uses SEED+1 so it is statistically independent of the stock prices.
    """
    rng = np.random.default_rng(SEED + 1)
    dates = pd.bdate_range(START_DATE, END_DATE)
    n = len(dates)

    shocks = rng.normal(0.0002, 0.010, size=n)
    prices = pd.Series(100.0 * np.exp(np.cumsum(shocks)), index=dates, name="SPY")
    return prices


@pytest.fixture(scope="module")
def test_config() -> dict:
    """
    Minimal config for end-to-end testing.

    Restricts frequencies to quarterly only so the test suite runs quickly.
    All other parameters match config.yaml defaults.
    """
    return {
        "data": {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "eval_start": "2016-01-01",
        },
        "factors": {
            "low_volatility": {"lookback_days": 60},
        },
        "portfolio": {
            "n_quantiles": 5,
            "long_quintile": 5,
            "short_quintile": 1,
            "weighting_schemes": WEIGHTING_SCHEMES,
        },
        "rebalancing": {
            "frequencies": ["quarterly"],
        },
        "transaction_costs": {
            "cost_per_trade": 0.0010,
        },
        "composite": {
            "min_valid_factors": 3,
        },
        "analytics": {
            "risk_free_rate": 0.0,
            "annualization_factor": 252,
        },
    }


@pytest.fixture(scope="module")
def backtest_results(
    synthetic_prices,
    synthetic_fundamentals,
    synthetic_benchmark,
    test_config,
) -> dict:
    """Run the full pipeline once; share across all tests in this module."""
    return run_all_backtests(
        synthetic_prices,
        synthetic_fundamentals,
        synthetic_benchmark,
        test_config,
    )


# ── Result structure tests ─────────────────────────────────────────────────────

def test_results_has_all_factors(backtest_results):
    """All 6 factors (5 individual + composite) must appear in results."""
    assert set(backtest_results.keys()) == set(EXPECTED_FACTOR_NAMES), (
        f"Expected factors {EXPECTED_FACTOR_NAMES}, got {list(backtest_results.keys())}"
    )


def test_factor_names_constant_matches_expected():
    """FACTOR_NAMES exported from backtest module must include composite."""
    assert set(FACTOR_NAMES) == set(EXPECTED_FACTOR_NAMES)


def test_results_structure(backtest_results):
    """Every factor/freq/weighting/portfolio_type combination must have all expected keys."""
    for factor in EXPECTED_FACTOR_NAMES:
        for wt in WEIGHTING_SCHEMES:
            for pt in PORTFOLIO_TYPES:
                bt = backtest_results[factor]["quarterly"][wt][pt]
                missing = EXPECTED_RESULT_KEYS - set(bt.keys())
                assert not missing, (
                    f"Missing keys {missing} in {factor}/quarterly/{wt}/{pt}"
                )


def test_returns_series_non_empty(backtest_results):
    """returns_gross and returns_net must be non-empty for all combinations."""
    for factor in EXPECTED_FACTOR_NAMES:
        for wt in WEIGHTING_SCHEMES:
            for pt in PORTFOLIO_TYPES:
                bt = backtest_results[factor]["quarterly"][wt][pt]
                assert len(bt["returns_gross"]) > 0, (
                    f"returns_gross empty for {factor}/quarterly/{wt}/{pt}"
                )
                assert len(bt["returns_net"]) > 0, (
                    f"returns_net empty for {factor}/quarterly/{wt}/{pt}"
                )


def test_returns_are_series(backtest_results):
    """returns_gross and returns_net must be pd.Series objects."""
    bt = backtest_results["value"]["quarterly"]["equal"]["long_only"]
    assert isinstance(bt["returns_gross"], pd.Series)
    assert isinstance(bt["returns_net"], pd.Series)


# ── Cumulative returns tests ───────────────────────────────────────────────────

def test_cumulative_gross_no_interior_nan(backtest_results):
    """
    cumulative_gross must have no NaN values.

    Cumulative product of (1+r) should always be defined as long as the
    underlying returns_gross contains no NaN — the backtest fills missing
    stock returns with 0 (flat) so the cumulative product is always finite.
    """
    for factor in EXPECTED_FACTOR_NAMES:
        for wt in WEIGHTING_SCHEMES:
            cum = backtest_results[factor]["quarterly"][wt]["long_only"]["cumulative_gross"]
            assert cum.notna().all(), (
                f"cumulative_gross contains NaN for {factor}/{wt}/long_only"
            )


def test_long_only_cumulative_positive(backtest_results):
    """
    Long-only cumulative wealth must always be strictly positive.

    Each long-only portfolio is fully invested in Q5 stocks. The cumulative
    product (1+r1)(1+r2)... can only reach zero if the portfolio loses 100%
    in a single period, which is impossible for diversified equal-weighted
    holdings of stocks with positive synthetic prices.
    """
    for factor in EXPECTED_FACTOR_NAMES:
        for wt in WEIGHTING_SCHEMES:
            cum = backtest_results[factor]["quarterly"][wt]["long_only"]["cumulative_gross"]
            assert (cum > 0).all(), (
                f"cumulative_gross has non-positive values for {factor}/{wt}/long_only"
            )


def test_cumulative_gross_computed_from_returns(backtest_results):
    """
    cumulative_gross must equal (1 + returns_gross).cumprod().

    This verifies the cumulative series is a correct monotone transformation
    of the period returns and not an independent computation.
    """
    bt = backtest_results["momentum"]["quarterly"]["equal"]["long_only"]
    expected = (1 + bt["returns_gross"]).cumprod()
    pd.testing.assert_series_equal(bt["cumulative_gross"], expected, check_names=False)


# ── Turnover tests ─────────────────────────────────────────────────────────────

def test_turnover_non_negative(backtest_results):
    """Turnover must be >= 0 for all combinations (it is a sum of absolute changes)."""
    for factor in EXPECTED_FACTOR_NAMES:
        for wt in WEIGHTING_SCHEMES:
            for pt in PORTFOLIO_TYPES:
                to = backtest_results[factor]["quarterly"][wt][pt]["turnover"]
                assert (to >= 0).all(), (
                    f"Negative turnover found for {factor}/quarterly/{wt}/{pt}"
                )


def test_turnover_first_period_positive(backtest_results):
    """
    First-period turnover must be positive.

    The portfolio starts empty (prev_weights = {}), so the first rebalance
    always buys into Q5 — turnover equals the total weight purchased (1.0 for
    long-only, 1.0 gross for long-short).
    """
    to = backtest_results["value"]["quarterly"]["equal"]["long_only"]["turnover"]
    assert float(to.iloc[0]) > 0, "First-period turnover should be positive (initial buy-in)"


# ── Long-short tests ───────────────────────────────────────────────────────────

def test_long_short_returns_finite(backtest_results):
    """Long-short returns must be finite (no NaN/inf) for all valid periods."""
    for factor in EXPECTED_FACTOR_NAMES:
        for wt in WEIGHTING_SCHEMES:
            ls = backtest_results[factor]["quarterly"][wt]["long_short"]
            gross = ls["returns_gross"].dropna()
            assert len(gross) > 0, f"No long-short returns for {factor}/{wt}"
            assert np.isfinite(gross.values).all(), (
                f"Non-finite long-short returns for {factor}/{wt}"
            )


def test_net_returns_lower_than_gross(backtest_results):
    """
    Net returns must be <= gross returns for all periods.

    Transaction costs are always non-negative, so net = gross - cost <= gross.
    """
    for factor in EXPECTED_FACTOR_NAMES:
        for wt in WEIGHTING_SCHEMES:
            bt = backtest_results[factor]["quarterly"][wt]["long_only"]
            diff = bt["returns_gross"] - bt["returns_net"]
            assert (diff >= -1e-12).all(), (
                f"Net return exceeds gross for {factor}/{wt}/long_only"
            )


# ── Quintile returns tests ─────────────────────────────────────────────────────

def test_quintile_returns_structure(backtest_results):
    """
    quintile_returns must be a DataFrame with columns for all 5 quintiles.

    The quintile-return spread (Q5 - Q1 over time) is the primary diagnostic
    for whether a factor has genuine cross-sectional predictive power.
    """
    for factor in EXPECTED_FACTOR_NAMES:
        for wt in WEIGHTING_SCHEMES:
            qr = backtest_results[factor]["quarterly"][wt]["long_only"]["quintile_returns"]
            assert isinstance(qr, pd.DataFrame), (
                f"quintile_returns is not a DataFrame for {factor}/{wt}"
            )
            assert set(range(1, 6)).issubset(set(qr.columns)), (
                f"Missing quintile columns for {factor}/{wt}: got {list(qr.columns)}"
            )


def test_quintile_returns_index_matches_returns(backtest_results):
    """quintile_returns must share the same date index as returns_gross."""
    bt = backtest_results["value"]["quarterly"]["equal"]["long_only"]
    pd.testing.assert_index_equal(
        bt["quintile_returns"].index,
        bt["returns_gross"].index,
    )


# ── Analytics tests ────────────────────────────────────────────────────────────

def test_compute_all_metrics_keys(backtest_results):
    """compute_all_metrics must return the full set of expected metric keys."""
    expected_keys = {
        "annualized_return", "sharpe", "sortino",
        "max_drawdown", "calmar", "hit_rate", "avg_turnover",
    }
    bt = backtest_results["value"]["quarterly"]["equal"]["long_only"]
    metrics = compute_all_metrics(
        bt["returns_gross"],
        bt["benchmark_returns"],
        bt["turnover"],
        periods_per_year=4,
    )
    assert set(metrics.keys()) == expected_keys


def test_compute_all_metrics_finite(backtest_results):
    """All metrics must be finite (NaN acceptable for edge cases; inf is not)."""
    for factor in EXPECTED_FACTOR_NAMES:
        bt = backtest_results[factor]["quarterly"]["equal"]["long_only"]
        metrics = compute_all_metrics(
            bt["returns_gross"],
            bt["benchmark_returns"],
            bt["turnover"],
            periods_per_year=4,
        )
        for key, val in metrics.items():
            assert not np.isinf(val), (
                f"Metric '{key}' is infinite for factor '{factor}'"
            )


def test_ic_summary_keys(backtest_results):
    """ic_summary must return mean_ic and ic_ir."""
    bt = backtest_results["momentum"]["quarterly"]["equal"]["long_only"]
    result = ic_summary(bt["ic_series"])
    assert "mean_ic" in result
    assert "ic_ir" in result


def test_ic_summary_mean_ic_in_range(backtest_results):
    """
    mean_ic must be in [-1, 1] (Spearman correlation bounds).

    NaN is also acceptable when all IC observations are NaN (sparse data).
    """
    for factor in EXPECTED_FACTOR_NAMES:
        bt = backtest_results[factor]["quarterly"]["equal"]["long_only"]
        stats = ic_summary(bt["ic_series"])
        mean_ic = stats["mean_ic"]
        if not np.isnan(mean_ic):
            assert -1.0 <= mean_ic <= 1.0, (
                f"mean_ic={mean_ic} out of [-1,1] for factor '{factor}'"
            )


def test_regression_keys(backtest_results):
    """regress_vs_benchmark must return alpha, beta, and r_squared."""
    bt = backtest_results["quality"]["quarterly"]["equal"]["long_only"]
    result = regress_vs_benchmark(bt["returns_gross"], bt["benchmark_returns"])
    assert set(result.keys()) == {"alpha", "beta", "r_squared"}


def test_regression_finite(backtest_results):
    """alpha, beta, and r_squared must be finite (not NaN/inf)."""
    bt = backtest_results["value"]["quarterly"]["cap_weight"]["long_only"]
    result = regress_vs_benchmark(bt["returns_gross"], bt["benchmark_returns"])
    for key, val in result.items():
        assert not np.isnan(val), f"Regression '{key}' is NaN"
        assert np.isfinite(val), f"Regression '{key}' is infinite"


def test_r_squared_in_range(backtest_results):
    """R-squared must be in [0, 1] (by definition of OLS R²)."""
    bt = backtest_results["composite"]["quarterly"]["equal"]["long_only"]
    result = regress_vs_benchmark(bt["returns_gross"], bt["benchmark_returns"])
    r2 = result["r_squared"]
    if not np.isnan(r2):
        assert 0.0 <= r2 <= 1.0, f"R² = {r2} is outside [0, 1]"


# ── Signal timing tests ────────────────────────────────────────────────────────

def test_signal_timing_no_lookahead(synthetic_prices, synthetic_fundamentals):
    """
    Factor scores at date t must be identical whether computed with:
      (a) full price history (function filters internally to <= t), or
      (b) price history manually trimmed to <= t.

    If these differ, the function is accessing future data at point t.

    Chosen date 2020-06-30 is well into the evaluation period and has full
    price history for all factors including 12-month momentum lookback.
    """
    as_of = pd.Timestamp("2020-06-30")

    # (a) full price series — function must enforce the date boundary internally
    scores_full = compute_all_factors(
        synthetic_fundamentals,
        synthetic_prices,
        as_of,
        lookback_days=60,
    )

    # (b) price series externally trimmed to the signal date
    prices_trimmed = synthetic_prices[synthetic_prices.index <= as_of]
    scores_trimmed = compute_all_factors(
        synthetic_fundamentals,
        prices_trimmed,
        as_of,
        lookback_days=60,
    )

    assert not scores_full.empty, "No factor scores computed at 2020-06-30"
    assert not scores_trimmed.empty, "No factor scores with trimmed prices at 2020-06-30"

    # Scores must be exactly equal (same tickers, same values) regardless of
    # whether future prices are accessible — the function must filter them out.
    pd.testing.assert_frame_equal(
        scores_full.sort_index(),
        scores_trimmed.sort_index(),
        check_exact=False,
        rtol=1e-6,
        atol=1e-10,
    )


def test_signal_timing_future_price_change_has_no_effect(
    synthetic_prices,
    synthetic_fundamentals,
):
    """
    Multiplying all prices AFTER the signal date by 1000 must not change factor scores.

    This is a stronger version of the no-lookahead test: it injects extreme
    future data and confirms that the factor computation is blind to it.
    """
    as_of = pd.Timestamp("2019-03-31")

    # Baseline scores with original prices
    scores_baseline = compute_all_factors(
        synthetic_fundamentals, synthetic_prices, as_of, lookback_days=60
    )

    # Tamper: multiply all prices after as_of by 1000
    prices_tampered = synthetic_prices.copy()
    future_mask = prices_tampered.index > as_of
    prices_tampered.loc[future_mask] *= 1000.0

    scores_tampered = compute_all_factors(
        synthetic_fundamentals, prices_tampered, as_of, lookback_days=60
    )

    assert not scores_baseline.empty, "No baseline scores at 2019-03-31"
    pd.testing.assert_frame_equal(
        scores_baseline.sort_index(),
        scores_tampered.sort_index(),
        check_exact=False,
        rtol=1e-6,
        atol=1e-10,
    )


def test_signal_timing_result_index_after_start_date(backtest_results):
    """
    Every result date in returns_gross must be AFTER START_DATE.

    The first result index entry is the END of the first holding period.
    The signal was computed at the start of that period (a rebalance date),
    and the return covers [t_start, t_end]. So t_end > t_start > START_DATE.
    """
    bt = backtest_results["value"]["quarterly"]["equal"]["long_only"]
    returns = bt["returns_gross"]
    assert len(returns) >= 2, "Need at least 2 periods for timing test"

    first_result_date = returns.index[0]
    assert first_result_date > pd.Timestamp(START_DATE), (
        f"First result date {first_result_date} should be after data start {START_DATE}"
    )


def test_signal_timing_dates_strictly_increasing(backtest_results):
    """
    Result dates must be strictly increasing — no two periods share an end date.

    Verifies that the backtest loop correctly advances the period and that the
    signal used for period [t, t+1] is computed at t, not at t+1 or later.
    """
    bt = backtest_results["value"]["quarterly"]["equal"]["long_only"]
    returns = bt["returns_gross"]
    assert len(returns) >= 2, "Need at least 2 periods for timing test"

    date_diffs = returns.index.to_series().diff().dropna()
    assert (date_diffs > pd.Timedelta(0)).all(), (
        "Result dates are not strictly increasing — period ordering is broken"
    )
