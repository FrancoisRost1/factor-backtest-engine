# CLAUDE.md — Factor Model + Backtesting Engine

> Auto-loaded by Claude Code. Read this fully before doing anything.

---

## What this project is

A multi-factor investment model with a rigorous backtesting engine.
Ranks S&P 500 stocks by 5 academic factors, constructs quintile portfolios,
and measures performance with institutional-grade analytics.

Built by a CFA student as a GitHub portfolio project.
Must signal: real factor investing knowledge, clean engineering, no shortcuts.

---

## Factor definitions (STRICT — one metric per factor)

| Factor | Metric | Direction | Source |
|--------|--------|-----------|--------|
| Value | Earnings Yield (1 / PE ratio) | Higher = better | yfinance `trailingPE` |
| Momentum | 12-month return minus last 1 month (12-1) | Higher = better | yfinance price history |
| Quality | Return on Equity (ROE) | Higher = better | yfinance `returnOnEquity` |
| Size | log(market cap), inverted | Smaller = higher score | yfinance `marketCap` |
| Low Volatility | 60-day rolling stdev of daily returns, inverted | Lower vol = higher score | yfinance price history |

**No multi-metric blends. No composite sub-factors. One metric = one factor.**

---

## Normalization (MANDATORY)

At each rebalance date, all factor values are **percentile-ranked cross-sectionally**.
- Rank range: 0.0 to 1.0
- Consistent across all 5 factors
- Handles NaN: stocks with missing factor values are excluded from that factor's ranking

---

## Composite factor

- Composite score = simple average of 5 percentile-ranked factor scores
- Equal weight across all factors (no custom weights for now)
- Stocks must have at least 3 valid factor scores to receive a composite score

---

## Signal timing (CRITICAL — no lookahead bias)

At time t:
1. Compute factor scores using ONLY data available at time t
2. Construct portfolio at time t
3. Measure returns from t → t+1

This must be enforced and verified by tests.

---

## Portfolio construction

### Quintile sorts
- At each rebalance: rank stocks by factor score, split into 5 equal buckets (Q1–Q5)
- Q5 = highest factor score, Q1 = lowest

### Portfolio types
- **Long-only**: hold Q5 stocks only, weights sum to 1.0
- **Long/short**: long Q5, short Q1, weights sum to 0.0

### Weighting schemes (both tested)
- **Equal weight**: uniform allocation within quintile
- **Cap weight**: proportional to market cap at rebalance date

---

## Rebalancing

Three frequencies tested and compared:
- Monthly
- Quarterly
- Annual

At each rebalance date:
- Recompute all factor scores
- Re-rank and re-sort into quintiles
- Reconstruct portfolios
- Compute turnover vs previous portfolio

---

## Transaction costs

- Cost = turnover × 10 bps (0.0010)
- Turnover = sum of absolute weight changes at rebalance
- Applied only at rebalance dates
- Both gross and net returns reported side by side

---

## Benchmark

- SPY (S&P 500 ETF)
- Used in equity curve comparison and regression

---

## Analytics (full institutional)

| Metric | Definition |
|--------|------------|
| Annualized Return | CAGR over evaluation period |
| Sharpe Ratio | (annualized return - risk-free) / annualized vol |
| Sortino Ratio | (annualized return - risk-free) / downside vol |
| Max Drawdown | worst peak-to-trough decline |
| Calmar Ratio | annualized return / abs(max drawdown) |
| Turnover | average sum of abs weight changes per rebalance |
| Hit Rate | % of periods with positive excess return vs benchmark |
| Information Coefficient | cross-sectional corr(factor_scores_t, returns_t→t+1) |
| Alpha | intercept from regression of strategy returns vs SPY |
| Beta | slope from regression of strategy returns vs SPY |

---

## Universe

- Current S&P 500 constituents (~500 stocks)
- **Survivorship bias acknowledged**: using today's index membership backtested over 10 years
- Documented in README and analysis.md

---

## Backtest period

- Raw data: ~2014–2026 (extra years for lookback windows)
- Evaluation period: ~2016–2026 (10 years)
- Earlier data used only for feature construction (momentum lookback, volatility window)

---

## Data source

- yfinance only (no API key needed)
- Price history: daily adjusted close
- Fundamentals: trailing PE, ROE, market cap
- Cached locally after first fetch to avoid repeated API calls

---

## Coding rules (same as all projects)

- No financial logic in `main.py` — it orchestrates only
- One file = one responsibility
- All thresholds and parameters in `config.yaml`
- Docstring on every class and method — explain financial rationale
- Handle edge cases: missing data → NaN, division by zero → NaN
- No file longer than ~150 lines
- All simplifying assumptions documented in inline comments

---

## Repo structure

```
factor-backtest-engine/
├── CLAUDE.md
├── README.md
├── config.yaml
├── requirements.txt
├── main.py                          # Orchestrator only
│
├── factor_engine/
│   ├── __init__.py
│   ├── universe.py                  # S&P 500 ticker list + SPY benchmark
│   ├── data_loader.py               # yfinance fetch: prices + fundamentals
│   ├── cache.py                     # Local caching to avoid re-fetching
│   ├── factors.py                   # 5 factor computations (one function each)
│   ├── normalize.py                 # Cross-sectional percentile ranking
│   ├── composite.py                 # Multi-factor composite score
│   ├── portfolio.py                 # Quintile sort + portfolio construction
│   ├── backtest.py                  # Backtest loop: signal → hold → return
│   ├── transaction_costs.py         # Turnover + cost calculation
│   ├── analytics.py                 # Sharpe, Sortino, drawdown, Calmar, hit rate
│   ├── ic.py                        # Information Coefficient computation
│   ├── regression.py                # Alpha/beta regression vs SPY
│   └── utils.py                     # Shared helpers (safe division, date utils)
│
├── app/
│   └── streamlit_app.py             # Dashboard
│
├── tests/
│   ├── __init__.py
│   ├── test_factors.py
│   ├── test_normalize.py
│   ├── test_portfolio.py
│   ├── test_backtest.py
│   └── test_analytics.py
│
├── data/
│   ├── raw/                         # Raw yfinance downloads
│   ├── processed/                   # Factor scores, portfolios, returns
│   └── cache/                       # Cached API responses
│
├── docs/
│   └── analysis.md                  # Investment thesis + methodology
│
└── outputs/
    └── backtest_results.csv
```

---

## Simplifying assumptions (all documented)

- Survivorship bias: current S&P 500 constituents used for full backtest period
- No market impact modeling beyond flat 10bps transaction cost
- Earnings yield uses trailing PE (not forward estimates)
- ROE is trailing (not forecasted)
- No sector neutralization (raw cross-sectional ranking)
- Risk-free rate assumed 0% for Sharpe/Sortino (or use 3-month T-bill if easily available)
- No shorting costs modeled for long/short portfolios
- Rebalance assumed at close prices (no intraday friction)
- **NOT point-in-time for fundamental factors (value, quality, size)**: P/E ratio, ROE, and
  market cap are fetched TODAY from yfinance and applied uniformly across the entire
  backtest period back to 2014. This introduces look-ahead bias into those three factors.
  Momentum (12-1) and Low Volatility (60-day vol) are fully historical — they use only
  price data up to each rebalance date, so they are genuinely point-in-time.
  A production system would use SEC quarterly filings (EDGAR) for P/E and ROE, and
  price × historical shares outstanding for market cap.
- **Target-to-target turnover**: transaction costs are computed by comparing target weights
  at consecutive rebalance dates. Drift between rebalances (prices moving weights away
  from targets) is not accounted for. This overstates turnover for low-frequency strategies.

---

## Project status

✅ COMPLETE — pushed to GitHub 2026-04-05; retroactive audit 2026-04-09/10

### Key results
- 503 tickers, 10-year backtest (2016-2026), 72 combinations
- Quality (ROE) is the dominant signal: Sharpe 1.55, IC 0.084
- Momentum: weak IC but decent portfolio returns
- Value/Size/Low-Vol IC negative due to point-in-time contamination
- 170 tests passing, Codex-audited twice (initial + retroactive follow-up)
- Streamlit dashboard: 4 tabs, dark mode

### Retroactive audit (2026-04-09 / 2026-04-10)

A second-pass review against the mature project standards (CLAUDE.md
"config is the source of truth" rule) found that several config keys were
advertised but never read by the code, plus one half-wired param. Two
commits fixed all of them:

**Commit `dde4a2d` — wire dead config keys**
- `risk_free_rate` (`config.yaml:analytics`) is now read by `main.py` and
  passed into `compute_all_metrics` instead of the hardcoded `0.0` default.
- `annualization_factor` (`config.yaml:analytics`) is now threaded through
  `backtest.py → compute_all_factors → compute_rolling_volatility` instead
  of the hardcoded `np.sqrt(252)` in `factors_price.py`.
- `min_valid_factors` (`config.yaml:composite`) is now threaded through
  `backtest.py → compute_all_factors` instead of the hardcoded module-level
  `_COMPOSITE_MIN_VALID = 3` in `factors.py`.
- `n_quantiles` (`config.yaml:portfolio`) is now passed into
  `compute_quintile_returns` instead of the hardcoded `range(1, 6)`.
- Removed dead keys: `data.price_column` (irrelevant under
  `auto_adjust=True`) and `transaction_costs.cost_per_trade_bps`
  (duplicate of `cost_per_trade`).

**Commit `37869a2` — Codex follow-up: half-wired n_quantiles**

Codex (`gpt-5.4`, HIGH reasoning, read-only) caught that `dde4a2d` only
half-wired `n_quantiles`: the backtest read it from config, but
`construct_long_only` and `construct_long_short` still hardcoded the
labels `5` and `1`. With `n_quantiles=10`, the long-only portfolio would
silently target the *middle* decile instead of the top, and long/short
would target the wrong extremes — non-empty but factor-wrong.

Fix: added `long_quintile` / `short_quintile` parameters to both
construction functions, threaded them through `backtest.py` from
`config['portfolio']['long_quintile'/'short_quintile']`. Added 4 new unit
tests (`TestConfigurableNQuantiles` in `tests/test_portfolio.py`) covering
decile sorts (`n_quantiles=10`), including a regression test that pins
the wrong-bucket bug to prevent recurrence.

### GitHub
github.com/FrancoisRost1/factor-backtest-engine

---

*Last updated: 2026-04-10*
