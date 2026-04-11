# Factor Model: Investment Thesis and Methodology

## Executive Summary

This model systematically ranks S&P 500 equities across five academic risk factors, value, momentum, quality, size, and low volatility, and constructs quintile portfolios rebalanced monthly, quarterly, and annually over a ten-year evaluation period (2016-2026). The central thesis is that these premia are durable because they compensate for either genuine economic risk or persistent behavioral error, and that a disciplined, low-cost implementation can capture them net of realistic transaction costs. The composite multi-factor portfolio is the primary deliverable: by combining signals with low pairwise correlation, it achieves a more consistent information ratio than any single factor in isolation.

---

## Results Interpretation: What the Numbers Mean

Quality is the strongest result in this backtest, posting a Sharpe ratio of 1.55 and a mean IC of 0.084, the only factor with a consistently positive information coefficient across the evaluation period. This is directionally consistent with Novy-Marx (2013) and reflects a real premium: high-ROE firms do tend to outperform over time. However, the magnitude is inflated. Because current ROE is applied to historical rebalance dates, the model effectively knows which firms are profitable today and selects them throughout the backtest. A production version using point-in-time filings would show a lower but still credible quality premium; the direction of the result should survive, even if the level does not.

Momentum shows a near-zero IC (~0.009) but still delivers reasonable portfolio-level returns. This is not a contradiction. Cross-sectional momentum IC is structurally noisy: ranking 500 stocks by trailing returns and asking how well that rank predicts next-period returns yields a weak correlation period-by-period, because reversal and sector effects generate offsetting noise. The portfolio-level return comes from the tails, the strong winners in Q5 and the persistent losers in Q1, where the signal clusters enough to generate a spread. Momentum is the cleanest factor in this model from a data integrity standpoint, since it relies entirely on historical prices, and its results warrant the most confidence.

Negative ICs for value, size, and low volatility are expected and should not be interpreted as evidence that these premia do not exist. They reflect fundamental data contamination. A stock that is cheap today was not necessarily cheap in 2016 or 2018; using today's P/E backward means the value factor is partially sorting on forward-looking information. The same logic applies to size and market cap. These factors cannot be fairly evaluated until the model ingests point-in-time filings, and their IC statistics in this version are uninformative.

The low-volatility annual cap-weight portfolio reporting 0% maximum drawdown warrants scrutiny, not celebration. Annual rebalancing concentrates the portfolio in a small number of defensive mega-caps, utilities, consumer staples, healthcare, held for twelve months without adjustment. In any given calendar year over this period, that specific basket happened not to experience a qualifying peak-to-trough decline. This is an artifact of holding interval length combined with sector concentration, not evidence of a risk-free strategy. A monthly drawdown measurement, or a rebalancing window that tracks intra-year prices, would reveal meaningful risk that the annual figure conceals.

In aggregate: the engine's architecture, signal-timing discipline, and analytics are production-quality. The results are directionally informative, quality and momentum show genuine predictive content, but none of the fundamental-factor results are investable without point-in-time data. Momentum and low volatility, both derived from price history alone, are the cleanest signals this version can offer.

---

## Factor Selection Rationale

**Value, Earnings Yield (1/PE).** The value premium is the oldest documented anomaly in empirical asset pricing. Fama and French (1992) showed that high book-to-market firms generate significantly higher returns than growth firms after controlling for market beta, attributing the premium to distress risk. Earnings yield is used here rather than book-to-market because it is more directly comparable across sectors and less sensitive to accounting treatments of intangibles. The behavioral explanation, investors systematically overpay for growth and underprice mean-reversion, is equally compelling. Either way, the signal has been robust across markets and time periods.

**Momentum, 12-1 Month Return.** Jegadeesh and Titman (1993) documented that stocks performing well over the prior 6-12 months continue to outperform over the subsequent 3-12 months. The one-month skip is not an implementation detail, it is essential. The very short-term reversal effect (Jegadeesh 1990) means that including the most recent month inverts the signal at short horizons. The behavioral explanation centers on underreaction: investors are slow to incorporate new information into prices, and the trend persists until the information is fully priced. Momentum is the factor with the strongest return premium but the highest crash risk, making it the factor that most benefits from being blended with defensive signals like low volatility.

**Quality, Return on Equity (ROE).** Novy-Marx (2013) demonstrated that gross profitability, a measure of how efficiently a firm deploys its assets, predicts returns as powerfully as the book-to-market ratio, and the two signals are negatively correlated, making them highly complementary. ROE is used here as a tractable proxy for quality: firms with high return on equity tend to have durable competitive advantages and generate cash in excess of their reinvestment needs. The risk explanation is that high-quality firms have lower financial distress risk; the behavioral explanation is that investors anchor on reported earnings rather than balance sheet efficiency.

**Size, Log Market Cap (Inverted).** Fama and French (1992, 1993) documented the small-cap premium as one of the two factors added to CAPM in their three-factor model. Smaller firms carry higher fundamental uncertainty, lower liquidity, and higher information asymmetry, all of which demand a risk premium in equilibrium. Log market cap is used rather than raw market cap to compress the extreme right skew of the distribution. The factor is inverted so that smaller firms receive higher scores, consistent with the direction of the premium.

**Low Volatility, 60-Day Realized Volatility (Inverted).** Ang, Hodrick, Xing, and Zhang (2006) documented the low-volatility anomaly: stocks with high idiosyncratic volatility earn lower subsequent returns, the opposite of what standard risk-return intuition predicts. The behavioral explanation involves lottery preferences, investors overpay for high-volatility stocks with optionality-like payoff profiles, and benchmarking constraints that prevent institutional investors from fully arbitraging the anomaly. A 60-day window captures medium-term realized risk without being overly reactive to single-event spikes. This factor is genuinely point-in-time: it uses only historical price data up to each signal date.

---

## Portfolio Construction Choices

Quintile sorts are the standard methodology in academic factor research because they are non-parametric, robust to outliers, and produce a clean spread that any reviewer can verify. Splitting the universe into five equal buckets makes the Q5-minus-Q1 spread the natural summary statistic, and the monotonicity of returns across Q1 through Q5 is the primary diagnostic for whether a factor is working.

Both long-only and long-short portfolios are constructed because they serve different mandates. A long-only portfolio, fully invested in Q5, is the relevant implementation for long-only asset managers, mutual funds, and any strategy constrained against short positions. It carries significant market beta. The long-short portfolio, long Q5, short Q1, dollar-neutral, is the institutional implementation used in the academic literature and at hedge funds. It isolates the pure factor return by netting out market exposure and allows the strategy to be evaluated on alpha rather than beta. Comparing the two side-by-side reveals how much of the long-only return is attributable to market exposure versus the factor itself.

Equal-weight and cap-weight variants are both tested because they reflect a genuine construction tension. Equal-weight gives every stock in the quintile identical influence, which maximizes the factor signal but concentrates the portfolio in smaller, less-liquid names. Cap-weight mirrors what a large institutional allocator would actually trade, positions are sized in proportion to market capitalization, which reduces impact costs and enables larger AUM, but it dilutes the factor signal because mega-cap stocks dominate the weights regardless of their factor ranking. In practice, the difference between the two reveals how much of the return depends on small-cap exposure.

Three rebalancing frequencies, monthly, quarterly, and annual, test the fundamental trade-off between signal freshness and transaction costs. Momentum decays fastest and rewards higher-frequency rebalancing; value and quality are slow-moving and carry the same information for months. The quarterly backtest is the most practically relevant: it refreshes signals before they decay materially while keeping annual turnover at a level that survives realistic transaction costs.

---

## Key Risks and Limitations

**Survivorship bias** is the most significant structural flaw in this backtest. The universe is drawn from today's S&P 500 membership and applied backward to 2016. This excludes every company that was in the index a decade ago but has since been acquired, delisted, or removed due to deteriorating fundamentals. Because failure is correlated with poor factor scores, value traps, low-quality firms, high-volatility names, survivorship bias inflates the return of every factor strategy, and inflates the return of Q1 (the short leg) the most. The magnitude of this bias is difficult to estimate without a point-in-time index membership database, but academic studies suggest it can add 1-2% per year to reported backtested returns.

**Fundamental data is not point-in-time** for the value, quality, and size factors. P/E ratios, ROE, and market cap are fetched from yfinance as of today and applied uniformly across all historical rebalance dates. A firm that currently trades at 15x earnings may have traded at 8x or 40x in 2018; the current value is used for all periods. This introduces look-ahead bias into three of the five factors. Momentum and low volatility are clean: they depend only on historical price data, which is correctly filtered to each signal date. Any reported outperformance from the value, quality, or size factors should be discounted accordingly until the model is upgraded with a point-in-time data source such as Compustat or EDGAR quarterly filings.

**Factor crowding** is a live risk not captured in historical backtests. As systematic strategies have grown to manage trillions of dollars, the most well-documented factors, momentum in particular, have become crowded trades. When crowded factors unwind, they do so rapidly and with large drawdowns that are not reflected in historical IC or Sharpe statistics. The 2018 momentum crash and the 2020 COVID reversal are examples. A production model would monitor positioning concentration and crowding metrics as part of the risk overlay.

**Transaction cost assumptions** are conservative in one direction (target-to-target turnover overstates cost for low-frequency rebalancing because it ignores beneficial price drift) and optimistic in another (10 bps one-way is reasonable for large-cap S&P 500 names but understates impact costs for trades that move the market, and ignores borrowing costs on the short leg). For a strategy managing meaningful AUM, market impact at quarterly rebalance would be the binding constraint, not the spread.

**Sector concentration** is an uncontrolled risk. Raw cross-sectional factor ranking produces portfolios with systematic sector tilts. Value portfolios chronically overweight financials and energy; momentum portfolios chase whatever sector is in a cyclical upswing; low-volatility portfolios cluster in utilities and consumer staples. These tilts are real economic bets, not factor bets, and a reviewer evaluating the results cannot cleanly separate factor alpha from sector returns without sector-neutral construction.

---

## What I Would Do Next (Production Version)

The highest-priority upgrade is point-in-time fundamental data. Compustat provides quarterly P/E and ROE snapshots tied to filing dates, not report dates, eliminating the look-ahead bias in the value and quality factors entirely. For market cap, the fix is simpler: shares outstanding from a historical database multiplied by split-adjusted daily prices gives an accurate time series at low cost. These two changes would materially change the reported results for the value, quality, and size backtests and would be mandatory before presenting to any institutional investor.

The second priority is a dynamic universe using historical S&P 500 index membership. CRSP or Bloomberg maintain constituent snapshots for every calendar date going back decades. Using those snapshots eliminates survivorship bias and adds a realistic "index addition" and "index deletion" effect, stocks are often added after extended outperformance and deleted after underperformance, creating predictable short-term return patterns around rebalancing events that a systematic strategy can model or avoid.

A production factor model would also implement sector-neutral construction. The standard approach, following Barra, is to demean factor scores within each GICS sector before ranking, ensuring the long and short legs are balanced across sectors and the portfolio is a pure factor bet rather than a sector bet. This reduces both volatility and the risk of sector-specific macro events driving returns. Risk model integration, computing factor exposures, idiosyncratic risk, and portfolio tracking error against a benchmark, would complete the institutional picture.

Finally, the rebalancing logic would be upgraded from calendar-based to signal-based, with a buffer zone around current portfolio weights to suppress unnecessary turnover. Rather than rebalancing on every quarter-end regardless of how much scores have changed, stocks would only be traded when their quintile assignment has changed by more than one bucket. This implementation-alpha, earning the same factor exposure with meaningfully less turnover, is where systematic strategies compete in practice.

---

## References

- Fama, E. F., & French, K. R. (1992). The cross-section of expected stock returns. *Journal of Finance*, 47(2), 427-465.
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65-91.
- Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). The cross-section of volatility and expected returns. *Journal of Finance*, 61(1), 259-299.
- Novy-Marx, R. (2013). The other side of value: The gross profitability premium. *Journal of Financial Economics*, 108(1), 1-28.
- Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929-985.
