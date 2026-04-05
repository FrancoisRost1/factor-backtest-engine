"""
Transaction cost modelling: turnover computation and cost application.

Turnover is the standard measure of how much a portfolio changes between
rebalances. It equals the sum of absolute weight changes across all positions.

Two canonical cases:
  No rebalancing (identical portfolios): turnover = 0
  Full replacement (new portfolio, no overlap): turnover = 2.0 for long-only
    (sum of prev weights out + sum of curr weights in = 1 + 1 = 2)

Net return after costs = gross return − turnover × one-way cost per unit.
The cost is applied once per unit of turnover, which already accounts for
the round-trip nature: selling A and buying B each cost 1× the one-way rate.

*** TARGET-TO-TARGET TURNOVER (simplifying assumption) ***
This implementation compares target weights at consecutive rebalance dates
(prev_target → curr_target).  It does NOT account for portfolio drift between
rebalances: between t and t+1, weights shift as prices move, so the actual
cost at t+1 is incurred from the DRIFTED weights, not the prior target weights.
Drift-adjusted turnover (also called 'actual turnover') would first compute
where weights drifted to by t+1, then compare to the new target.  Target-to-
target overstates turnover for factors with low rebalancing frequency because
price drift partially accomplishes the rebalance 'for free'.  This is documented
in CLAUDE.md under 'Simplifying assumptions'.
"""

import pandas as pd


def compute_turnover(
    prev_weights: pd.Series,
    curr_weights: pd.Series,
) -> float:
    """
    Compute portfolio turnover as the sum of absolute weight changes.

    Turnover = Σ |curr_weight[i] - prev_weight[i]| over all tickers in
    the union of prev and curr portfolios. Tickers that exit are treated
    as going to weight 0; tickers that enter are treated as coming from 0.

    Reference values:
      Identical portfolios          → 0.0
      Complete replacement (L/O)    → 2.0  (out 1.0, in 1.0)
      One stock swapped (equal wt)  → 2 × weight of that stock
      First rebalance (empty prev)  → sum(|curr_weights|)

    Parameters
    ----------
    prev_weights : pd.Series
        Portfolio weights before rebalancing. Empty Series = first rebalance.
    curr_weights : pd.Series
        Portfolio weights after rebalancing.

    Returns
    -------
    float
        Non-negative turnover. 0.0 = no change. 2.0 = full long-only replacement.
    """
    all_tickers = prev_weights.index.union(curr_weights.index)
    prev_aligned = prev_weights.reindex(all_tickers, fill_value=0.0)
    curr_aligned = curr_weights.reindex(all_tickers, fill_value=0.0)
    return float((curr_aligned - prev_aligned).abs().sum())


def apply_transaction_costs(
    gross_return: float,
    turnover: float,
    cost_per_unit: float = 0.001,
) -> float:
    """
    Deduct transaction costs from a gross period return.

    Net return = gross return − turnover × cost_per_unit

    Cost is expressed as a fraction (e.g., 0.001 = 10 basis points one-way).
    With turnover = 2.0 (full replacement) and cost = 0.001 (10bps), the drag
    is 0.002 = 20bps per period, which is the round-trip cost.

    Parameters
    ----------
    gross_return : float
        Period return before transaction costs.
    turnover : float
        Portfolio turnover for the period (from compute_turnover).
    cost_per_unit : float
        One-way transaction cost as a fraction (default 0.001 = 10bps).

    Returns
    -------
    float
        Net return after deducting transaction costs. Can be negative.
    """
    return gross_return - turnover * cost_per_unit
