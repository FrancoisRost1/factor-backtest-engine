"""
Tests for factor_engine/transaction_costs.py.

All expected values are hand-calculated from first principles.
"""

import numpy as np
import pandas as pd
import pytest

from factor_engine.transaction_costs import apply_transaction_costs, compute_turnover


# ---------------------------------------------------------------------------
# compute_turnover
# ---------------------------------------------------------------------------

class TestComputeTurnover:
    def test_identical_portfolios_zero_turnover(self):
        prev = pd.Series({"A": 0.5, "B": 0.5})
        curr = pd.Series({"A": 0.5, "B": 0.5})
        assert compute_turnover(prev, curr) == 0.0

    def test_completely_different_portfolios_turnover_2(self):
        """
        L/O prev={A:0.5, B:0.5}, curr={C:0.5, D:0.5}.
        Changes: A:-0.5, B:-0.5, C:+0.5, D:+0.5. Sum = 2.0.
        """
        prev = pd.Series({"A": 0.5, "B": 0.5})
        curr = pd.Series({"C": 0.5, "D": 0.5})
        assert abs(compute_turnover(prev, curr) - 2.0) < 1e-10

    def test_one_stock_swapped_turnover_1(self):
        """
        prev={A:0.5, B:0.5}, curr={A:0.5, C:0.5}.
        Changes: A:0, B:-0.5, C:+0.5. Sum = 1.0.
        """
        prev = pd.Series({"A": 0.5, "B": 0.5})
        curr = pd.Series({"A": 0.5, "C": 0.5})
        assert abs(compute_turnover(prev, curr) - 1.0) < 1e-10

    def test_rebalance_same_stocks_different_weights(self):
        """
        prev={A:0.5, B:0.5}, curr={A:0.6, B:0.4}.
        Changes: A:+0.1, B:-0.1. Sum = 0.2.
        """
        prev = pd.Series({"A": 0.5, "B": 0.5})
        curr = pd.Series({"A": 0.6, "B": 0.4})
        assert abs(compute_turnover(prev, curr) - 0.2) < 1e-10

    def test_first_rebalance_from_empty_portfolio(self):
        """
        First period: prev is empty, curr={A:0.6, B:0.4}.
        Turnover = |0.6 - 0| + |0.4 - 0| = 1.0.
        """
        prev = pd.Series(dtype=float)
        curr = pd.Series({"A": 0.6, "B": 0.4})
        assert abs(compute_turnover(prev, curr) - 1.0) < 1e-10

    def test_long_short_full_replacement_turnover_2(self):
        """
        L/S prev={A:+0.5, B:-0.5}, curr={C:+0.5, D:-0.5}.
        Changes: A:-0.5, B:+0.5, C:+0.5, D:-0.5. Sum = 2.0.
        """
        prev = pd.Series({"A": 0.5, "B": -0.5})
        curr = pd.Series({"C": 0.5, "D": -0.5})
        assert abs(compute_turnover(prev, curr) - 2.0) < 1e-10

    def test_turnover_is_always_nonnegative(self):
        prev = pd.Series({"A": 0.3, "B": 0.7})
        curr = pd.Series({"A": 0.7, "B": 0.3})
        assert compute_turnover(prev, curr) >= 0.0

    def test_three_stock_partial_overlap(self):
        """
        prev={A:0.4, B:0.4, C:0.2}, curr={A:0.5, B:0.3, D:0.2}.
        Changes: A:+0.1, B:-0.1, C:-0.2, D:+0.2. Sum = 0.6.
        """
        prev = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
        curr = pd.Series({"A": 0.5, "B": 0.3, "D": 0.2})
        assert abs(compute_turnover(prev, curr) - 0.6) < 1e-10


# ---------------------------------------------------------------------------
# apply_transaction_costs
# ---------------------------------------------------------------------------

class TestApplyTransactionCosts:
    def test_net_equals_gross_minus_turnover_times_cost(self):
        """net = gross - turnover × cost_per_unit."""
        gross = 0.05
        turnover = 1.0
        cost = 0.001
        expected_net = gross - turnover * cost   # 0.05 - 0.001 = 0.049
        assert abs(apply_transaction_costs(gross, turnover, cost) - expected_net) < 1e-10

    def test_zero_turnover_no_cost_impact(self):
        gross = 0.05
        net = apply_transaction_costs(gross, turnover=0.0, cost_per_unit=0.001)
        assert abs(net - gross) < 1e-10

    def test_full_replacement_10bps_gives_20bps_drag(self):
        """
        Turnover=2.0 (full L/O replacement) at 10bps → drag = 20bps = 0.002.
        gross=0.01, net=0.01-0.002=0.008.
        """
        gross = 0.01
        net = apply_transaction_costs(gross, turnover=2.0, cost_per_unit=0.001)
        assert abs(net - 0.008) < 1e-10

    def test_high_turnover_can_make_return_negative(self):
        gross = 0.001   # 10bps gross
        net = apply_transaction_costs(gross, turnover=2.0, cost_per_unit=0.001)
        assert net < 0   # drag (20bps) > gross (10bps)

    def test_cost_scales_linearly_with_turnover(self):
        gross = 0.05
        cost = 0.002
        net_t1 = apply_transaction_costs(gross, turnover=1.0, cost_per_unit=cost)
        net_t2 = apply_transaction_costs(gross, turnover=2.0, cost_per_unit=cost)
        # Double the turnover → double the drag
        drag_t1 = gross - net_t1
        drag_t2 = gross - net_t2
        assert abs(drag_t2 - 2.0 * drag_t1) < 1e-10

    def test_exact_values_multiple_scenarios(self):
        cases = [
            # (gross, turnover, cost, expected_net)
            (0.10, 0.5, 0.001, 0.10 - 0.0005),
            (0.02, 1.0, 0.002, 0.02 - 0.002),
            (0.00, 1.5, 0.001, 0.00 - 0.0015),
        ]
        for gross, to, cost, expected in cases:
            result = apply_transaction_costs(gross, to, cost)
            assert abs(result - expected) < 1e-10, (
                f"gross={gross}, turnover={to}, cost={cost}: "
                f"expected {expected}, got {result}"
            )
