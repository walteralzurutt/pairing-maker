"""
Generic finite zero-sum game solver, via the standard LP formulation of
matrix-game minimax. Used as a building block by every pairing format
(4-player, N-player, ...) -- nothing here is specific to how a format's
stages are structured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class ZeroSumSolution:
    value: float                 # game value from the row player's POV
    row_strategy: np.ndarray     # optimal mixed strategy for the row player
    col_strategy: np.ndarray     # optimal mixed strategy for the column player


def solve_zero_sum_game(payoff: np.ndarray) -> ZeroSumSolution:
    """Solve an m x n zero-sum game given a payoff matrix from the row
    player's perspective (row player maximizes, column player minimizes):

        max_p min_j sum_i p_i * A[i, j]

    solved as an LP, plus the dual LP for the column player's strategy.
    """
    m, n = payoff.shape

    # --- Row player (maximizer) LP ---
    # variables: p_1..p_m, v   ; minimize -v
    c = np.zeros(m + 1)
    c[-1] = -1.0
    # constraints: v - sum_i p_i * A[i, j] <= 0   for each column j
    A_ub = np.hstack([-payoff.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    A_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))])
    b_eq = np.array([1.0])
    bounds = [(0, None)] * m + [(None, None)]
    res_row = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                       bounds=bounds, method="highs")
    if not res_row.success:
        raise RuntimeError(f"Row-player LP failed: {res_row.message}")
    p = res_row.x[:m]
    value = res_row.x[-1]

    # --- Column player (minimizer) LP ---
    # variables: q_1..q_n, v ; minimize v
    c2 = np.zeros(n + 1)
    c2[-1] = 1.0
    # constraints: sum_j q_j * A[i, j] - v <= 0   for each row i
    A_ub2 = np.hstack([payoff, -np.ones((m, 1))])
    b_ub2 = np.zeros(m)
    A_eq2 = np.hstack([np.ones((1, n)), np.zeros((1, 1))])
    b_eq2 = np.array([1.0])
    bounds2 = [(0, None)] * n + [(None, None)]
    res_col = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                       bounds=bounds2, method="highs")
    if not res_col.success:
        raise RuntimeError(f"Column-player LP failed: {res_col.message}")
    q = res_col.x[:n]

    return ZeroSumSolution(value=value, row_strategy=p, col_strategy=q)
