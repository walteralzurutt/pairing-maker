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


# ---------------------------------------------------------------------
# Dominance reduction: strictly/weakly dominated rows (row player, who
# maximizes) or columns (column player, who minimizes) can never carry
# positive equilibrium weight, so stripping them first shrinks the LP the
# recursive N-player engine ends up re-solving many times over for
# overlapping sub-games. Value-preserving -- purely a performance/
# robustness optimization, the returned equilibrium is still valid for
# the original, un-reduced matrix.
# ---------------------------------------------------------------------

def _remove_dominated_rows(mat: np.ndarray, row_idx: list) -> list:
    row_idx = list(row_idx)
    changed = True
    while changed and len(row_idx) > 1:
        changed = False
        for i in row_idx:
            for i2 in row_idx:
                if i == i2:
                    continue
                row_i = mat[i, :]
                row_i2 = mat[i2, :]
                # i2 dominates i if it's always >= (weakly) and somewhere > (strictly)
                if np.all(row_i2 >= row_i - 1e-9) and np.any(row_i2 > row_i + 1e-9):
                    row_idx.remove(i)
                    changed = True
                    break
            if changed:
                break
    return row_idx


def _remove_dominated_cols(mat: np.ndarray, col_idx: list) -> list:
    # Column domination (minimizer prefers smaller) on `mat` is exactly row
    # domination (maximizer prefers larger) on `-mat.T`: negating flips which
    # direction is "better", and transposing turns columns into rows.
    return _remove_dominated_rows(-mat.T, col_idx)


def reduce_dominance(payoff: np.ndarray) -> tuple[list, list]:
    """Iteratively strip dominated rows and columns, alternating, until
    neither changes. Returns the surviving ORIGINAL row/column indices.
    """
    rows = list(range(payoff.shape[0]))
    cols = list(range(payoff.shape[1]))
    prev = None
    while prev != (tuple(rows), tuple(cols)):
        prev = (tuple(rows), tuple(cols))
        sub = payoff[np.ix_(rows, cols)]
        new_rows_local = _remove_dominated_rows(sub, list(range(len(rows))))
        rows = [rows[i] for i in new_rows_local]
        sub = payoff[np.ix_(rows, cols)]
        new_cols_local = _remove_dominated_cols(sub, list(range(len(cols))))
        cols = [cols[j] for j in new_cols_local]
    return rows, cols


def solve_zero_sum_game(payoff: np.ndarray) -> ZeroSumSolution:
    """Solve an m x n zero-sum game given a payoff matrix from the row
    player's perspective (row player maximizes, column player minimizes):

        max_p min_j sum_i p_i * A[i, j]

    First strips dominated rows/columns (see reduce_dominance), then
    short-circuits on a 1x1 reduced game or a pure saddle point, falling
    back to the LP formulation of minimax (plus the dual LP for the
    column player's strategy) only when the reduced game truly has no
    pure equilibrium. Strategies are returned over the FULL original
    index space (zero weight on anything stripped as dominated).
    """
    payoff = np.asarray(payoff, dtype=float)
    m, n = payoff.shape
    rows, cols = reduce_dominance(payoff)
    sub = payoff[np.ix_(rows, cols)]

    row_strategy = np.zeros(m)
    col_strategy = np.zeros(n)

    if sub.shape == (1, 1):
        row_strategy[rows[0]] = 1.0
        col_strategy[cols[0]] = 1.0
        return ZeroSumSolution(value=float(sub[0, 0]), row_strategy=row_strategy, col_strategy=col_strategy)

    row_mins = sub.min(axis=1)
    best_row_local = int(np.argmax(row_mins))
    maximin = row_mins[best_row_local]

    col_maxs = sub.max(axis=0)
    best_col_local = int(np.argmin(col_maxs))
    minimax = col_maxs[best_col_local]

    if np.isclose(maximin, minimax, atol=1e-9):
        row_strategy[rows[best_row_local]] = 1.0
        col_strategy[cols[best_col_local]] = 1.0
        return ZeroSumSolution(value=float(maximin), row_strategy=row_strategy, col_strategy=col_strategy)

    r, c = sub.shape

    # --- Row player (maximizer) LP ---
    # variables: p_1..p_r, v   ; minimize -v
    c_lp = np.zeros(r + 1)
    c_lp[-1] = -1.0
    # constraints: v - sum_i p_i * A[i, j] <= 0   for each column j
    A_ub = np.hstack([-sub.T, np.ones((c, 1))])
    b_ub = np.zeros(c)
    A_eq = np.hstack([np.ones((1, r)), np.zeros((1, 1))])
    b_eq = np.array([1.0])
    bounds = [(0, None)] * r + [(None, None)]
    res_row = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                       bounds=bounds, method="highs")
    if not res_row.success:
        raise RuntimeError(f"Row-player LP failed: {res_row.message}")
    p = res_row.x[:r]
    value = res_row.x[-1]

    # --- Column player (minimizer) LP ---
    # variables: q_1..q_c, v ; minimize v
    c_lp2 = np.zeros(c + 1)
    c_lp2[-1] = 1.0
    # constraints: sum_j q_j * A[i, j] - v <= 0   for each row i
    A_ub2 = np.hstack([sub, -np.ones((r, 1))])
    b_ub2 = np.zeros(r)
    A_eq2 = np.hstack([np.ones((1, c)), np.zeros((1, 1))])
    b_eq2 = np.array([1.0])
    bounds2 = [(0, None)] * c + [(None, None)]
    res_col = linprog(c_lp2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                       bounds=bounds2, method="highs")
    if not res_col.success:
        raise RuntimeError(f"Column-player LP failed: {res_col.message}")
    q = res_col.x[:c]

    for i, ridx in enumerate(rows):
        row_strategy[ridx] = p[i]
    for j, cidx in enumerate(cols):
        col_strategy[cidx] = q[j]

    return ZeroSumSolution(value=value, row_strategy=row_strategy, col_strategy=col_strategy)
