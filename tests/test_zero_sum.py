import numpy as np

from pairing_engine.zero_sum import reduce_dominance, solve_zero_sum_game


def test_reduce_dominance_removes_strictly_dominated_row():
    # Row 0 is strictly worse than row 1 for the maximizer in every column.
    mat = np.array([
        [1.0, 1.0],
        [5.0, 5.0],
    ])
    rows, cols = reduce_dominance(mat)
    assert rows == [1]


def test_reduce_dominance_removes_strictly_dominated_column():
    # Column 1 is strictly worse than column 0 for the minimizer in every row.
    mat = np.array([
        [1.0, 5.0],
        [1.0, 5.0],
    ])
    rows, cols = reduce_dominance(mat)
    assert cols == [0]


def test_solve_zero_sum_game_pure_saddle_point():
    # Row 1 / Col 0 is a pure saddle point: value 3.
    mat = np.array([
        [2.0, 4.0],
        [3.0, 5.0],
    ])
    solution = solve_zero_sum_game(mat)
    assert np.isclose(solution.value, 3.0)
    assert np.isclose(solution.row_strategy[1], 1.0)
    assert np.isclose(solution.col_strategy[0], 1.0)


def test_solve_zero_sum_game_matching_pennies():
    # Classic mixed-equilibrium game: both players play 50/50, value 0.
    mat = np.array([
        [1.0, -1.0],
        [-1.0, 1.0],
    ])
    solution = solve_zero_sum_game(mat)
    assert np.isclose(solution.value, 0.0, atol=1e-6)
    assert np.allclose(solution.row_strategy, [0.5, 0.5], atol=1e-6)
    assert np.allclose(solution.col_strategy, [0.5, 0.5], atol=1e-6)
