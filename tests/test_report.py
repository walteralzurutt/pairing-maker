import numpy as np

from pairing_engine.report import rank_options, tie_break_note, format_option


def test_tie_break_false_sorts_by_equilibrium_weight():
    """Regression test for the bug already fixed: when predicting an
    opponent, two options can tie exactly on vs_equilibrium_opponent while
    the solved equilibrium puts all weight on only one of them. With
    tie_break=False, the top row must be the one actually holding weight,
    not whichever the input order happens to favor.
    """
    # Two opponent shields ("Marines", "Necron") tie exactly on value (34.0),
    # but the equilibrium puts 100% weight on Necron (the second option).
    payoff_for_me = np.array([
        [34.0],
        [34.0],
        [27.25],
        [27.0],
    ])
    opp_equilibrium_strategy = np.array([1.0])
    my_equilibrium_strategy = np.array([0.0, 1.0, 0.0, 0.0])
    option_labels = ["Marines", "Necron", "CSM", "IK"]

    report = rank_options(payoff_for_me, opp_equilibrium_strategy, my_equilibrium_strategy,
                           option_labels, tie_break=False)

    assert report.iloc[0]["option"] == "Necron"
    assert report.iloc[0]["equilibrium_weight"] > 0


def test_tie_break_true_prefers_best_value_when_spread_is_uniform():
    """When conditioning on a known opponent shield, the payoff collapses to
    a single column, so worst_case == best_case == value for every row and
    `spread` carries no signal. Even so, rank_options must still put the
    actual best/positive-weight option at row 0, not whichever option
    happens to be listed first.
    """
    option_labels = ["C_player", "B_player", "A_player"]
    payoff_for_me = np.array([
        [9.9],
        [10.3],
        [10.5],  # A_player is the true best -- and NOT first in the list
    ])
    opp_equilibrium_strategy = np.array([1.0])
    my_equilibrium_strategy = np.array([0.0, 0.0, 1.0])  # one-hot on the true best

    report = rank_options(payoff_for_me, opp_equilibrium_strategy, my_equilibrium_strategy,
                           option_labels, tie_break=True)

    assert report.iloc[0]["option"] == "A_player"
    assert report.iloc[0]["equilibrium_weight"] > 0


def test_format_option():
    assert format_option("Astra") == "Astra"
    assert format_option(("A1", "A2")) == "A1 y A2"


def test_tie_break_note_flags_when_recommendation_differs_from_equilibrium_weight():
    """When the near-tie/lower-spread override (rank_options) promotes a
    0%-weight option to row 0 over the option that actually holds the
    equilibrium weight, tie_break_note must surface that discrepancy."""
    option_labels = ["Ultramarines", "Death Guard"]
    payoff_for_me = np.array([
        [10.0, 10.0],  # Ultramarines: low spread
        [10.5, 9.0],   # Death Guard: higher average, but wider spread
    ])
    opp_equilibrium_strategy = np.array([0.5, 0.5])
    my_equilibrium_strategy = np.array([0.0, 1.0])  # equilibrium is all on Death Guard

    report = rank_options(payoff_for_me, opp_equilibrium_strategy, my_equilibrium_strategy,
                           option_labels, tie_break=True)

    assert report.iloc[0]["option"] == "Ultramarines"
    assert report.iloc[0]["equilibrium_weight"] == 0.0

    note = tie_break_note(report)
    assert note == ("El peso de equilibrio favorece a Death Guard, pero Ultramarines tiene "
                     "menor volatilidad.")


def test_tie_break_note_is_none_when_recommendation_already_holds_the_weight():
    option_labels = ["A_player", "B_player"]
    payoff_for_me = np.array([
        [10.5],
        [9.0],
    ])
    opp_equilibrium_strategy = np.array([1.0])
    my_equilibrium_strategy = np.array([1.0, 0.0])

    report = rank_options(payoff_for_me, opp_equilibrium_strategy, my_equilibrium_strategy,
                           option_labels, tie_break=True)

    assert tie_break_note(report) is None
