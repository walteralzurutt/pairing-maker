import math

import numpy as np
import pandas as pd
import pytest

from pairing_engine.formats import n_player


# ---------------------------------------------------------------------
# Phase 2 -- known_opponent_shield
# ---------------------------------------------------------------------

def test_recommend_shield_conditions_opp_report_too(sample_model):
    """Declaring the opponent's shield as certain must be reflected in the
    'Rival (predicción)' report too, not just our own recommendation --
    otherwise the UI can show a different shield as 'probable' than the one
    the user just told the tool they already know for certain.
    """
    session = n_player.LivePairingSession(sample_model, my_team="team1")
    known = "IK"  # currently 0% equilibrium weight in the unconditioned prediction
    assert known in session.opp_players

    my_report, opp_report = session.recommend_shield(known_opponent_shield=known)

    assert opp_report.iloc[0]["option"] == known
    assert opp_report.iloc[0]["equilibrium_weight"] == 1.0


# ---------------------------------------------------------------------
# Explainability: naive-baseline comparisons at every stage
# ---------------------------------------------------------------------

def test_naive_shield_prefers_best_average_escudo(sample_model):
    naive = n_player._naive_shield(sample_model.escudo_df, sample_model.team1_players)
    assert naive == "Marines"  # highest mean escudo score on the fixture data


def test_naive_swords_prefers_best_average_descarte(sample_model):
    remaining = [p for p in sample_model.team1_players if p != "Astra"]
    naive = n_player._naive_swords(sample_model.descarte_df, remaining)
    assert set(naive) == {"Marines", "Necron"}  # 2 highest mean descarte scores among the remaining 3


def test_recommend_shield_sets_shield_explanation(sample_model):
    session = n_player.LivePairingSession(sample_model, my_team="team1")
    session.recommend_shield()
    # on the fixture data the recommended shield (Marines) matches the naive
    # pick too -- reassurance case
    assert session.shield_explanation == (
        "Esta opción también es el mejor promedio en escudo, así que coincide con lo que la "
        "intuición sugeriría."
    )


def test_shield_explanation_flags_counterintuitive_pick():
    """When the recommendation genuinely differs from the naive best-average
    pick, the explanation must say so (not just reassure) -- found via a
    fixed seed where the recommended shield ('A4') differs from the naive
    best-average-escudo pick ('A2')."""
    escudo, espada, descarte = _synthetic_matrices(4, seed=1)
    model = n_player.build_model(escudo, espada, descarte)
    session = n_player.LivePairingSession(model, my_team="team1")
    my_report, _ = session.recommend_shield()

    assert my_report.iloc[0]["option"] != n_player._naive_shield(escudo, model.team1_players)
    assert "Aunque" in session.shield_explanation


def test_recommend_swords_sets_swords_explanation(sample_model):
    session = n_player.LivePairingSession(sample_model, my_team="team1")
    my_report, opp_report = session.recommend_shield()
    session.lock_shields(my_report.iloc[0]["option"], opp_report.iloc[0]["option"])
    assert isinstance(session.swords_explanation, str) and session.swords_explanation


# ---------------------------------------------------------------------
# Phase 4 -- engine-level logic/robustness bugs
# ---------------------------------------------------------------------

def _synthetic_matrices(n, seed):
    players_a = [f"A{i}" for i in range(1, n + 1)]
    players_b = [f"B{i}" for i in range(1, n + 1)]
    rng = np.random.default_rng(seed)
    descarte = pd.DataFrame(rng.uniform(0, 20, size=(n, n)).round(1), index=players_a, columns=players_b)
    dep = pd.DataFrame(rng.uniform(0, 8, size=(n, n)).round(1), index=players_a, columns=players_b)
    escudo = (descarte + dep).clip(0, 20)
    espada = (descarte - dep).clip(0, 20)
    return escudo, espada, descarte


def test_explain_deviation_no_contradiction_on_exact_tie():
    """When two accept-stage options tie exactly on direct escudo value, the
    explanation must not claim one is worth strictly more than the other --
    found via a fixed seed where the true recommendation ('B3') isn't the
    first-iterated tied option ('B2' -- what max() would pick by iteration
    order alone).
    """
    escudo, espada, descarte = _synthetic_matrices(4, seed=0)
    model = n_player.build_model(escudo, espada, descarte)
    players_a, players_b = tuple(escudo.index), tuple(escudo.columns)

    explanation = n_player.explain_deviation(model, players_a, players_b, "A3", "B1", ("A1", "A2"), ("B2", "B3"))

    assert explanation is None, (
        f"recommended option ties with the individually-best one on value -- there's no real "
        f"deviation to explain, but got: {explanation!r}"
    )


def test_lock_swords_accepts_reversed_order_pair(sample_model):
    """A validly-membered sword pair given in reversed order must not crash
    -- only membership should matter, not the specific tuple order."""
    session = n_player.LivePairingSession(sample_model, my_team="team1")
    session.recommend_shield()
    session.lock_shields(session.my_players[0], session.opp_players[0])

    my_report, opp_report = session.swords_reports
    real_combo = my_report.iloc[0]["option"]
    reversed_combo = tuple(reversed(real_combo))

    session.lock_swords(reversed_combo, opp_report.iloc[0]["option"])  # should not raise


def test_role_assignment_leftover_matches_always_descarte():
    """Regression, locking in the rule fixed earlier this project: only the
    actively-chosen shield and actively-accepted sword score under their
    named roles -- every leftover match (rejected-sword pair, and any lone
    final combat) must score as descarte, never espada.
    """
    players_a = ["A1", "A2", "A3", "A4"]
    players_b = ["B1", "B2", "B3", "B4"]
    escudo = pd.DataFrame(1.0, index=players_a, columns=players_b)
    espada = pd.DataFrame(2.0, index=players_a, columns=players_b)
    descarte = pd.DataFrame(3.0, index=players_a, columns=players_b)
    model = n_player.build_model(escudo, espada, descarte)

    session = n_player.LivePairingSession(model, my_team="team1")
    my_report, opp_report = session.recommend_shield()
    session.lock_shields(my_report.iloc[0]["option"], opp_report.iloc[0]["option"])
    my_report, opp_report = session.swords_reports
    session.lock_swords(my_report.iloc[0]["option"], opp_report.iloc[0]["option"])
    my_report, opp_report = session.accept_reports
    session.lock_accept(my_report.iloc[0]["option"], opp_report.iloc[0]["option"])

    expected_score = {"escudo": 1.0, "espada": 2.0, "descarte": 3.0}
    for match in session.result["matches"]:
        assert match["team1_score"] == expected_score[match["matrix"]], (
            f"match {match} scored under the wrong role"
        )
    # exactly one escudo match, one espada match, the rest descarte
    roles = [m["matrix"] for m in session.result["matches"]]
    assert roles.count("escudo") == 1
    assert roles.count("espada") == 1
    assert roles.count("descarte") == len(players_a) - 2


def test_full_session_real_data_sums_to_pool(sample_model):
    """Regression: a full walkthrough on the real sheet fixture data sums
    exactly to 20 * team_size."""
    session = n_player.LivePairingSession(sample_model, my_team="team1")
    my_report, opp_report = session.recommend_shield()
    session.lock_shields(my_report.iloc[0]["option"], opp_report.iloc[0]["option"])
    my_report, opp_report = session.swords_reports
    session.lock_swords(my_report.iloc[0]["option"], opp_report.iloc[0]["option"])
    my_report, opp_report = session.accept_reports
    status = session.lock_accept(my_report.iloc[0]["option"], opp_report.iloc[0]["option"])

    assert status == "done"
    assert session.result["my_total"] + session.result["opp_total"] == 80.0


@pytest.mark.parametrize("n,seed", [(5, 1), (6, 2)])
def test_multi_round_recursion(n, seed):
    """Regression: the N-player generalization correctly chains through
    multiple rounds for both odd (direct-1v1 leftover) and even
    (forced-pair leftover) team sizes, accounting for every player exactly
    once and summing to 20*N.
    """
    escudo, espada, descarte = _synthetic_matrices(n, seed)
    model = n_player.build_model(escudo, espada, descarte)
    session = n_player.LivePairingSession(model, my_team="team1")

    while True:
        my_report, opp_report = session.recommend_shield()
        session.lock_shields(my_report.iloc[0]["option"], opp_report.iloc[0]["option"])
        my_report, opp_report = session.swords_reports
        session.lock_swords(my_report.iloc[0]["option"], opp_report.iloc[0]["option"])
        my_report, opp_report = session.accept_reports
        status = session.lock_accept(my_report.iloc[0]["option"], opp_report.iloc[0]["option"])
        if status == "done":
            break

    assert len(session.result["matches"]) == n
    assert abs(session.result["my_total"] + session.result["opp_total"] - 20 * n) < 1e-6
