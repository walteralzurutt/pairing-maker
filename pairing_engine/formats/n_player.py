"""
N-player team pairing format: shield / swords / accept, recursing round by
round over however many players remain.

Generalizes formats.four_player's fixed single round (shield -> throw 2 of
3 -> accept -> 2 leftover matches) to any team size >= 3: each round still
resolves one shield + 2 "swords" (thrown candidates) + 1 accept decision,
consuming exactly 2 players per team, then recurses on whichever players
remain -- until 1 remain (direct final combat) or 2 remain (forced
pairing), the same base cases formats.four_player already had.

ROLE RULE (the one behavioral difference from the original v2 notebook
this was ported from): only the actively-chosen shield and the actively-
accepted sword score under their named escudo/espada roles. EVERY other
match -- the two rejected swords facing each other, the two untouched
players facing each other, or a lone final leftover combat -- scores as
descarte. v2 scored the rejected-swords match as espada; that was a
confirmed bug, fixed here.

Every recommend_*/decision-report function returns plain pandas
DataFrames / dataclasses, no notebook or Streamlit display calls baked
in -- rendering is entirely the UI layer's job, same separation
formats.four_player established.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..zero_sum import solve_zero_sum_game
from ..report import DEFAULT_TIE_TOLERANCE, rank_options, format_option

MIN_TEAM_SIZE = 3  # shield + 2 sword-candidates needs at least this many

SUMMARY_COLUMN_LABELS_ES = {
    "round": "Ronda",
    "matchup_type": "Tipo de partida",
    "you": "Tú",
    "opponent": "Rival",
    "your_score": "Tu puntaje",
}

WORST_CASE_COLUMN_LABELS_ES = {
    "shield": "Escudo propio",
    "worst_case_value": "Valor esperado (peor caso)",
    "worst_case_opponent_shield": "Escudo rival (peor caso)",
    "predicted_opponent_swords": "Espadas rival (peor caso)",
    "should_accept": "Deberíamos aceptar",
    "sacrifice_note": "¿Escudo sacrificado?",
}


def _moda(labels: Sequence, strategy: np.ndarray):
    """The modal (highest-probability) label in an equilibrium strategy."""
    idx = int(np.argmax(strategy))
    return labels[idx], float(strategy[idx])


def _swords_match(a: tuple, b: tuple) -> bool:
    """Two sword pairs are the same pick regardless of tuple order."""
    return set(a) == set(b)


# ---------------------------------------------------------------------
# "Naive" baselines: the pick a human would make by eye, without any game
# theory -- used to explain the equilibrium recommendation when it differs
# (see _explain_shield_choice/_explain_swords_choice below). No new
# solving here, just a simple heuristic over the raw matrices.
# ---------------------------------------------------------------------

def _naive_shield(escudo_df: pd.DataFrame, candidates: Sequence):
    """The candidate with the best average escudo score across all
    opponents -- "who's generally our strongest shield" by eye."""
    return max(candidates, key=lambda p: escudo_df.loc[p, :].mean())


def _naive_swords(descarte_df: pd.DataFrame, candidates: Sequence) -> tuple:
    """The 2 candidates with the best average descarte score -- "throw your
    2 strongest remaining players" by eye."""
    ranked = sorted(candidates, key=lambda p: descarte_df.loc[p, :].mean(), reverse=True)
    return tuple(sorted(ranked[:2]))


# ---------------------------------------------------------------------
# Representative outcome tracing: walk the single most-likely (modal)
# path forward through the nested subgame for one specific candidate
# choice, grounding the naive-vs-recommended comparison in a concrete,
# named matchup with an actual probability -- not just an abstract point
# gap. No new solving: everything here is already memoized in the
# round's breakdown dict.
# ---------------------------------------------------------------------

def _representative_shield_outcome(round_res: dict, shield, opp_shield):
    """Assuming the opponent plays `opp_shield` (their modal pick), who do
    we most likely end up fighting (escudo role) if we pick `shield`, and
    how likely is that specific outcome? Walks: both sides' modal swords
    -> our shield's modal accept."""
    swords_res = round_res["breakdown"][(shield, opp_shield)]
    swordsA_modal, _ = _moda(swords_res["row_options"], swords_res["solution"].row_strategy)
    swordsB_modal, p_swordsB = _moda(swords_res["col_options"], swords_res["solution"].col_strategy)
    accept_res = swords_res["breakdown"][(swordsA_modal, swordsB_modal)]
    opponent, p_accept = _moda(accept_res["row_options"], accept_res["solution"].row_strategy)
    return opponent, p_swordsB * p_accept


def _representative_swords_outcome(swords_res: dict, my_swords):
    """Assuming the opponent throws their modal pick, which of `my_swords`
    do they most likely accept (espada role), and how likely is that?"""
    swordsB_modal, p_swordsB = _moda(swords_res["col_options"], swords_res["solution"].col_strategy)
    accept_res = swords_res["breakdown"][(my_swords, swordsB_modal)]
    my_accepted, p_accept = _moda(accept_res["col_options"], accept_res["solution"].col_strategy)
    return my_accepted, p_swordsB * p_accept


def _explain_shield_choice(model: "NPlayerModel", round_res: dict, recommended, naive) -> str:
    opp_shield, _ = _moda(round_res["remB"], round_res["solution"].col_strategy)
    opp_rec, p_rec = _representative_shield_outcome(round_res, recommended, opp_shield)
    val_rec = float(model.escudo_df.loc[recommended, opp_rec])

    if recommended == naive:
        return (f"Asumiendo que el escudo rival es {opp_shield} (su elección más probable), hay "
                f"una probabilidad del {p_rec:.0%} de que tu escudo ({recommended}) termine "
                f"enfrentando a {opp_rec}, ganando {val_rec:.1f} puntos. Esta también es la opción "
                f"con mejor promedio individual, así que coincide con la intuición.")

    opp_naive, p_naive = _representative_shield_outcome(round_res, naive, opp_shield)
    val_naive = float(model.escudo_df.loc[naive, opp_naive])

    msg = (f"Asumiendo que el escudo rival es {opp_shield} (su elección más probable): con tu "
           f"escudo recomendado ({recommended}) hay un {p_rec:.0%} de probabilidad de enfrentar a "
           f"{opp_rec}, ganando {val_rec:.1f} puntos. Con {naive} (el de mejor promedio individual), "
           f"hay un {p_naive:.0%} de enfrentar a {opp_naive}, ganando {val_naive:.1f} puntos.")

    if val_rec < val_naive:
        row_idx = {p: i for i, p in enumerate(round_res["remA"])}
        avg_rec = float(round_res["payoff"][row_idx[recommended], :] @ round_res["solution"].col_strategy)
        avg_naive = float(round_res["payoff"][row_idx[naive], :] @ round_res["solution"].col_strategy)
        msg += (f" Aunque en este escenario concreto {naive} rendiría más, {recommended} sigue "
                f"siendo mejor en promedio contra todas las respuestas posibles del rival "
                f"({avg_rec:.1f} vs {avg_naive:.1f} puntos esperados).")
    return msg


def _explain_swords_choice(model: "NPlayerModel", swords_res: dict, shieldB, recommended, naive) -> str:
    swordsB_modal, _ = _moda(swords_res["col_options"], swords_res["solution"].col_strategy)
    acc_rec, p_rec = _representative_swords_outcome(swords_res, recommended)
    val_rec = float(model.espada_df.loc[acc_rec, shieldB])

    if _swords_match(recommended, naive):
        return (f"Asumiendo que el rival lanza a {format_option(swordsB_modal)} (su elección más "
                f"probable), hay una probabilidad del {p_rec:.0%} de que el escudo rival acepte a "
                f"{acc_rec}, ganando {val_rec:.1f} puntos. Esta también es la opción con mejor "
                f"promedio individual, así que coincide con la intuición.")

    acc_naive, p_naive = _representative_swords_outcome(swords_res, naive)
    val_naive = float(model.espada_df.loc[acc_naive, shieldB])

    msg = (f"Asumiendo que el rival lanza a {format_option(swordsB_modal)} (su elección más "
           f"probable): con tus espadas recomendadas ({format_option(recommended)}) hay un "
           f"{p_rec:.0%} de que el escudo rival acepte a {acc_rec}, ganando {val_rec:.1f} puntos. "
           f"Con {format_option(naive)} (mejor promedio individual), hay un {p_naive:.0%} de que "
           f"acepte a {acc_naive}, ganando {val_naive:.1f} puntos.")

    if val_rec < val_naive:
        row_idx = {opt: i for i, opt in enumerate(swords_res["row_options"])}
        avg_rec = float(swords_res["payoff"][row_idx[recommended], :] @ swords_res["solution"].col_strategy)
        avg_naive = float(swords_res["payoff"][row_idx[naive], :] @ swords_res["solution"].col_strategy)
        msg += (f" Aunque en este escenario concreto {format_option(naive)} rendiría más, "
                f"{format_option(recommended)} sigue siendo mejor en promedio contra todas las "
                f"respuestas posibles del rival ({avg_rec:.1f} vs {avg_naive:.1f} puntos esperados).")
    return msg


# ---------------------------------------------------------------------
# Stage 3 (accept): given a shield pair and two thrown sword pairs, each
# shield secretly picks which of the RIVAL's 2 thrown swords to accept --
# a 2x2 simultaneous game, since whichever candidate gets rejected also
# determines who's left over for the following matches.
# ---------------------------------------------------------------------

def solve_accept_stage(shieldA, shieldB, swordsA: Sequence, swordsB: Sequence,
                        remA: Sequence, remB: Sequence,
                        escudo_df: pd.DataFrame, espada_df: pd.DataFrame,
                        descarte_df: pd.DataFrame, memo: dict) -> dict:
    row_options = list(swordsB)   # A's shield picks among these (accepted -> "j_star")
    col_options = list(swordsA)   # B's shield picks among these (accepted -> "i_star")

    payoff = np.zeros((len(row_options), len(col_options)))
    for a_idx, j_star in enumerate(row_options):
        for b_idx, i_star in enumerate(col_options):
            immediate = (float(escudo_df.loc[shieldA, j_star])
                         + float(espada_df.loc[i_star, shieldB]))

            remA_next = tuple(x for x in remA if x not in (shieldA, i_star))
            remB_next = tuple(x for x in remB if x not in (shieldB, j_star))

            if len(remA_next) == 1:
                # lone leftover -> direct final combat, scored as descarte
                continuation = float(descarte_df.loc[remA_next[0], remB_next[0]])
            elif len(remA_next) == 2:
                # forced pairing: rejected-vs-rejected AND untouched-vs-untouched,
                # both scored as descarte
                i_left = [x for x in swordsA if x != i_star][0]
                j_left = [x for x in swordsB if x != j_star][0]
                d1 = [x for x in remA_next if x != i_left][0]
                d2 = [x for x in remB_next if x != j_left][0]
                continuation = (float(descarte_df.loc[i_left, j_left])
                                 + float(descarte_df.loc[d1, d2]))
            else:
                continuation = solve_round(remA_next, remB_next, escudo_df, espada_df,
                                            descarte_df, memo)["solution"].value

            payoff[a_idx, b_idx] = immediate + continuation

    solution = solve_zero_sum_game(payoff)
    return {"row_options": row_options, "col_options": col_options, "payoff": payoff, "solution": solution}


# ---------------------------------------------------------------------
# Stage 2 (swords): given a shield pair, each team secretly picks which 2
# of its remaining players to throw at the opponent's shield.
# ---------------------------------------------------------------------

def solve_swords_stage(shieldA, shieldB, remA: Sequence, remB: Sequence,
                        escudo_df: pd.DataFrame, espada_df: pd.DataFrame,
                        descarte_df: pd.DataFrame, memo: dict) -> dict:
    candA = [x for x in remA if x != shieldA]
    candB = [x for x in remB if x != shieldB]
    subsetsA = list(itertools.combinations(candA, 2))
    subsetsB = list(itertools.combinations(candB, 2))

    payoff = np.zeros((len(subsetsA), len(subsetsB)))
    breakdown = {}
    for i, sA in enumerate(subsetsA):
        for j, sB in enumerate(subsetsB):
            res = solve_accept_stage(shieldA, shieldB, sA, sB, remA, remB,
                                      escudo_df, espada_df, descarte_df, memo)
            payoff[i, j] = res["solution"].value
            breakdown[(sA, sB)] = res

    solution = solve_zero_sum_game(payoff)
    return {"row_options": subsetsA, "col_options": subsetsB, "payoff": payoff,
            "breakdown": breakdown, "solution": solution}


# ---------------------------------------------------------------------
# Stage 1 (shield): each team secretly picks its shield for this round.
# Memoized on the (sorted) remaining-player sets, since the recursive
# continuation revisits overlapping sub-games across different branches.
# ---------------------------------------------------------------------

def solve_round(remA: Sequence, remB: Sequence,
                 escudo_df: pd.DataFrame, espada_df: pd.DataFrame,
                 descarte_df: pd.DataFrame, memo: dict) -> dict:
    remA = tuple(sorted(remA))
    remB = tuple(sorted(remB))
    key = (remA, remB)
    if key in memo:
        return memo[key]

    k = len(remA)
    if k != len(remB):
        raise ValueError("Both teams must have the same number of remaining players.")
    if k < MIN_TEAM_SIZE:
        raise ValueError(f"solve_round only applies with >= {MIN_TEAM_SIZE} remaining players per team.")

    payoff = np.zeros((k, k))
    breakdown = {}
    for i, sA in enumerate(remA):
        for j, sB in enumerate(remB):
            res = solve_swords_stage(sA, sB, remA, remB, escudo_df, espada_df, descarte_df, memo)
            payoff[i, j] = res["solution"].value
            breakdown[(sA, sB)] = res

    solution = solve_zero_sum_game(payoff)
    result = {"remA": remA, "remB": remB, "payoff": payoff, "breakdown": breakdown, "solution": solution}
    memo[key] = result
    return result


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

@dataclass
class NPlayerModel:
    escudo_df: pd.DataFrame
    espada_df: pd.DataFrame
    descarte_df: pd.DataFrame
    team1_players: list
    team2_players: list
    memo: dict = field(default_factory=dict)

    def solve_round(self, remA: Sequence, remB: Sequence) -> dict:
        return solve_round(tuple(remA), tuple(remB), self.escudo_df, self.espada_df, self.descarte_df, self.memo)

    @property
    def expected_team1_score(self) -> float:
        return self.solve_round(self.team1_players, self.team2_players)["solution"].value

    @property
    def expected_team2_score(self) -> float:
        return 20.0 * len(self.team1_players) - self.expected_team1_score


def build_model(escudo_df: pd.DataFrame, espada_df: pd.DataFrame, descarte_df: pd.DataFrame) -> NPlayerModel:
    team1_players = list(escudo_df.index)
    team2_players = list(escudo_df.columns)
    n1, n2 = len(team1_players), len(team2_players)
    if n1 != n2:
        raise ValueError(f"Both teams must have the same number of players, got {n1} (team1) and {n2} (team2).")
    if n1 < MIN_TEAM_SIZE:
        raise ValueError(
            f"pairing_engine.formats.n_player requires at least {MIN_TEAM_SIZE} players per team "
            f"(got {n1}) -- the shield + 2 sword-candidates mechanic needs at least that many."
        )
    shapes = {escudo_df.shape, espada_df.shape, descarte_df.shape}
    if len(shapes) != 1:
        raise ValueError(f"escudo/espada/descarte must all share one shape, got {shapes}.")

    model = NPlayerModel(escudo_df=escudo_df, espada_df=espada_df, descarte_df=descarte_df,
                          team1_players=team1_players, team2_players=team2_players)
    model.solve_round(team1_players, team2_players)  # solve eagerly so build_model() surfaces errors up front
    return model


# ---------------------------------------------------------------------
# Decision reports (primary recommended-pick tables, via the shared
# rank_options tie-break rule) -- one per stage, symmetric per perspective.
# `total_here` is 20 points per match times however many players remain
# THIS round (not the full original team size): that's exactly how many
# individual matches this round's value + its recursive continuation
# covers, so it's the correct pool to complement against for team2's view.
# ---------------------------------------------------------------------

def _stage_report(payoff: np.ndarray, solution, row_labels: Sequence, col_labels: Sequence,
                   total_here: float, perspective: str, tie_tolerance: float, tie_break: bool) -> pd.DataFrame:
    """Shared shape behind shield/swords/accept_decision_report: orient an
    already-solved stage's payoff matrix for the requested perspective and
    rank it via rank_options. `total_here` is 20 points per match times
    however many players remain THIS round (not the full original team
    size): that's exactly how many individual matches this round's value +
    its recursive continuation covers, so it's the correct pool to
    complement against for team2's view.
    """
    if perspective == "team1":
        payoff_for_me = payoff
        opp_strategy = solution.col_strategy
        my_strategy = solution.row_strategy
        option_labels = list(row_labels)
    elif perspective == "team2":
        payoff_for_me = (total_here - payoff).T
        opp_strategy = solution.row_strategy
        my_strategy = solution.col_strategy
        option_labels = list(col_labels)
    else:
        raise ValueError("perspective must be 'team1' or 'team2'")

    return rank_options(payoff_for_me, opp_strategy, my_strategy, option_labels, tie_tolerance, tie_break)


def shield_decision_report(model: NPlayerModel, remA: Sequence, remB: Sequence, perspective: str = "team1",
                            tie_tolerance: float = DEFAULT_TIE_TOLERANCE, tie_break: bool = True,
                            known_opponent_shield: Optional[str] = None) -> pd.DataFrame:
    """Stage-1 (shield) insight table. If `known_opponent_shield` is given,
    conditions on that shield being certain rather than a mix (used for the
    "I already know their shield" pre-declaration before round 1) --
    reduces to a deterministic single-column ranking, still tie-broken the
    same way via rank_options (when tie_break=True).

    Pass tie_break=True (default) when this table describes OUR OWN
    options -- the near-tie/lower-spread override is valid advice there.
    Pass tie_break=False when predicting the OPPONENT's likely pick, so
    the top row reflects their actual equilibrium weight instead (see
    rank_options' docstring for why).
    """
    round_res = model.solve_round(remA, remB)
    total_here = 20.0 * len(round_res["remA"])

    if known_opponent_shield is None:
        return _stage_report(round_res["payoff"], round_res["solution"], round_res["remA"], round_res["remB"],
                              total_here, perspective, tie_tolerance, tie_break)

    # Deterministic conditioning: collapse to the single column/row for the
    # declared shield instead of the ordinary equilibrium mix.
    if perspective == "team1":
        full_payoff = round_res["payoff"]
        option_labels = list(round_res["remA"])
        rival_labels = list(round_res["remB"])
    elif perspective == "team2":
        full_payoff = (total_here - round_res["payoff"]).T
        option_labels = list(round_res["remB"])
        rival_labels = list(round_res["remA"])
    else:
        raise ValueError("perspective must be 'team1' or 'team2'")

    j = rival_labels.index(known_opponent_shield)
    payoff_for_me = full_payoff[:, [j]]
    opp_strategy = np.array([1.0])
    best_idx = int(np.argmax(payoff_for_me[:, 0]))
    my_strategy = np.zeros(len(option_labels))
    my_strategy[best_idx] = 1.0

    return rank_options(payoff_for_me, opp_strategy, my_strategy, option_labels, tie_tolerance, tie_break)


def swords_decision_report(model: NPlayerModel, remA: Sequence, remB: Sequence, shieldA, shieldB,
                            perspective: str = "team1", tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
                            tie_break: bool = True) -> pd.DataFrame:
    """Stage-2 (swords) insight table for an already-known shield pair."""
    round_res = model.solve_round(remA, remB)
    total_here = 20.0 * len(round_res["remA"])
    swords_res = round_res["breakdown"][(shieldA, shieldB)]
    return _stage_report(swords_res["payoff"], swords_res["solution"],
                          swords_res["row_options"], swords_res["col_options"],
                          total_here, perspective, tie_tolerance, tie_break)


def accept_decision_report(model: NPlayerModel, remA: Sequence, remB: Sequence, shieldA, shieldB,
                            swordsA: Sequence, swordsB: Sequence, perspective: str = "team1",
                            tie_tolerance: float = DEFAULT_TIE_TOLERANCE, tie_break: bool = True) -> pd.DataFrame:
    """Stage-3 (accept) insight table for already-known shields AND swords."""
    round_res = model.solve_round(remA, remB)
    total_here = 20.0 * len(round_res["remA"])
    swords_res = round_res["breakdown"][(shieldA, shieldB)]
    accept_res = swords_res["breakdown"][(swordsA, swordsB)]
    return _stage_report(accept_res["payoff"], accept_res["solution"],
                          accept_res["row_options"], accept_res["col_options"],
                          total_here, perspective, tie_tolerance, tie_break)


# ---------------------------------------------------------------------
# Worst-case (adversarial) table: assume the opponent already knows our
# shield pick and responds to hurt us specifically, rather than playing
# their equilibrium mix. Always from team1's ("our") perspective, matching
# how this app always treats team1/rows as us.
# ---------------------------------------------------------------------

def worst_case_report(model: NPlayerModel, remA: Sequence, remB: Sequence) -> pd.DataFrame:
    round_res = model.solve_round(remA, remB)
    remA_sorted, remB_sorted = round_res["remA"], round_res["remB"]
    payoff = round_res["payoff"]

    rows = []
    for i, a in enumerate(remA_sorted):
        values = payoff[i, :]
        j_worst = int(np.argmin(values))
        b_worst = remB_sorted[j_worst]
        worst_value = float(values[j_worst])

        swords_res = round_res["breakdown"][(a, b_worst)]
        own_pair, _ = _moda(swords_res["row_options"], swords_res["solution"].row_strategy)
        rival_pair, _ = _moda(swords_res["col_options"], swords_res["solution"].col_strategy)
        accept_res = swords_res["breakdown"][(own_pair, rival_pair)]
        recommended, _ = _moda(accept_res["row_options"], accept_res["solution"].row_strategy)

        direct = float(model.escudo_df.loc[a, recommended])
        others = [x for x in rival_pair if x != recommended]
        other = others[0] if others else None
        if other is not None:
            direct_other = float(model.escudo_df.loc[a, other])
            sacrificed = direct_other > direct + 1e-6
        else:
            direct_other = direct
            sacrificed = False

        if sacrificed:
            note = (f"Sí: aceptar a {recommended} nos cuesta {(direct_other - direct):.2f} puntos "
                     f"frente a aceptar a {other} en el enfrentamiento directo del escudo, pero el "
                     f"resultado total del equipo mejora.")
        else:
            note = "No: la opción individualmente mejor para el escudo también es la mejor para el equipo."

        rows.append({
            "shield": a,
            "worst_case_value": worst_value,
            "worst_case_opponent_shield": b_worst,
            "predicted_opponent_swords": " y ".join(rival_pair),
            "should_accept": recommended,
            "sacrifice_note": note,
        })

    return pd.DataFrame(rows).sort_values("worst_case_value", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------
# Deviation explanation: why the accept-stage recommendation isn't the
# individually-best pick for the shield's own match, if it isn't. Always
# from team1's perspective, same restriction as worst_case_report.
# ---------------------------------------------------------------------

def explain_deviation(model: NPlayerModel, remA: Sequence, remB: Sequence,
                       shieldA, shieldB, swordsA: Sequence, swordsB: Sequence) -> Optional[str]:
    round_res = model.solve_round(remA, remB)
    swords_res = round_res["breakdown"][(shieldA, shieldB)]
    accept_res = swords_res["breakdown"][(swordsA, swordsB)]
    solution = accept_res["solution"]

    recommended, _ = _moda(accept_res["row_options"], solution.row_strategy)
    direct_values = {b: float(model.escudo_df.loc[shieldA, b]) for b in swordsB}
    if math.isclose(direct_values[recommended], max(direct_values.values()), abs_tol=1e-6):
        # recommended already ties for individually-best -- nothing to explain,
        # regardless of which tied option max() happens to return first
        return None
    best_direct = max(direct_values, key=direct_values.get)

    Q = solution.col_strategy  # opponent shield's equilibrium mix over accepting our swordsA candidates
    payoff = accept_res["payoff"]
    row_idx = {b: i for i, b in enumerate(accept_res["row_options"])}
    total_recommended = float(payoff[row_idx[recommended], :] @ Q)
    total_alternative = float(payoff[row_idx[best_direct], :] @ Q)

    sa_modal, _ = _moda(accept_res["col_options"], Q)  # our sword the opponent shield most likely accepts
    remA_next_base = tuple(x for x in remA if x not in (shieldA, sa_modal))

    def remB_next_for(b_accepted):
        return tuple(x for x in remB if x not in (shieldB, b_accepted))

    k_next = len(remA_next_base)

    if k_next == 1:
        a = remA_next_base[0]
        b_rec, b_alt = remB_next_for(recommended)[0], remB_next_for(best_direct)[0]
        v_rec = float(model.descarte_df.loc[a, b_rec])
        v_alt = float(model.descarte_df.loc[a, b_alt])
        detail = (f"En concreto: aceptando a {recommended}, el combate final sería {a} vs {b_rec} "
                  f"(puntúa {v_rec:.2f}); aceptando a {best_direct} sería en cambio {a} vs {b_alt} "
                  f"(puntúa {v_alt:.2f}), una diferencia de {(v_rec - v_alt):+.2f} en ese combate.")
    elif k_next == 2:
        rejected_a = [x for x in swordsA if x != sa_modal][0]

        def parts_b(b_accepted):
            rejected_b = [x for x in swordsB if x != b_accepted][0]
            leftover_b = [x for x in remB_next_for(b_accepted) if x != rejected_b][0]
            return rejected_b, leftover_b

        rej_b_rec, _ = parts_b(recommended)
        rej_b_alt, _ = parts_b(best_direct)
        v_rec = float(model.descarte_df.loc[rejected_a, rej_b_rec])
        v_alt = float(model.descarte_df.loc[rejected_a, rej_b_alt])
        detail = (f"En concreto: aceptando a {recommended}, tu jugador rechazado {rejected_a} se "
                  f"enfrenta en el emparejamiento forzoso a {rej_b_rec} (puntúa {v_rec:.2f}); "
                  f"aceptando a {best_direct} se enfrentaría en cambio a {rej_b_alt} (puntúa {v_alt:.2f}), "
                  f"una diferencia de {(v_rec - v_alt):+.2f} en ese combate.")
    else:
        detail = ("La diferencia viene de cómo queda determinado el resto de emparejamientos en las "
                  "rondas siguientes, no de un único combate concreto.")

    if total_recommended > total_alternative + 1e-6:
        comparison = f"es mayor: {total_recommended:.2f} frente a {total_alternative:.2f} si se aceptara a {best_direct}"
    else:
        comparison = (f"es igual en ambos casos ({total_recommended:.2f}): aceptar a {recommended} es al "
                       f"menos tan buena elección como aceptar a {best_direct}")

    return (f"Individualmente tu escudo ({shieldA}) sacaría más puntos aceptando a {best_direct} "
            f"({direct_values[best_direct]:.2f} puntos) que aceptando a {recommended} "
            f"({direct_values[recommended]:.2f} puntos). Aun así, se recomienda aceptar a {recommended} "
            f"porque el valor esperado TOTAL del equipo {comparison}. {detail}")


def summarize_matches(matches: list, my_team: str) -> pd.DataFrame:
    """Reshape every resolved matchup across every round (as accumulated by
    LivePairingSession) into a plain, presentation-agnostic DataFrame.
    """
    rows = []
    for m in matches:
        t1_name, t2_name = m["team1_player"], m["team2_player"]
        mine, theirs = (t1_name, t2_name) if my_team == "team1" else (t2_name, t1_name)
        my_score = m["team1_score"] if my_team == "team1" else 20.0 - m["team1_score"]
        rows.append({"round": m["round"], "matchup_type": m["matrix"], "you": mine, "opponent": theirs,
                      "your_score": my_score})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Live pairing session: stateful walk-through of one real match, looping
# back into a new round for as long as players remain. Every recommend_*
# method returns (my_report, opp_report) DataFrames -- no printing/display
# side effects -- so the UI layer decides how to render them.
# ---------------------------------------------------------------------

class LivePairingSession:
    """`my_team` is "team1" or "team2" depending on which side of the
    escudo/espada/descarte matrices your team is. The worst-case table and
    deviation explanation are only computed when my_team == "team1" (the
    app always treats team1/rows as us)."""

    def __init__(self, model: NPlayerModel, my_team: str = "team1"):
        if my_team not in ("team1", "team2"):
            raise ValueError("my_team must be 'team1' or 'team2'")
        self.model = model
        self.my_team = my_team
        self.opp_team = "team2" if my_team == "team1" else "team1"

        self.remA: Tuple = tuple(model.team1_players)
        self.remB: Tuple = tuple(model.team2_players)
        self.round_number = 1

        self.my_shield = None
        self.opp_shield = None
        self.my_swords = None
        self.opp_swords = None
        self.shield_reports: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None
        self.swords_reports: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None
        self.accept_reports: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None
        self.worst_case: Optional[pd.DataFrame] = None
        self.shield_explanation: Optional[str] = None
        self.swords_explanation: Optional[str] = None
        self.deviation_explanation: Optional[str] = None

        self.history: list = []
        self.result: Optional[dict] = None

    @property
    def my_players(self) -> list:
        return list(self.remA if self.my_team == "team1" else self.remB)

    @property
    def opp_players(self) -> list:
        return list(self.remB if self.my_team == "team1" else self.remA)

    def _to_team1_team2(self, mine, theirs):
        return (mine, theirs) if self.my_team == "team1" else (theirs, mine)

    def _validate_player(self, name, valid_list, field_name):
        if name not in valid_list:
            raise ValueError(f"{field_name}={name!r} is not one of {valid_list}")

    def _validate_pair(self, pair, valid_list, field_name):
        if len(pair) != 2 or len(set(pair)) != 2:
            raise ValueError(f"{field_name} must be exactly 2 distinct players, got {pair}")
        for p in pair:
            if p not in valid_list:
                raise ValueError(f"{field_name} contains {p!r}, which is not one of {valid_list}")

    # ---- Stage 1: shield ------------------------------------------------
    def recommend_shield(self, known_opponent_shield: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Call BEFORE committing to a shield. `known_opponent_shield` (only
        honored on round 1) lets you condition on already knowing the
        opponent's shield for this round instead of their equilibrium mix.
        """
        koc = known_opponent_shield if self.round_number == 1 else None
        my_report = shield_decision_report(self.model, self.remA, self.remB, perspective=self.my_team,
                                            known_opponent_shield=koc)
        opp_report = shield_decision_report(self.model, self.remA, self.remB, perspective=self.opp_team,
                                             tie_break=False)
        if koc is not None:
            # We're not predicting anymore -- we were told this for certain, so
            # the "probable pick" must be the declared shield, not whatever the
            # (now-irrelevant) equilibrium mix happens to favor.
            opp_report = opp_report.copy()
            opp_report["equilibrium_weight"] = 0.0
            opp_report.loc[opp_report["option"] == koc, "equilibrium_weight"] = 1.0
            opp_report = opp_report.sort_values(
                ["equilibrium_weight", "vs_equilibrium_opponent"], ascending=[False, False]
            ).reset_index(drop=True)
        self.shield_reports = (my_report, opp_report)
        self.worst_case = worst_case_report(self.model, self.remA, self.remB) if self.my_team == "team1" else None
        if self.my_team == "team1":
            naive = _naive_shield(self.model.escudo_df, self.my_players)
            round_res = self.model.solve_round(self.remA, self.remB)
            self.shield_explanation = _explain_shield_choice(
                self.model, round_res, my_report.iloc[0]["option"], naive)
        else:
            self.shield_explanation = None
        return my_report, opp_report

    def lock_shields(self, my_shield: str, opp_shield: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._validate_player(my_shield, self.my_players, "my_shield")
        self._validate_player(opp_shield, self.opp_players, "opp_shield")
        self.my_shield, self.opp_shield = my_shield, opp_shield
        return self.recommend_swords()

    # ---- Stage 2: swords --------------------------------------------------
    def recommend_swords(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        s1, s2 = self._to_team1_team2(self.my_shield, self.opp_shield)
        my_report = swords_decision_report(self.model, self.remA, self.remB, s1, s2, perspective=self.my_team)
        opp_report = swords_decision_report(self.model, self.remA, self.remB, s1, s2, perspective=self.opp_team,
                                             tie_break=False)
        self.swords_reports = (my_report, opp_report)
        if self.my_team == "team1":
            remaining = [p for p in self.my_players if p != self.my_shield]
            naive = _naive_swords(self.model.descarte_df, remaining)
            round_res = self.model.solve_round(self.remA, self.remB)
            swords_res = round_res["breakdown"][(s1, s2)]
            self.swords_explanation = _explain_swords_choice(
                self.model, swords_res, s2, my_report.iloc[0]["option"], naive)
        else:
            self.swords_explanation = None
        return my_report, opp_report

    def lock_swords(self, my_swords: Sequence[str], opp_swords: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        remaining_mine = [p for p in self.my_players if p != self.my_shield]
        remaining_theirs = [p for p in self.opp_players if p != self.opp_shield]
        my_swords, opp_swords = tuple(my_swords), tuple(opp_swords)
        self._validate_pair(my_swords, remaining_mine, "my_swords")
        self._validate_pair(opp_swords, remaining_theirs, "opp_swords")
        # Normalize to the canonical (sorted) order used internally as breakdown-dict
        # keys -- only membership should matter, not which order the pair was given in.
        self.my_swords, self.opp_swords = tuple(sorted(my_swords)), tuple(sorted(opp_swords))
        return self.recommend_accept()

    # ---- Stage 3: accept ----------------------------------------------------
    def recommend_accept(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        s1, s2 = self._to_team1_team2(self.my_shield, self.opp_shield)
        t1, t2 = self._to_team1_team2(self.my_swords, self.opp_swords)
        my_report = accept_decision_report(self.model, self.remA, self.remB, s1, s2, t1, t2, perspective=self.my_team)
        opp_report = accept_decision_report(self.model, self.remA, self.remB, s1, s2, t1, t2,
                                             perspective=self.opp_team, tie_break=False)
        self.accept_reports = (my_report, opp_report)
        self.deviation_explanation = (
            explain_deviation(self.model, self.remA, self.remB, s1, s2, t1, t2)
            if self.my_team == "team1" else None
        )
        return my_report, opp_report

    # ---- Resolution -----------------------------------------------------
    def lock_accept(self, my_pick: Optional[str] = None, opp_pick: Optional[str] = None) -> str:
        """Records the actual accept picks, resolves this round's 2 primary
        matches (+ any auto-determined leftover matches, if the round ends
        here), and either finalizes the whole session (self.result set,
        returns "done") or advances to a new round -- call recommend_shield()
        again -- (returns "continue"). Leave a pick as None to fall back to
        the model's top recommendation/prediction for it.
        """
        s1, s2 = self._to_team1_team2(self.my_shield, self.opp_shield)
        t1, t2 = self._to_team1_team2(self.my_swords, self.opp_swords)
        my_report, opp_report = self.accept_reports
        my_pick = my_pick or my_report.iloc[0]["option"]
        opp_pick = opp_pick or opp_report.iloc[0]["option"]
        # sb = which B sword A's shield accepted; sa = which A sword B's shield accepted
        sb, sa = (my_pick, opp_pick) if self.my_team == "team1" else (opp_pick, my_pick)

        self.history.append({"team1_player": s1, "team2_player": sb, "matrix": "escudo",
                              "team1_score": float(self.model.escudo_df.loc[s1, sb]),
                              "round": self.round_number})
        self.history.append({"team1_player": sa, "team2_player": s2, "matrix": "espada",
                              "team1_score": float(self.model.espada_df.loc[sa, s2]),
                              "round": self.round_number})

        remA_next = tuple(x for x in self.remA if x not in (s1, sa))
        remB_next = tuple(x for x in self.remB if x not in (s2, sb))

        if len(remA_next) == 1:
            a, b = remA_next[0], remB_next[0]
            self.history.append({"team1_player": a, "team2_player": b, "matrix": "descarte",
                                  "team1_score": float(self.model.descarte_df.loc[a, b]),
                                  "round": self.round_number})
            self._finish()
            return "done"

        if len(remA_next) == 2:
            rejected_a = [x for x in t1 if x != sa][0]
            rejected_b = [x for x in t2 if x != sb][0]
            leftover_a = [x for x in remA_next if x != rejected_a][0]
            leftover_b = [x for x in remB_next if x != rejected_b][0]
            self.history.append({"team1_player": rejected_a, "team2_player": rejected_b, "matrix": "descarte",
                                  "team1_score": float(self.model.descarte_df.loc[rejected_a, rejected_b]),
                                  "round": self.round_number})
            self.history.append({"team1_player": leftover_a, "team2_player": leftover_b, "matrix": "descarte",
                                  "team1_score": float(self.model.descarte_df.loc[leftover_a, leftover_b]),
                                  "round": self.round_number})
            self._finish()
            return "done"

        # >= 3 remain -> another round
        self.remA, self.remB = remA_next, remB_next
        self.round_number += 1
        self.my_shield = self.opp_shield = self.my_swords = self.opp_swords = None
        self.shield_reports = self.swords_reports = self.accept_reports = None
        self.worst_case = None
        self.shield_explanation = None
        self.swords_explanation = None
        self.deviation_explanation = None
        return "continue"

    def _finish(self):
        team_size = len(self.model.team1_players)
        total_pool = 20.0 * team_size
        team1_total = sum(m["team1_score"] for m in self.history)
        my_total = team1_total if self.my_team == "team1" else total_pool - team1_total
        opp_total = total_pool - my_total
        self.result = {"matches": self.history, "my_total": my_total, "opp_total": opp_total,
                        "team_size": team_size, "total_pool": total_pool}
