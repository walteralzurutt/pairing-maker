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
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..zero_sum import solve_zero_sum_game
from ..report import DEFAULT_TIE_TOLERANCE, rank_options

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

def shield_decision_report(model: NPlayerModel, remA: Sequence, remB: Sequence, perspective: str = "team1",
                            tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
                            known_opponent_shield: Optional[str] = None) -> pd.DataFrame:
    """Stage-1 (shield) insight table. If `known_opponent_shield` is given,
    conditions on that shield being certain rather than a mix (used for the
    "I already know their shield" pre-declaration before round 1) --
    reduces to a deterministic single-column ranking, still tie-broken the
    same way via rank_options.
    """
    round_res = model.solve_round(remA, remB)
    total_here = 20.0 * len(round_res["remA"])
    solution = round_res["solution"]

    if perspective == "team1":
        full_payoff = round_res["payoff"]
        full_opp_strategy = solution.col_strategy
        full_my_strategy = solution.row_strategy
        option_labels = list(round_res["remA"])
        rival_labels = list(round_res["remB"])
    elif perspective == "team2":
        full_payoff = (total_here - round_res["payoff"]).T
        full_opp_strategy = solution.row_strategy
        full_my_strategy = solution.col_strategy
        option_labels = list(round_res["remB"])
        rival_labels = list(round_res["remA"])
    else:
        raise ValueError("perspective must be 'team1' or 'team2'")

    if known_opponent_shield is not None:
        j = rival_labels.index(known_opponent_shield)
        payoff_for_me = full_payoff[:, [j]]
        opp_strategy = np.array([1.0])
        best_idx = int(np.argmax(payoff_for_me[:, 0]))
        my_strategy = np.zeros(len(option_labels))
        my_strategy[best_idx] = 1.0
    else:
        payoff_for_me = full_payoff
        opp_strategy = full_opp_strategy
        my_strategy = full_my_strategy

    return rank_options(payoff_for_me, opp_strategy, my_strategy, option_labels, tie_tolerance)


def swords_decision_report(model: NPlayerModel, remA: Sequence, remB: Sequence, shieldA, shieldB,
                            perspective: str = "team1", tie_tolerance: float = DEFAULT_TIE_TOLERANCE) -> pd.DataFrame:
    """Stage-2 (swords) insight table for an already-known shield pair."""
    round_res = model.solve_round(remA, remB)
    total_here = 20.0 * len(round_res["remA"])
    swords_res = round_res["breakdown"][(shieldA, shieldB)]
    solution = swords_res["solution"]

    if perspective == "team1":
        payoff_for_me = swords_res["payoff"]
        opp_strategy = solution.col_strategy
        my_strategy = solution.row_strategy
        option_labels = swords_res["row_options"]
    elif perspective == "team2":
        payoff_for_me = (total_here - swords_res["payoff"]).T
        opp_strategy = solution.row_strategy
        my_strategy = solution.col_strategy
        option_labels = swords_res["col_options"]
    else:
        raise ValueError("perspective must be 'team1' or 'team2'")

    return rank_options(payoff_for_me, opp_strategy, my_strategy, option_labels, tie_tolerance)


def accept_decision_report(model: NPlayerModel, remA: Sequence, remB: Sequence, shieldA, shieldB,
                            swordsA: Sequence, swordsB: Sequence, perspective: str = "team1",
                            tie_tolerance: float = DEFAULT_TIE_TOLERANCE) -> pd.DataFrame:
    """Stage-3 (accept) insight table for already-known shields AND swords."""
    round_res = model.solve_round(remA, remB)
    total_here = 20.0 * len(round_res["remA"])
    swords_res = round_res["breakdown"][(shieldA, shieldB)]
    accept_res = swords_res["breakdown"][(swordsA, swordsB)]
    solution = accept_res["solution"]

    if perspective == "team1":
        payoff_for_me = accept_res["payoff"]
        opp_strategy = solution.col_strategy
        my_strategy = solution.row_strategy
        option_labels = accept_res["row_options"]
    elif perspective == "team2":
        payoff_for_me = (total_here - accept_res["payoff"]).T
        opp_strategy = solution.row_strategy
        my_strategy = solution.col_strategy
        option_labels = accept_res["col_options"]
    else:
        raise ValueError("perspective must be 'team1' or 'team2'")

    return rank_options(payoff_for_me, opp_strategy, my_strategy, option_labels, tie_tolerance)


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
            note = (f"Sí: cede {(direct_other - direct):.2f} pts individuales aceptando a "
                     f"{recommended} en vez de a {other}, pero mejora el resultado total del equipo.")
        else:
            note = "No: la mejor opción individual del escudo también es la mejor para el equipo."

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
    best_direct = max(direct_values, key=direct_values.get)
    if best_direct == recommended:
        return None

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
        opp_report = shield_decision_report(self.model, self.remA, self.remB, perspective=self.opp_team)
        self.shield_reports = (my_report, opp_report)
        self.worst_case = worst_case_report(self.model, self.remA, self.remB) if self.my_team == "team1" else None
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
        opp_report = swords_decision_report(self.model, self.remA, self.remB, s1, s2, perspective=self.opp_team)
        self.swords_reports = (my_report, opp_report)
        return my_report, opp_report

    def lock_swords(self, my_swords: Sequence[str], opp_swords: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        remaining_mine = [p for p in self.my_players if p != self.my_shield]
        remaining_theirs = [p for p in self.opp_players if p != self.opp_shield]
        my_swords, opp_swords = tuple(my_swords), tuple(opp_swords)
        self._validate_pair(my_swords, remaining_mine, "my_swords")
        self._validate_pair(opp_swords, remaining_theirs, "opp_swords")
        self.my_swords, self.opp_swords = my_swords, opp_swords
        return self.recommend_accept()

    # ---- Stage 3: accept ----------------------------------------------------
    def recommend_accept(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        s1, s2 = self._to_team1_team2(self.my_shield, self.opp_shield)
        t1, t2 = self._to_team1_team2(self.my_swords, self.opp_swords)
        my_report = accept_decision_report(self.model, self.remA, self.remB, s1, s2, t1, t2, perspective=self.my_team)
        opp_report = accept_decision_report(self.model, self.remA, self.remB, s1, s2, t1, t2, perspective=self.opp_team)
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
