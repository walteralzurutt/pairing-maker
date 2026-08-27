"""
4-player team pairing format: shield / throw / espada / descarte.

This is the format-SPECIFIC half of the algorithm -- the exact sequence of
decisions (each team picks 1 shield; then throws 2 of its remaining 3
players at the opponent's shield; the shield picks 1 of the 2 thrown at
it; the two leftover players from each side play each other) only makes
sense for 4-player teams. It's kept separate from pairing_engine.zero_sum
(the generic LP solver) and pairing_engine.imputation (generic matrix
prep) precisely so that an N-player format can be added later as a
sibling module -- see pairing_engine/formats/__init__.py.

Every recommend_*/decision-report function here returns plain pandas
DataFrames / dataclasses, with no notebook or Streamlit display calls
baked in -- rendering is entirely the UI layer's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..zero_sum import ZeroSumSolution, solve_zero_sum_game
from ..imputation import impute_dependent_matrices

TEAM_SIZE = 4
DEFAULT_TIE_TOLERANCE = 0.75  # points; see _option_report's tie-breaking rule below


# ---------------------------------------------------------------------
# Full-match scoring for one FULLY specified outcome (explicit espada
# picks -- no resolution/argmax happens here, it just adds up the 4
# resulting matches). Used both as a building block for solving the
# espada stage below, and directly once the real, observed picks are
# known.
# ---------------------------------------------------------------------

def score_combination(s1, s2, throw1: Sequence, throw2: Sequence,
                       j_star, i_star,
                       team1_players: Sequence, team2_players: Sequence,
                       escudo_df: pd.DataFrame, espada_df: pd.DataFrame,
                       descarte_df: pd.DataFrame):
    """Given shields (s1, s2), throw sets (throw1 from Team1, throw2 from
    Team2), and the two shields' ACTUAL espada picks (j_star = the Team2
    player Team1's shield chose to fight; i_star = the Team1 player
    Team2's shield chose to fight), return the total Team1 score plus the
    4 individual matchups. See solve_espada_stage() if you want the
    game-theoretically optimal picks computed for you instead of
    supplying them directly.
    """
    j_left = [j for j in throw2 if j != j_star][0]
    i_left = [i for i in throw1 if i != i_star][0]

    d1 = [p for p in team1_players if p != s1 and p not in throw1][0]
    d2 = [p for p in team2_players if p != s2 and p not in throw2][0]

    matches = [
        {"team1_player": s1, "team2_player": j_star, "matrix": "escudo",
         "team1_score": escudo_df.loc[s1, j_star]},
        {"team1_player": i_star, "team2_player": s2, "matrix": "espada",
         "team1_score": espada_df.loc[i_star, s2]},
        {"team1_player": i_left, "team2_player": j_left, "matrix": "descarte",
         "team1_score": descarte_df.loc[i_left, j_left]},
        {"team1_player": d1, "team2_player": d2, "matrix": "descarte",
         "team1_score": descarte_df.loc[d1, d2]},
    ]
    total_team1_score = sum(m["team1_score"] for m in matches)
    return total_team1_score, matches


# ---------------------------------------------------------------------
# Stage 3 (espada choice), modeled as a 2x2 simultaneous game. Team1's
# shield picks which of throw2's 2 candidates to fight; Team2's shield
# picks which of throw1's 2 candidates to fight. Each pick also
# determines -- via whoever gets rejected -- who plays in the resulting
# descarte-vs-descarte match, so the two shields' choices are coupled
# through that shared leftover match. Solved exactly like every other
# stage: build the payoff matrix, hand it to solve_zero_sum_game().
# ---------------------------------------------------------------------

def solve_espada_stage(s1, s2, throw1: Sequence, throw2: Sequence,
                        team1_players: Sequence, team2_players: Sequence,
                        escudo_df: pd.DataFrame, espada_df: pd.DataFrame,
                        descarte_df: pd.DataFrame):
    row_options = list(throw2)   # Team1 shield (s1) picks among these
    col_options = list(throw1)   # Team2 shield (s2) picks among these

    payoff = np.zeros((len(row_options), len(col_options)))
    breakdown = {}
    for a, j_star in enumerate(row_options):
        for b, i_star in enumerate(col_options):
            total, matches = score_combination(
                s1, s2, throw1, throw2, j_star, i_star,
                team1_players, team2_players, escudo_df, espada_df, descarte_df)
            payoff[a, b] = total
            breakdown[(j_star, i_star)] = (total, matches)

    solution = solve_zero_sum_game(payoff)
    return {
        "row_options": row_options,
        "col_options": col_options,
        "payoff": payoff,
        "breakdown": breakdown,
        "solution": solution,
    }


# ---------------------------------------------------------------------
# Stage 2 (throw choice): build the 3x3 payoff matrix for fixed shields
# and solve it as a zero-sum game. Each cell's value comes from solving
# the stage-3 espada subgame, not a single deterministic score.
# ---------------------------------------------------------------------

def throw_options(remaining_players: Sequence, n_thrown: int = 2):
    """All ways to choose which `n_thrown` of the remaining players get
    thrown at the opponent's shield. For 4-player teams remaining=3,
    n_thrown=2 -> 3 options."""
    return list(combinations(remaining_players, n_thrown))


def solve_throw_stage(s1, s2, team1_players, team2_players,
                       escudo_df, espada_df, descarte_df):
    remaining1 = [p for p in team1_players if p != s1]
    remaining2 = [p for p in team2_players if p != s2]
    throws1 = throw_options(remaining1)   # 3 options
    throws2 = throw_options(remaining2)   # 3 options

    payoff = np.zeros((len(throws1), len(throws2)))
    breakdown = {}
    for a, t1 in enumerate(throws1):
        for b, t2 in enumerate(throws2):
            espada_details = solve_espada_stage(
                s1, s2, t1, t2, team1_players, team2_players,
                escudo_df, espada_df, descarte_df)
            payoff[a, b] = espada_details["solution"].value
            breakdown[(t1, t2)] = espada_details

    solution = solve_zero_sum_game(payoff)
    return {
        "throws1": throws1,
        "throws2": throws2,
        "payoff": payoff,
        "breakdown": breakdown,
        "solution": solution,
    }


# ---------------------------------------------------------------------
# Stage 1 (shield choice): build the NxN matrix (value of the nested
# throw-stage zero-sum game) and solve it as a zero-sum game.
# ---------------------------------------------------------------------

@dataclass
class MatchModelResult:
    team1_players: list
    team2_players: list
    shield_payoff: np.ndarray                  # n x n matrix of throw-stage game values
    shield_solution: ZeroSumSolution
    throw_stage_details: dict                  # {(s1, s2): solve_throw_stage(...) output}
    expected_team1_score: float
    expected_team2_score: float

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.shield_payoff,
                           index=self.team1_players, columns=self.team2_players)
        return df.round(2)

    def optimal_shield_strategy(self):
        p1 = pd.Series(self.shield_solution.row_strategy, index=self.team1_players, name="P(chosen as Team1 shield)")
        p2 = pd.Series(self.shield_solution.col_strategy, index=self.team2_players, name="P(chosen as Team2 shield)")
        return p1.round(4), p2.round(4)


def build_model(escudo_df: pd.DataFrame, espada_df: pd.DataFrame,
                 descarte_df: pd.DataFrame) -> MatchModelResult:
    team1_players = list(escudo_df.index)
    team2_players = list(escudo_df.columns)
    n1, n2 = len(team1_players), len(team2_players)
    if n1 != TEAM_SIZE or n2 != TEAM_SIZE:
        raise ValueError(
            f"formats.four_player requires exactly {TEAM_SIZE} players per team, "
            f"got {n1} (team1) and {n2} (team2). Use a different pairing_engine.formats "
            f"module for other team sizes."
        )

    shield_payoff = np.zeros((n1, n2))
    throw_stage_details = {}
    for a, s1 in enumerate(team1_players):
        for b, s2 in enumerate(team2_players):
            details = solve_throw_stage(s1, s2, team1_players, team2_players,
                                         escudo_df, espada_df, descarte_df)
            shield_payoff[a, b] = details["solution"].value
            throw_stage_details[(s1, s2)] = details

    shield_solution = solve_zero_sum_game(shield_payoff)

    return MatchModelResult(
        team1_players=team1_players,
        team2_players=team2_players,
        shield_payoff=shield_payoff,
        shield_solution=shield_solution,
        throw_stage_details=throw_stage_details,
        expected_team1_score=shield_solution.value,
        expected_team2_score=80.0 - shield_solution.value,
    )


def build_model_from_partial_inputs(
    descarte_df: pd.DataFrame,
    map_dependency_df: Optional[pd.DataFrame] = None,
    escudo_df: Optional[pd.DataFrame] = None,
    espada_df: Optional[pd.DataFrame] = None,
):
    """Convenience one-liner: impute whatever escudo/espada cells are
    missing from descarte_df + map_dependency_df, then build the full
    model. Returns (model, escudo_df_filled, espada_df_filled,
    was_imputed_df) -- the filled matrices and the imputation mask are
    returned too so the UI can display/audit them alongside the results.
    """
    escudo_filled, espada_filled, was_imputed_df = impute_dependent_matrices(
        descarte_df, map_dependency_df=map_dependency_df,
        escudo_df=escudo_df, espada_df=espada_df)
    model = build_model(escudo_filled, espada_filled, descarte_df)
    return model, escudo_filled, espada_filled, was_imputed_df


# ---------------------------------------------------------------------
# Full enumeration (every possible combination) -- useful for auditing /
# sanity-checking the equilibrium, not part of the live recommendation
# flow.
# ---------------------------------------------------------------------

def enumerate_all_combinations(escudo_df, espada_df, descarte_df) -> pd.DataFrame:
    """Brute-force every (shield1, shield2, throw1, throw2) combination. The
    reported score is the EQUILIBRIUM value of the nested espada-stage
    subgame for that combination (both shields playing their optimal 2x2
    mix), not a single deterministic outcome -- see solve_espada_stage()
    for the full breakdown of any specific combination.
    """
    team1_players = list(escudo_df.index)
    team2_players = list(escudo_df.columns)
    rows = []
    for s1 in team1_players:
        for s2 in team2_players:
            remaining1 = [p for p in team1_players if p != s1]
            remaining2 = [p for p in team2_players if p != s2]
            for t1 in throw_options(remaining1):
                for t2 in throw_options(remaining2):
                    espada_details = solve_espada_stage(
                        s1, s2, t1, t2, team1_players, team2_players,
                        escudo_df, espada_df, descarte_df)
                    total = espada_details["solution"].value
                    rows.append({
                        "shield1": s1, "shield2": s2,
                        "throw1": t1, "throw2": t2,
                        "team1_expected_score": total,
                        "team2_expected_score": 80 - total,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Decision-tree style insights: how much is each option at each decision
# node actually worth, examined independently of the rest of the tree.
# ---------------------------------------------------------------------

def _option_report(payoff_for_me: np.ndarray, opp_equilibrium_strategy: np.ndarray,
                    my_equilibrium_strategy: np.ndarray, option_labels: Sequence,
                    tie_tolerance: float = DEFAULT_TIE_TOLERANCE) -> pd.DataFrame:
    """Given a payoff matrix already oriented so rows = my options, columns =
    opponent options, and values = MY score (higher is better for me),
    report the expected outcome for each option assuming the opponent
    sticks to their game-theoretically optimal mix, regardless of what I
    pick (`vs_equilibrium_opponent`). This is the most defensible
    "expected value" number -- it equals the overall game value for
    whichever option(s) are actually part of my own equilibrium mix.

    `worst_case`/`best_case` are the SPREAD around that average: the real
    opponent doesn't actually play a probabilistic blend in any one match,
    they commit to one specific option, so the real result of picking this
    option could land anywhere from `worst_case` (if they happen to pick
    their best counter to you) to `best_case` (if they happen to pick
    whatever helps you most) -- `vs_equilibrium_opponent` is the average
    across those outcomes if this exact situation repeated many times
    against an opponent playing their equilibrium mix, not a guarantee
    about this one match. `spread` = best_case - worst_case, a quick sense
    of how much a single match could bounce around that average.

    `my_equilibrium_strategy` is included so you can see which options the
    solver actually recommends mixing between (weight > 0) versus options
    that are strictly worse in equilibrium (weight == 0).

    Tie-breaking: with real data, two options are rarely EXACTLY equal on
    `vs_equilibrium_opponent` -- they're often off by a fraction of a
    point for reasons that don't really matter (rounding, near-symmetric
    matchups). Ranking purely by that value would recommend whichever one
    happens to be a hair ahead, ignoring that it might also be far riskier.
    So: any option within `tie_tolerance` points of the best value is
    treated as "effectively tied" with it, and among that near-best group,
    the recommendation (row 0) is whichever has the LOWEST spread rather
    than the highest raw value. Options clearly outside that tolerance are
    still ranked purely by `vs_equilibrium_opponent`, below the near-best
    group.
    """
    vs_equilibrium_opp = payoff_for_me @ opp_equilibrium_strategy
    worst_case = payoff_for_me.min(axis=1)
    best_case = payoff_for_me.max(axis=1)

    df = pd.DataFrame({
        "option": list(option_labels),
        "vs_equilibrium_opponent": vs_equilibrium_opp,
        "worst_case": worst_case,
        "best_case": best_case,
        "spread": best_case - worst_case,
        "equilibrium_weight": np.round(my_equilibrium_strategy, 4),
    })
    df["gap_to_best"] = (df["vs_equilibrium_opponent"] - df["vs_equilibrium_opponent"].max()).round(2)

    best_value = df["vs_equilibrium_opponent"].max()
    near_best_mask = (best_value - df["vs_equilibrium_opponent"]) <= tie_tolerance
    near_best = df[near_best_mask].sort_values("spread", ascending=True)
    rest = df[~near_best_mask].sort_values("vs_equilibrium_opponent", ascending=False)
    return pd.concat([near_best, rest]).reset_index(drop=True).round(2)


def shield_decision_report(model: MatchModelResult, perspective: str = "team1",
                            tie_tolerance: float = DEFAULT_TIE_TOLERANCE) -> pd.DataFrame:
    """Stage-1 (shield choice) insight table, examined independently of the
    throw stage underneath it. `perspective` is "team1" or "team2".
    """
    if perspective == "team1":
        payoff_for_me = model.shield_payoff                # rows=team1, cols=team2, Team1 score
        opp_strategy = model.shield_solution.col_strategy   # Team2's equilibrium shield mix
        my_strategy = model.shield_solution.row_strategy
        option_labels = model.team1_players
    elif perspective == "team2":
        payoff_for_me = (80.0 - model.shield_payoff).T       # rows=team2, cols=team1, Team2 score
        opp_strategy = model.shield_solution.row_strategy    # Team1's equilibrium shield mix
        my_strategy = model.shield_solution.col_strategy
        option_labels = model.team2_players
    else:
        raise ValueError("perspective must be 'team1' or 'team2'")

    return _option_report(payoff_for_me, opp_strategy, my_strategy, option_labels, tie_tolerance)


def throw_decision_report(model: MatchModelResult, s1, s2, perspective: str = "team1",
                           tie_tolerance: float = DEFAULT_TIE_TOLERANCE) -> pd.DataFrame:
    """Stage-2 (throw choice) insight table for one specific, already-known
    pair of shields (s1 from Team1, s2 from Team2) -- by the time throws
    are chosen, shields are public, so this is conditioned on a real pair
    rather than averaged over Team2's shield mix.
    """
    details = model.throw_stage_details[(s1, s2)]
    payoff = details["payoff"]           # rows=team1 throws, cols=team2 throws, Team1 score
    solution = details["solution"]

    if perspective == "team1":
        payoff_for_me = payoff
        opp_strategy = solution.col_strategy
        my_strategy = solution.row_strategy
        option_labels = details["throws1"]
    elif perspective == "team2":
        payoff_for_me = (80.0 - payoff).T
        opp_strategy = solution.row_strategy
        my_strategy = solution.col_strategy
        option_labels = details["throws2"]
    else:
        raise ValueError("perspective must be 'team1' or 'team2'")

    return _option_report(payoff_for_me, opp_strategy, my_strategy, option_labels, tie_tolerance)


def espada_decision_report(s1, s2, throw1: Sequence, throw2: Sequence,
                            team1_players: Sequence, team2_players: Sequence,
                            escudo_df: pd.DataFrame, espada_df: pd.DataFrame,
                            descarte_df: pd.DataFrame, perspective: str = "team1",
                            tie_tolerance: float = DEFAULT_TIE_TOLERANCE) -> pd.DataFrame:
    """Stage-3 (espada choice) insight table for one specific, already-known
    combination of shields AND throws. A shield's pick also determines who
    ends up in the resulting descarte-vs-descarte match, so this is solved
    as a small 2x2 game (see solve_espada_stage) rather than a simple
    best-response.
    """
    details = solve_espada_stage(s1, s2, throw1, throw2, team1_players, team2_players,
                                  escudo_df, espada_df, descarte_df)
    payoff = details["payoff"]     # rows = team1 shield's options (from throw2), cols = team2 shield's options (from throw1)
    solution = details["solution"]

    if perspective == "team1":
        payoff_for_me = payoff
        opp_strategy = solution.col_strategy
        my_strategy = solution.row_strategy
        option_labels = details["row_options"]
    elif perspective == "team2":
        payoff_for_me = (80.0 - payoff).T
        opp_strategy = solution.row_strategy
        my_strategy = solution.col_strategy
        option_labels = details["col_options"]
    else:
        raise ValueError("perspective must be 'team1' or 'team2'")

    return _option_report(payoff_for_me, opp_strategy, my_strategy, option_labels, tie_tolerance)


# Spanish display labels for the decision-report columns -- shared by
# insight_sentences() below and by the UI layer when rendering a report
# table, so the two stay consistent.
COLUMN_LABELS_ES = {
    "option": "Opción",
    "vs_equilibrium_opponent": "Valor esperado",
    "worst_case": "Peor caso",
    "best_case": "Mejor caso",
    "spread": "Volatilidad",
    "equilibrium_weight": "Peso de equilibrio",
    "gap_to_best": "Diferencia con la mejor",
}


def insight_sentences(report: pd.DataFrame, metric: str = "vs_equilibrium_opponent",
                       option_col: str = "option") -> list:
    """Turn a decision report into plain-language (Spanish) comparison
    sentences, e.g. 'Elegir A1 en vez de A3 vale +10.40 puntos en Valor
    esperado.' Always compares every option to the top-ranked
    (recommended) one.

    Because the top row can be picked for having a lower `spread` rather
    than the strictly highest `vs_equilibrium_opponent` (see the
    tie-breaking rule in _option_report), the gap can occasionally come
    out negative -- that just means the recommended pick is marginally
    lower on this metric but was preferred for being less volatile.
    """
    metric_label = COLUMN_LABELS_ES.get(metric, metric)
    best_label = report.iloc[0][option_col]
    best_value = report.iloc[0][metric]
    sentences = []
    for _, row in report.iterrows():
        if row[option_col] == best_label:
            continue
        gap = best_value - row[metric]
        sentences.append(
            f"Elegir {best_label} en vez de {row[option_col]} vale "
            f"{gap:+.2f} puntos en {metric_label}."
        )
    return sentences


SUMMARY_COLUMN_LABELS_ES = {
    "matchup_type": "Tipo de partida",
    "you": "Tú",
    "opponent": "Rival",
    "your_score": "Tu puntaje",
}


def summarize_matches(matches: list, my_team: str, my_total: float, opp_total: float) -> pd.DataFrame:
    """Reshape the 4 resolved matchups (as returned by score_combination /
    LivePairingSession.finalize) into a plain, presentation-agnostic
    DataFrame of (matchup type, you, opponent, your score) -- rendering it
    as a colored/styled table is the UI layer's job.
    """
    rows = []
    for m in matches:
        t1_name, t2_name = m["team1_player"], m["team2_player"]
        mine, theirs = (t1_name, t2_name) if my_team == "team1" else (t2_name, t1_name)
        my_score = m["team1_score"] if my_team == "team1" else 20 - m["team1_score"]
        rows.append({"matchup_type": m["matrix"], "you": mine, "opponent": theirs,
                      "your_score": my_score})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Live pairing session: stateful walk-through of one real match's
# pairing process, from your team's point of view. Every recommend_*
# method returns (my_report, opp_report) DataFrames -- no printing/
# display side effects -- so the UI layer decides how to render them.
# ---------------------------------------------------------------------

class LivePairingSession:
    """`my_team` is "team1" or "team2" depending on which side of the
    escudo/espada/descarte matrices your team is."""

    def __init__(self, model: MatchModelResult, escudo_df: pd.DataFrame,
                 espada_df: pd.DataFrame, descarte_df: pd.DataFrame,
                 my_team: str = "team1"):
        if my_team not in ("team1", "team2"):
            raise ValueError("my_team must be 'team1' or 'team2'")
        self.model = model
        self.escudo_df, self.espada_df, self.descarte_df = escudo_df, espada_df, descarte_df
        self.my_team = my_team
        self.opp_team = "team2" if my_team == "team1" else "team1"
        self.my_players = model.team1_players if my_team == "team1" else model.team2_players
        self.opp_players = model.team2_players if my_team == "team1" else model.team1_players

        self.my_shield: Optional[str] = None
        self.opp_shield: Optional[str] = None
        self.my_throw: Optional[Tuple[str, str]] = None
        self.opp_throw: Optional[Tuple[str, str]] = None
        self.shield_reports: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None
        self.throw_reports: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None
        self.espada_reports: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None
        self.result: Optional[dict] = None

    # ---- orientation helpers ----------------------------------------
    def _to_team1_team2(self, mine, theirs):
        """Reorder (mine, theirs) into (team1_value, team2_value)."""
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

    # ---- Stage 1: shield ----------------------------------------------
    def recommend_shield(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Call BEFORE you commit to a shield -- the opponent's shield is
        not knowable yet (simultaneous/hidden), so this ranks your options
        against their equilibrium shield mix, and also shows the mirror
        image: what the opponent's own best shield is expected to be.
        Returns (my_report, opp_report); row 0 of each is the top pick.
        """
        my_report = shield_decision_report(self.model, perspective=self.my_team)
        opp_report = shield_decision_report(self.model, perspective=self.opp_team)
        self.shield_reports = (my_report, opp_report)
        return my_report, opp_report

    def lock_shields(self, my_shield: str, opp_shield: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Call once both shields are revealed face-up. Immediately runs the
        stage-2 (throw) recommendation for you, since that's the next real
        decision point."""
        self._validate_player(my_shield, self.my_players, "my_shield")
        self._validate_player(opp_shield, self.opp_players, "opp_shield")
        self.my_shield, self.opp_shield = my_shield, opp_shield
        return self.recommend_throw()

    # ---- Stage 2: throw -------------------------------------------------
    def recommend_throw(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Call once both shields are known: which 2 of your remaining
        players should you throw at the opponent's shield? Also shows what
        the opponent's own best throw is expected to be, now that shields
        are public for both sides.
        """
        if self.my_shield is None or self.opp_shield is None:
            raise RuntimeError("Call lock_shields(...) first.")
        s1, s2 = self._to_team1_team2(self.my_shield, self.opp_shield)

        my_report = throw_decision_report(self.model, s1, s2, perspective=self.my_team)
        opp_report = throw_decision_report(self.model, s1, s2, perspective=self.opp_team)
        self.throw_reports = (my_report, opp_report)
        return my_report, opp_report

    def lock_throws(self, my_throw: Sequence[str], opp_throw: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Call once both teams' throws are revealed face-up. Immediately
        runs the stage-3 (espada) recommendation, since that's next."""
        if self.my_shield is None:
            raise RuntimeError("Call lock_shields(...) first.")
        remaining_mine = [p for p in self.my_players if p != self.my_shield]
        remaining_theirs = [p for p in self.opp_players if p != self.opp_shield]
        my_throw, opp_throw = tuple(my_throw), tuple(opp_throw)
        self._validate_pair(my_throw, remaining_mine, "my_throw")
        self._validate_pair(opp_throw, remaining_theirs, "opp_throw")
        self.my_throw, self.opp_throw = my_throw, opp_throw
        return self.recommend_espada()

    # ---- Stage 3: espada --------------------------------------------------
    def recommend_espada(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Call once both throws are known: which of the 2 opponents thrown
        at YOUR shield should your shield fight? This accounts for the
        resulting descarte-vs-descarte match too (whichever candidate you
        reject also determines who plays who in that match), so it's
        solved as a small 2x2 game rather than just picking your best
        individual duel -- occasionally that means the equilibrium
        genuinely mixes between your two options, same as the throw stage
        can. Also predicts (but cannot guarantee) the opponent's pick."""
        if self.my_throw is None or self.opp_throw is None:
            raise RuntimeError("Call lock_throws(...) first.")
        s1, s2 = self._to_team1_team2(self.my_shield, self.opp_shield)
        t1, t2 = self._to_team1_team2(self.my_throw, self.opp_throw)

        my_report = espada_decision_report(
            s1, s2, t1, t2, self.model.team1_players, self.model.team2_players,
            self.escudo_df, self.espada_df, self.descarte_df, perspective=self.my_team)
        opp_report = espada_decision_report(
            s1, s2, t1, t2, self.model.team1_players, self.model.team2_players,
            self.escudo_df, self.espada_df, self.descarte_df, perspective=self.opp_team)
        self.espada_reports = (my_report, opp_report)
        return my_report, opp_report

    # ---- Resolution -------------------------------------------------------
    def finalize(self, my_espada_pick: Optional[str] = None,
                 opp_espada_pick: Optional[str] = None) -> dict:
        """Call once you know the ACTUAL espada picks (your own choice, and
        the opponent's real revealed choice). Leave a value as None to fall
        back to the model's top recommendation/prediction for it. Returns
        {"matches": [...], "my_total": float, "opp_total": float}.
        """
        if self.espada_reports is None:
            self.recommend_espada()
        my_report, opp_report = self.espada_reports
        s1, s2 = self._to_team1_team2(self.my_shield, self.opp_shield)
        t1, t2 = self._to_team1_team2(self.my_throw, self.opp_throw)

        my_pick = my_espada_pick or my_report.iloc[0]["option"]
        opp_pick = opp_espada_pick or opp_report.iloc[0]["option"]
        j_star, i_star = (my_pick, opp_pick) if self.my_team == "team1" else (opp_pick, my_pick)

        total_team1, matches = score_combination(
            s1, s2, t1, t2, j_star, i_star,
            self.model.team1_players, self.model.team2_players,
            self.escudo_df, self.espada_df, self.descarte_df)

        my_total = total_team1 if self.my_team == "team1" else 80.0 - total_team1
        opp_total = 80.0 - my_total

        self.result = {"matches": matches, "my_total": my_total, "opp_total": opp_total}
        return self.result
