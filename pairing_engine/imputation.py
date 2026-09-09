"""
Spreadsheet-format bridging: turns the "Matriz Simple" sheet -- the only
input format this app receives -- into the escudo/espada/descarte matrices
the rest of the engine (zero_sum, formats.n_player) computes with.

Players fill in one general score per matchup plus two independent map
modifiers, all on player-facing scales: the general score is 1-7 (1 =
worst/least, 7 = best/most), and each modifier is a signed -3..+3 delta on
that same scale -- one for when WE pick the map (feeds escudo), one for
when the OPPONENT picks the map (feeds espada). The combined score is
clipped back to the 1-7 range before everything is rescaled onto the
internal 0-20 scale the engine actually uses.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

MIN_SCORE, MAX_SCORE = 0.0, 20.0
# Player-facing "Matriz Simple" input scale.
INPUT_MIN_SCORE, INPUT_MAX_SCORE = 1.0, 7.0
INPUT_MODIFIER_MIN, INPUT_MODIFIER_MAX = -3.0, 3.0
# Map-modifier column suffixes: "si elijo yo" (our map choice) feeds escudo,
# "si elije el rival" (their map choice) feeds espada.
DEPENDENCIA_SUFFIX_PROPIO = " Dependencia Mapa si elijo yo"
DEPENDENCIA_SUFFIX_RIVAL = " Dependencia Mapa si elije el rival"
# A blank general-score cell in the "Matriz Simple" sheet (a player/matchup
# nobody rated yet) defaults to this neutral midpoint of the 1-7 input scale
# -- "assume a coinflip" -- rather than blocking the whole model from loading.
DEFAULT_GENERAL_SCORE_INPUT = 4.0
DEFAULT_GENERAL_SCORE = (DEFAULT_GENERAL_SCORE_INPUT - INPUT_MIN_SCORE) / (INPUT_MAX_SCORE - INPUT_MIN_SCORE) \
    * (MAX_SCORE - MIN_SCORE) + MIN_SCORE


def clip_scores(*dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """Clamp one or more score matrices to the valid [0, 20] range, shape/
    labels untouched. Used to sanitize hand-edited matrices coming back
    from the app's in-app editor.
    """
    return tuple(df.clip(lower=MIN_SCORE, upper=MAX_SCORE) for df in dfs)


def _check_no_duplicate_labels(df: pd.DataFrame) -> None:
    dup_index = df.index[df.index.duplicated()].unique().tolist()
    dup_columns = df.columns[df.columns.duplicated()].unique().tolist()
    if dup_index or dup_columns:
        raise ValueError(
            f"descarte_df has duplicate labels -- rows: {dup_index}, columns: {dup_columns}. "
            f"Every player name must be unique."
        )


def _validate_range(df: pd.DataFrame, min_val: float, max_val: float, label: str) -> None:
    out_of_range = df[(df < min_val) | (df > max_val)]
    if out_of_range.notna().to_numpy().any():
        bad_cells = [(i, j, df.loc[i, j]) for i in df.index for j in df.columns
                     if pd.notna(df.loc[i, j]) and not (min_val <= df.loc[i, j] <= max_val)]
        raise ValueError(
            f"{label} has {len(bad_cells)} value(s) outside the valid [{min_val}, {max_val}] "
            f"range, e.g. {bad_cells[:5]}."
        )


def _rescale_input_block(df: pd.DataFrame, in_min: float, in_max: float,
                          out_min: float, out_max: float, label: str) -> pd.DataFrame:
    """Linearly rescale a raw "Matriz Simple" input block (values expected in
    [in_min, in_max]) onto the internal [out_min, out_max] range the rest of
    the engine computes with. NaN cells pass through unchanged.

    Raises if any non-NaN cell falls outside [in_min, in_max] -- almost
    always a typo, a stray formula, or a cell still on the wrong scale.
    """
    _validate_range(df, in_min, in_max, label)
    return (df - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


def set_index_from_leftover_column(
    raw_df: pd.DataFrame,
    suffix_propio: str = DEPENDENCIA_SUFFIX_PROPIO,
    suffix_rival: str = DEPENDENCIA_SUFFIX_RIVAL,
) -> pd.DataFrame:
    """A "Matriz Simple" sheet has one row per Team1 player, and for each
    Team2 opponent 3 columns: the general/default score, a
    f"{opponent}{suffix_propio}" map modifier (our choice -> escudo), and a
    f"{opponent}{suffix_rival}" map modifier (their choice -> espada). Plus
    one extra column holding the Team1 player's own name/label. This
    detects that label column (whatever's left over once opponent +
    modifier columns are accounted for -- no hardcoded column name needed)
    and sets it as the index, so the result is ready for
    build_matrices_from_simple_format().
    """
    propio_opponents = {c[: -len(suffix_propio)] for c in raw_df.columns if c.endswith(suffix_propio)}
    rival_opponents = {c[: -len(suffix_rival)] for c in raw_df.columns if c.endswith(suffix_rival)}
    general_cols = [c for c in raw_df.columns if not c.endswith(suffix_propio) and not c.endswith(suffix_rival)]
    general_set = set(general_cols)

    paired_opponents = general_set & propio_opponents & rival_opponents
    # Names mentioned via at least one modifier suffix but not fully paired
    # across all 3 columns (general + both modifiers) -- typos/incomplete
    # entries, not candidates for the leftover label column.
    incomplete = (propio_opponents | rival_opponents) - paired_opponents
    if incomplete:
        details = []
        for name in sorted(incomplete):
            missing = []
            if name not in general_set:
                missing.append(f"the general score column {name!r}")
            if name not in propio_opponents:
                missing.append(f"{name + suffix_propio!r}")
            if name not in rival_opponents:
                missing.append(f"{name + suffix_rival!r}")
            details.append(f"{name!r} (missing {', '.join(missing)})")
        raise ValueError(
            f"Incomplete opponent column set for: {'; '.join(details)}. Each opponent needs all 3 "
            f"columns: a general score, '{{name}}{suffix_propio}', and '{{name}}{suffix_rival}'."
        )

    leftover = [c for c in general_cols if c not in paired_opponents]
    if len(leftover) != 1:
        raise ValueError(
            f"Expected exactly 1 leftover label column, found {len(leftover)}: {leftover}. "
            f"If one of these is meant to be an opponent column, make sure it has matching "
            f"'{{name}}{suffix_propio}' and '{{name}}{suffix_rival}' columns."
        )
    return raw_df.set_index(leftover[0])


def build_matrices_from_simple_format(
    matriz_simple_df: pd.DataFrame,
    suffix_propio: str = DEPENDENCIA_SUFFIX_PROPIO,
    suffix_rival: str = DEPENDENCIA_SUFFIX_RIVAL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bridge for the "Matriz Simple" input format: one row per Team1 player
    (as the index), and for each Team2 opponent `p` three columns -- `p`
    (the general score), f"{p}{suffix_propio}" (map modifier when WE choose
    the map, feeds escudo), and f"{p}{suffix_rival}" (map modifier when the
    OPPONENT chooses, feeds espada). All on the player-facing 1-7/±3 scales
    (see INPUT_MIN_SCORE/INPUT_MAX_SCORE, INPUT_MODIFIER_MIN/
    INPUT_MODIFIER_MAX).

    escudo/espada are computed by adding the matching modifier directly to
    the general score ON THE INPUT SCALE, clipping the result back to
    [INPUT_MIN_SCORE, INPUT_MAX_SCORE] (a modifier can never push a score
    outside what a 1-7 rating could represent), THEN rescaling -- so the
    exact same rescale used for the general score applies to all three. A
    blank modifier cell means "no map effect" (0). A matchup left blank
    everywhere defaults to DEFAULT_GENERAL_SCORE for all three matrices.

    Returns (escudo, espada, descarte, was_imputed_df) -- was_imputed_df is
    a same-shape boolean DataFrame, True wherever the matchup was left
    blank in the sheet and defaulted, so you can audit at a glance how much
    of the model rests on the neutral-coinflip default versus real input.
    """
    team2_players = [c for c in matriz_simple_df.columns
                      if not c.endswith(suffix_propio) and not c.endswith(suffix_rival)]
    team1_players = matriz_simple_df.index

    general_input = matriz_simple_df[team2_players].reindex(index=team1_players, columns=team2_players)
    _validate_range(general_input, INPUT_MIN_SCORE, INPUT_MAX_SCORE, "matriz_simple_df's general score columns")

    propio_input = matriz_simple_df[[f"{p}{suffix_propio}" for p in team2_players]].copy()
    propio_input.columns = team2_players
    propio_input = propio_input.reindex(index=team1_players, columns=team2_players)
    _validate_range(propio_input, INPUT_MODIFIER_MIN, INPUT_MODIFIER_MAX,
                     "matriz_simple_df's 'si elijo yo' map-modifier columns")
    propio_input = propio_input.fillna(0.0)

    rival_input = matriz_simple_df[[f"{p}{suffix_rival}" for p in team2_players]].copy()
    rival_input.columns = team2_players
    rival_input = rival_input.reindex(index=team1_players, columns=team2_players)
    _validate_range(rival_input, INPUT_MODIFIER_MIN, INPUT_MODIFIER_MAX,
                     "matriz_simple_df's 'si elije el rival' map-modifier columns")
    rival_input = rival_input.fillna(0.0)

    was_imputed_df = general_input.isna()
    general_input = general_input.fillna(DEFAULT_GENERAL_SCORE_INPUT)

    escudo_input = (general_input + propio_input).clip(lower=INPUT_MIN_SCORE, upper=INPUT_MAX_SCORE)
    espada_input = (general_input + rival_input).clip(lower=INPUT_MIN_SCORE, upper=INPUT_MAX_SCORE)

    descarte = _rescale_input_block(general_input, INPUT_MIN_SCORE, INPUT_MAX_SCORE, MIN_SCORE, MAX_SCORE,
                                     "matriz_simple_df's general score columns")
    escudo = _rescale_input_block(escudo_input, INPUT_MIN_SCORE, INPUT_MAX_SCORE, MIN_SCORE, MAX_SCORE,
                                   "the combined escudo (general + 'si elijo yo') score")
    espada = _rescale_input_block(espada_input, INPUT_MIN_SCORE, INPUT_MAX_SCORE, MIN_SCORE, MAX_SCORE,
                                   "the combined espada (general + 'si elije el rival') score")

    _check_no_duplicate_labels(descarte)

    return escudo, espada, descarte, was_imputed_df


def matrices_from_raw_sheet(
    raw_df: pd.DataFrame,
    suffix_propio: str = DEPENDENCIA_SUFFIX_PROPIO,
    suffix_rival: str = DEPENDENCIA_SUFFIX_RIVAL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One-call convenience: a raw "Matriz Simple" sheet grid (as returned by
    sheets.load_matrix) straight to (escudo, espada, descarte, was_imputed_df).
    """
    matriz_simple = set_index_from_leftover_column(raw_df, suffix_propio, suffix_rival)
    numeric = matriz_simple.apply(pd.to_numeric, errors="coerce")
    return build_matrices_from_simple_format(numeric, suffix_propio, suffix_rival)
