"""
Spreadsheet-format bridging and map-dependency imputation.

Players normally only fill in one "general" score per matchup (descarte_df)
plus how map-dependent that matchup is (0-10). This derives the full
escudo_df ("shield", scores boosted by map dependency) and espada_df
("sword", scores reduced by map dependency) from that, while still trusting
any cells a team filled in by hand instead. None of this assumes any
particular number of players per team -- it operates on whatever
Team1-rows x Team2-columns matrix shape it's given.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_MAP_DEPENDENCY = 0.0
MIN_DEPENDENCY, MAX_DEPENDENCY = 0.0, 10.0
MIN_SCORE, MAX_SCORE = 0.0, 20.0
DEFAULT_DEPENDENCIA_SUFFIX = " Dependencia Mapa"


def _validate_dependency_df(map_dependency_df: pd.DataFrame, descarte_df: pd.DataFrame) -> pd.DataFrame:
    """Reindex onto descarte_df's exact shape (missing rows/cols/cells -> NaN,
    which later becomes the default of 0), and warn about any out-of-range
    values instead of silently misbehaving."""
    aligned = map_dependency_df.reindex(index=descarte_df.index, columns=descarte_df.columns)
    out_of_range = aligned[(aligned < MIN_DEPENDENCY) | (aligned > MAX_DEPENDENCY)]
    if out_of_range.notna().to_numpy().any():
        bad_cells = [(i, j, aligned.loc[i, j]) for i in aligned.index for j in aligned.columns
                     if pd.notna(aligned.loc[i, j]) and not (MIN_DEPENDENCY <= aligned.loc[i, j] <= MAX_DEPENDENCY)]
        raise ValueError(
            f"map_dependency_df has {len(bad_cells)} value(s) outside the valid "
            f"[{MIN_DEPENDENCY}, {MAX_DEPENDENCY}] range, e.g. {bad_cells[:5]}. "
            f"Fix these (or leave them blank/NaN to default to {DEFAULT_MAP_DEPENDENCY})."
        )
    return aligned


def impute_dependent_matrices(
    descarte_df: pd.DataFrame,
    map_dependency_df: Optional[pd.DataFrame] = None,
    escudo_df: Optional[pd.DataFrame] = None,
    espada_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fill in any missing escudo_df/espada_df cells from descarte_df plus a
    per-matchup map-dependency value, leaving any hand-filled cells as-is.

    Parameters
    ----------
    descarte_df : required, fully filled Team1-rows x Team2-columns matrix
        of baseline (no-map-advantage) scores in [0, 20].
    map_dependency_df : optional, same shape/index/columns as descarte_df,
        values in [0, 10]. Any missing matchup (whole matrix is None, or
        individual cells are NaN, or rows/columns are simply absent)
        defaults to 0 -- i.e. "the map has no effect on this matchup",
        which makes the imputed escudo/espada cell identical to descarte.
    escudo_df, espada_df : optional, same shape as descarte_df. Pass None
        to fully derive from descarte_df + map_dependency_df. Pass a
        partially filled DataFrame (NaN for the matchups you don't know)
        to have ONLY those specific cells imputed -- anything you did
        fill in is trusted and left untouched.

    Returns
    -------
    (escudo_df_filled, espada_df_filled, was_imputed_df) -- the first two
    are fully populated (no NaNs), clipped to [0, 20]. `was_imputed_df`
    is a same-shape boolean DataFrame, True wherever EITHER escudo or
    espada was auto-filled for that matchup, so you can audit at a
    glance how much of the model is resting on the map-dependency
    assumption versus real hand-entered data.
    """
    if descarte_df.isna().to_numpy().any():
        raise ValueError("descarte_df must be fully filled in -- it has no fallback/default of its own.")

    if map_dependency_df is None:
        dependency = pd.DataFrame(np.nan, index=descarte_df.index, columns=descarte_df.columns)
    else:
        dependency = _validate_dependency_df(map_dependency_df, descarte_df)
    dependency = dependency.fillna(DEFAULT_MAP_DEPENDENCY)

    if escudo_df is None:
        escudo_df = pd.DataFrame(np.nan, index=descarte_df.index, columns=descarte_df.columns)
    else:
        escudo_df = escudo_df.reindex(index=descarte_df.index, columns=descarte_df.columns)

    if espada_df is None:
        espada_df = pd.DataFrame(np.nan, index=descarte_df.index, columns=descarte_df.columns)
    else:
        espada_df = espada_df.reindex(index=descarte_df.index, columns=descarte_df.columns)

    escudo_missing = escudo_df.isna()
    espada_missing = espada_df.isna()

    escudo_imputed_values = (descarte_df + dependency).clip(lower=MIN_SCORE, upper=MAX_SCORE)
    espada_imputed_values = (descarte_df - dependency).clip(lower=MIN_SCORE, upper=MAX_SCORE)

    escudo_filled = escudo_df.where(~escudo_missing, escudo_imputed_values)
    espada_filled = espada_df.where(~espada_missing, espada_imputed_values)

    was_imputed_df = escudo_missing | espada_missing

    return escudo_filled, espada_filled, was_imputed_df


def set_index_from_leftover_column(raw_df: pd.DataFrame, dependencia_suffix: str = DEFAULT_DEPENDENCIA_SUFFIX) -> pd.DataFrame:
    """A "Matriz Simple" sheet has one row per Team1 player, one column per
    Team2 opponent (the general/default score), one dependency column per
    opponent (named f"{opponent}{dependencia_suffix}"), and one extra column
    holding the Team1 player's own name/label. This detects that label
    column (whatever's left over once opponent + dependency columns are
    accounted for -- no hardcoded column name needed) and sets it as the
    index, so the result is ready for build_matrices_from_simple_format().
    """
    dep_cols = [c for c in raw_df.columns if c.endswith(dependencia_suffix)]
    team2_players = [c[: -len(dependencia_suffix)] for c in dep_cols]
    known_cols = set(team2_players) | set(dep_cols)

    leftover = [c for c in raw_df.columns if c not in known_cols]
    if len(leftover) != 1:
        raise ValueError(
            f"Expected exactly 1 leftover label column, found {len(leftover)}: {leftover}. "
            f"Couldn't auto-detect the Team1 player-name column."
        )
    return raw_df.set_index(leftover[0])


def build_matrices_from_simple_format(
    escudo_df: Optional[pd.DataFrame],
    espada_df: Optional[pd.DataFrame],
    descarte_df: Optional[pd.DataFrame],
    matriz_simple_df: pd.DataFrame,
    dependencia_suffix: str = DEFAULT_DEPENDENCIA_SUFFIX,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bridge for the "Matriz Simple" input format: one row per Team1
    player (as the index), and for each Team2 opponent `p` two columns --
    `p` (the general/default score, equivalent to a descarte_df cell) and
    `f"{p}{dependencia_suffix}"` (that matchup's map-dependency, 0-10).

    Any of escudo_df/espada_df/descarte_df can be None or partially
    filled -- real cells always win; descarte_df falls back to the
    matriz_simple general column, and escudo_df/espada_df fall back to
    descarte (+/-) that matchup's dependency, via impute_dependent_matrices.

    Returns (escudo_filled, espada_filled, descarte_filled, was_imputed_df).
    """
    team2_players = [c for c in matriz_simple_df.columns if not c.endswith(dependencia_suffix)]
    team1_players = matriz_simple_df.index

    general_df = matriz_simple_df[team2_players].reindex(index=team1_players, columns=team2_players)
    dependency_df = matriz_simple_df[[f"{p}{dependencia_suffix}" for p in team2_players]].copy()
    dependency_df.columns = team2_players
    dependency_df = dependency_df.reindex(index=team1_players, columns=team2_players)

    if descarte_df is None:
        descarte_df = pd.DataFrame(np.nan, index=team1_players, columns=team2_players)
    descarte_df = descarte_df.reindex(index=team1_players, columns=team2_players)
    descarte_filled = descarte_df.where(~descarte_df.isna(), general_df)

    escudo_filled, espada_filled, was_imputed_df = impute_dependent_matrices(
        descarte_filled, map_dependency_df=dependency_df, escudo_df=escudo_df, espada_df=espada_df)

    return escudo_filled, espada_filled, descarte_filled, was_imputed_df


def matrices_from_raw_sheet(
    raw_df: pd.DataFrame,
    dependencia_suffix: str = DEFAULT_DEPENDENCIA_SUFFIX,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One-call convenience for the common case: a raw "Matriz Simple"
    sheet grid (as returned by sheets.load_matrix), with no hand-filled
    escudo/espada matrices to merge in. Returns
    (escudo_filled, espada_filled, descarte_filled, was_imputed_df).
    """
    matriz_simple = set_index_from_leftover_column(raw_df, dependencia_suffix)
    numeric = matriz_simple.apply(pd.to_numeric, errors="coerce")
    return build_matrices_from_simple_format(None, None, None, numeric, dependencia_suffix)
