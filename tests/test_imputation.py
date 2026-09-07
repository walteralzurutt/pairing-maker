import pandas as pd
import pytest

from pairing_engine.imputation import (
    DEFAULT_GENERAL_SCORE,
    clip_scores,
    impute_dependent_matrices,
    matrices_from_raw_sheet,
)


def _raw_sheet(general, dependency):
    return pd.DataFrame({
        "Player": ["P1"],
        "Ana": [general],
        "Ana Dependencia Mapa": [dependency],
    })


def test_1_to_7_input_scale_rescales_onto_internal_0_to_20_and_0_to_10():
    """The "Matriz Simple" sheet is filled in on the player-facing 1-7 scale
    (1 = worst/least, 7 = best/most) for both the general score and the
    map-dependency column; both must land on the internal 0-20/0-10 scale
    the rest of the engine (escudo/espada/descarte) actually computes with.
    """
    # general=7 (best) -> descarte 20; dependency=7 (matters a lot) -> 10,
    # so escudo (descarte + dependency, clipped) is also 20 and espada
    # (descarte - dependency, clipped) is 10.
    escudo, espada, descarte, _ = matrices_from_raw_sheet(_raw_sheet(general=7.0, dependency=7.0))
    assert descarte.loc["P1", "Ana"] == pytest.approx(20.0)
    assert escudo.loc["P1", "Ana"] == pytest.approx(20.0)
    assert espada.loc["P1", "Ana"] == pytest.approx(10.0)

    # general=1 (worst) -> descarte 0; dependency=1 (barely matters) -> 0,
    # so escudo/espada both collapse to descarte's 0.
    escudo, espada, descarte, _ = matrices_from_raw_sheet(_raw_sheet(general=1.0, dependency=1.0))
    assert descarte.loc["P1", "Ana"] == pytest.approx(0.0)
    assert escudo.loc["P1", "Ana"] == pytest.approx(0.0)
    assert espada.loc["P1", "Ana"] == pytest.approx(0.0)

    # general=4 (midpoint) -> descarte 10; dependency=4 (midpoint) -> 5.
    _, _, descarte, _ = matrices_from_raw_sheet(_raw_sheet(general=4.0, dependency=4.0))
    assert descarte.loc["P1", "Ana"] == pytest.approx(10.0)


def test_general_score_outside_1_to_7_raises():
    with pytest.raises(ValueError, match=r"\[1\.0, 7\.0\]"):
        matrices_from_raw_sheet(_raw_sheet(general=8.0, dependency=4.0))


def test_dependency_outside_1_to_7_raises():
    with pytest.raises(ValueError, match=r"\[1\.0, 7\.0\]"):
        matrices_from_raw_sheet(_raw_sheet(general=4.0, dependency=0.0))


def test_blank_general_score_defaults_to_neutral_coinflip():
    """A player who hasn't rated a matchup yet leaves that cell blank in the
    sheet -- that must default to a neutral score (4 on the 1-7 scale, the
    internal midpoint) rather than blocking the whole model from loading."""
    raw = pd.DataFrame({
        "Player": ["P1"],
        "Ana": [None],
        "Ana Dependencia Mapa": [3.0],
    })
    _, _, descarte, _ = matrices_from_raw_sheet(raw)
    assert descarte.loc["P1", "Ana"] == pytest.approx(DEFAULT_GENERAL_SCORE)


def test_sheet_with_mostly_blank_rows_loads_instead_of_raising():
    """Regression: a sheet where only one Team1 player has rated any
    matchups (the rest of the rows are entirely blank) used to raise
    "descarte_df must be fully filled in"; it must now load, defaulting
    every unrated matchup to a neutral coinflip."""
    raw = pd.DataFrame({
        "Player": ["Necron", "Death Guard", "IK"],
        "Votann": [1.0, None, None],
        "Votann Dependencia Mapa": [2.0, None, None],
        "Aeldari": [4.0, None, None],
        "Aeldari Dependencia Mapa": [5.0, None, None],
    })
    escudo, espada, descarte, _ = matrices_from_raw_sheet(raw)
    assert descarte.shape == (3, 2)
    assert descarte.loc["Death Guard", "Votann"] == pytest.approx(DEFAULT_GENERAL_SCORE)
    assert descarte.loc["IK", "Aeldari"] == pytest.approx(DEFAULT_GENERAL_SCORE)


def test_missing_descarte_cell_error_is_descriptive():
    """impute_dependent_matrices itself (the lower-level function, used
    directly by callers who bypass the "Matriz Simple" bridge and its
    default) still has no fallback for a missing descarte cell -- but the
    error must explain what's missing and where the default lives instead
    of just saying "must be fully filled in"."""
    descarte = pd.DataFrame({"Ana": [10.0, None]}, index=["Juan", "Pedro"])
    with pytest.raises(ValueError) as exc_info:
        impute_dependent_matrices(descarte)
    message = str(exc_info.value)
    assert "Pedro" in message and "Ana" in message
    assert "Matriz Simple" in message
    assert "1 matchup" in message


def test_matrices_from_real_sheet_fixture(sample_raw_df):
    """Regression: parsing the real 'Matriz Simple' sheet fixture produces
    the expected shape and player names."""
    escudo, espada, descarte, was_imputed = matrices_from_raw_sheet(sample_raw_df)
    assert list(descarte.index) == ["Death Guard", "Necron", "Astra", "Marines"]
    assert list(descarte.columns) == ["Marines", "CSM", "IK", "Necron"]
    assert escudo.shape == (4, 4)
    assert espada.shape == (4, 4)


def test_dependency_column_without_general_column_raises():
    """A dependency column with no matching general column (e.g. a sheet
    typo) must raise a clear error, not silently drop that opponent.
    """
    raw = pd.DataFrame({
        "Player": ["P1"],
        "Ana": [10.0],
        "Ana Dependencia Mapa": [2.0],
        "Carla Dependencia Mapa": [3.0],  # no "Carla" general column
    })
    with pytest.raises(ValueError, match="Carla"):
        matrices_from_raw_sheet(raw)


def test_general_column_without_dependency_column_raises_with_correct_name():
    """A general column with no matching dependency column must raise an
    error naming that specific column, not a generic 'couldn't detect the
    label column' message that blames something else.
    """
    raw = pd.DataFrame({
        "Player": ["P1"],
        "Ana": [10.0],
        "Ana Dependencia Mapa": [2.0],
        "Beto": [8.0],  # no "Beto Dependencia Mapa"
    })
    with pytest.raises(ValueError, match="Beto") as exc_info:
        matrices_from_raw_sheet(raw)
    # must actually diagnose the missing dependency column, not just list
    # "Beto" incidentally while blaming the wrong thing (the label column)
    assert "Dependencia Mapa" in str(exc_info.value)


def test_hand_filled_cells_are_clipped_to_valid_range():
    """impute_dependent_matrices documents escudo/espada as clipped to
    [0, 20] -- that must hold for hand-filled cells too, not just the
    auto-imputed ones.
    """
    descarte = pd.DataFrame({"Ana": [10.0]}, index=["Juan"])
    escudo_hand = pd.DataFrame({"Ana": [25.0]}, index=["Juan"])
    escudo, espada, was_imputed = impute_dependent_matrices(descarte, escudo_df=escudo_hand)
    assert escudo.loc["Juan", "Ana"] <= 20.0


def test_duplicate_player_names_raise_clear_error():
    """A duplicated player name should raise a clear, actionable error, not
    the opaque pandas 'truth value of a Series is ambiguous' crash.
    """
    descarte = pd.DataFrame({"Ana": [10.0, 12.0]}, index=["Juan", "Juan"])
    dependency = pd.DataFrame({"Ana": [2.0, 99.0]}, index=["Juan", "Juan"])  # 99 is out of range
    with pytest.raises(ValueError, match="[Dd]uplicat"):
        impute_dependent_matrices(descarte, map_dependency_df=dependency)


def test_clip_scores_clamps_out_of_range_values():
    """Used to sanitize hand-edited matrices coming back from the app's
    in-app editor -- values outside [0, 20] must be clamped, shape/labels
    preserved."""
    df1 = pd.DataFrame({"B": [-5.0, 25.0, 10.0]}, index=["A1", "A2", "A3"])
    df2 = pd.DataFrame({"B": [30.0, -1.0, 5.0]}, index=["A1", "A2", "A3"])

    clipped1, clipped2 = clip_scores(df1, df2)

    assert clipped1["B"].tolist() == [0.0, 20.0, 10.0]
    assert clipped2["B"].tolist() == [20.0, 0.0, 5.0]
    assert list(clipped1.index) == list(df1.index)
