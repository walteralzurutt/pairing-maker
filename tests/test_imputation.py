import pandas as pd
import pytest

from pairing_engine.imputation import (
    DEFAULT_GENERAL_SCORE,
    clip_scores,
    matrices_from_raw_sheet,
)


def _raw_sheet(general, propio, rival):
    return pd.DataFrame({
        "Player": ["P1"],
        "Ana": [general],
        "Ana Dependencia Mapa si elijo yo": [propio],
        "Ana Dependencia Mapa si elije el rival": [rival],
    })


def test_escudo_and_espada_apply_independent_modifiers_on_input_scale():
    """general=4 (descarte mid, 10.0); propio=+3 -> escudo input 7 (max) ->
    20.0; rival=-2 -> espada input 2 -> (2-1)/6*20 = 3.33 -- escudo and
    espada must diverge independently from the same base score, using
    different modifiers.
    """
    escudo, espada, descarte, _ = matrices_from_raw_sheet(_raw_sheet(general=4.0, propio=3.0, rival=-2.0))
    assert descarte.loc["P1", "Ana"] == pytest.approx(10.0)
    assert escudo.loc["P1", "Ana"] == pytest.approx(20.0)
    assert espada.loc["P1", "Ana"] == pytest.approx(10.0 / 3.0)


def test_modifier_clips_combined_score_to_1_7_bounds_instead_of_raising():
    """A modifier that would push the combined score outside the 1-7 range
    must clip back to the bound, not raise -- a score of 7 (max) + a +3
    modifier is still just "the best possible", not an error.
    """
    escudo, _, _, _ = matrices_from_raw_sheet(_raw_sheet(general=7.0, propio=3.0, rival=0.0))
    assert escudo.loc["P1", "Ana"] == pytest.approx(20.0)

    _, espada, _, _ = matrices_from_raw_sheet(_raw_sheet(general=1.0, propio=0.0, rival=-3.0))
    assert espada.loc["P1", "Ana"] == pytest.approx(0.0)


def test_general_score_outside_1_to_7_raises():
    with pytest.raises(ValueError, match=r"\[1\.0, 7\.0\]"):
        matrices_from_raw_sheet(_raw_sheet(general=8.0, propio=0.0, rival=0.0))


def test_modifier_outside_range_raises():
    with pytest.raises(ValueError, match=r"\[-3\.0, 3\.0\]"):
        matrices_from_raw_sheet(_raw_sheet(general=4.0, propio=5.0, rival=0.0))
    with pytest.raises(ValueError, match=r"\[-3\.0, 3\.0\]"):
        matrices_from_raw_sheet(_raw_sheet(general=4.0, propio=0.0, rival=-4.0))


def test_blank_general_score_defaults_to_neutral_coinflip():
    """A player who hasn't rated a matchup yet leaves the general-score cell
    blank -- that must default to a neutral score (4 on the 1-7 scale, the
    internal midpoint) for descarte, escudo, and espada alike, rather than
    blocking the whole model from loading.
    """
    raw = pd.DataFrame({
        "Player": ["P1"],
        "Ana": [None],
        "Ana Dependencia Mapa si elijo yo": [1.0],
        "Ana Dependencia Mapa si elije el rival": [1.0],
    })
    escudo, espada, descarte, was_imputed = matrices_from_raw_sheet(raw)
    assert descarte.loc["P1", "Ana"] == pytest.approx(DEFAULT_GENERAL_SCORE)
    assert was_imputed.loc["P1", "Ana"]
    # the modifier still applies on top of the defaulted 4.0 base
    assert escudo.loc["P1", "Ana"] == pytest.approx((5.0 - 1.0) / 6.0 * 20.0)


def test_sheet_with_mostly_blank_rows_loads_instead_of_raising():
    """Regression: a sheet where only one Team1 player has rated any
    matchups (the rest of the rows are entirely blank) must load, defaulting
    every unrated matchup to a neutral coinflip."""
    raw = pd.DataFrame({
        "Player": ["Necron", "Death Guard", "IK"],
        "Votann": [1.0, None, None],
        "Votann Dependencia Mapa si elijo yo": [2.0, None, None],
        "Votann Dependencia Mapa si elije el rival": [1.0, None, None],
        "Aeldari": [4.0, None, None],
        "Aeldari Dependencia Mapa si elijo yo": [1.0, None, None],
        "Aeldari Dependencia Mapa si elije el rival": [2.0, None, None],
    })
    escudo, espada, descarte, _ = matrices_from_raw_sheet(raw)
    assert descarte.shape == (3, 2)
    assert descarte.loc["Death Guard", "Votann"] == pytest.approx(DEFAULT_GENERAL_SCORE)
    assert descarte.loc["IK", "Aeldari"] == pytest.approx(DEFAULT_GENERAL_SCORE)


def test_matrices_from_real_sheet_fixture(sample_raw_df):
    """Regression: parsing the real 'Matriz Simple' sheet fixture produces
    the expected shape and player names."""
    escudo, espada, descarte, was_imputed = matrices_from_raw_sheet(sample_raw_df)
    assert list(descarte.index) == ["Death Guard", "Necron", "Astra", "Marines"]
    assert list(descarte.columns) == ["Marines", "CSM", "IK", "Necron"]
    assert escudo.shape == (4, 4)
    assert espada.shape == (4, 4)


def test_missing_one_modifier_suffix_raises_with_correct_name():
    """An opponent with a general column and only ONE of the two modifier
    columns (a sheet typo) must raise an error naming that opponent and the
    specific missing column, not a generic 'couldn't detect the label
    column' message.
    """
    raw = pd.DataFrame({
        "Player": ["P1"],
        "Ana": [4.0],
        "Ana Dependencia Mapa si elijo yo": [1.0],
        # missing "Ana Dependencia Mapa si elije el rival"
    })
    with pytest.raises(ValueError, match="Ana") as exc_info:
        matrices_from_raw_sheet(raw)
    assert "si elije el rival" in str(exc_info.value)


def test_modifier_columns_without_general_column_raises():
    """Both modifier columns present but no matching general column (e.g. a
    typo in the general column's name) must raise a clear error, not
    silently drop that opponent.
    """
    raw = pd.DataFrame({
        "Player": ["P1"],
        "Ana": [4.0],
        "Ana Dependencia Mapa si elijo yo": [1.0],
        "Ana Dependencia Mapa si elije el rival": [1.0],
        "Carla Dependencia Mapa si elijo yo": [2.0],
        "Carla Dependencia Mapa si elije el rival": [1.0],
    })
    with pytest.raises(ValueError, match="Carla"):
        matrices_from_raw_sheet(raw)


def test_blank_label_column_header_resolves_correctly():
    """The real sheet's label column has a blank header -- confirm that
    still gets detected as the leftover/index column rather than being
    mistaken for an incomplete opponent."""
    raw = pd.DataFrame({
        "": ["P1", "P2"],
        "Ana": [4.0, 5.0],
        "Ana Dependencia Mapa si elijo yo": [1.0, 0.0],
        "Ana Dependencia Mapa si elije el rival": [0.0, 1.0],
    })
    _, _, descarte, _ = matrices_from_raw_sheet(raw)
    assert list(descarte.index) == ["P1", "P2"]


def test_duplicate_player_names_raise_clear_error():
    """A duplicated player name should raise a clear, actionable error, not
    an opaque crash.
    """
    raw = pd.DataFrame({
        "Player": ["Juan", "Juan"],
        "Ana": [4.0, 5.0],
        "Ana Dependencia Mapa si elijo yo": [1.0, 0.0],
        "Ana Dependencia Mapa si elije el rival": [0.0, 1.0],
    })
    with pytest.raises(ValueError, match="[Dd]uplicat"):
        matrices_from_raw_sheet(raw)


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
