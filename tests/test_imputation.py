import pandas as pd
import pytest

from pairing_engine.imputation import impute_dependent_matrices, matrices_from_raw_sheet


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
