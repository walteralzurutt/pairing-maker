from pathlib import Path

import pandas as pd
import pytest

from pairing_engine.imputation import matrices_from_raw_sheet
from pairing_engine.formats import n_player

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """A static snapshot of the real "Matriz Simple" sheet used throughout this
    project, so tests don't depend on live network access or the live sheet's
    content changing.
    """
    return pd.read_csv(FIXTURES_DIR / "matriz_simple_sample.csv")


@pytest.fixture
def sample_matrices(sample_raw_df):
    """(escudo, espada, descarte, was_imputed) from the real sheet fixture."""
    return matrices_from_raw_sheet(sample_raw_df)


@pytest.fixture
def sample_model(sample_matrices):
    escudo, espada, descarte, _ = sample_matrices
    return n_player.build_model(escudo, espada, descarte)
