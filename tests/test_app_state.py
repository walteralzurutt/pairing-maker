from unittest.mock import patch

import pandas as pd
import pytest

from streamlit.testing.v1 import AppTest

from pairing_engine import sheets as sheets_module


def _load_app(sample_raw_df):
    """Boot the app with the network call to Google Sheets patched out, and
    click through 'Cargar / actualizar predicciones' so a session exists.
    """
    with patch.object(sheets_module, "load_matrix_from_url", lambda url: sample_raw_df):
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        at.button[0].click().run()
    return at


def test_known_opponent_shield_survives_stage_advance(sample_raw_df):
    """Declaring a known opponent shield before round 1 must stick -- the
    still-visible Etapa 1 panel shouldn't silently recompute itself with
    known_opponent_shield=None once the round advances past the shield stage.
    """
    at = _load_app(sample_raw_df)
    with patch.object(sheets_module, "load_matrix_from_url", lambda url: sample_raw_df):
        at.selectbox(key="known_opponent_shield_choice").select("IK").run()
        session = at.session_state["session"]
        assert session.shield_reports[1].iloc[0]["option"] == "IK"

        fijar_escudos = next(b for b in at.button if b.label == "Fijar escudos ->")
        fijar_escudos.click().run()

        assert not at.exception
        assert at.session_state["stage"] == "swords"
        session_after = at.session_state["session"]
        assert session_after.shield_reports[1].iloc[0]["option"] == "IK", (
            "opp_report reverted to the unconditioned equilibrium prediction "
            "after the round advanced past the shield stage"
        )


def test_new_match_button_keeps_loaded_predictions(sample_raw_df):
    """'Empezar una nueva partida' should only reset the match in progress,
    not force a full spreadsheet reload."""
    at = _load_app(sample_raw_df)
    with patch.object(sheets_module, "load_matrix_from_url", lambda url: sample_raw_df):
        assert "model" in at.session_state

        # Drive the match to completion via the session's own API (already covered
        # by n_player-level tests) rather than re-driving every widget interaction
        # here -- this test only cares about what the reset button does.
        session = at.session_state["session"]
        my_report, opp_report = session.recommend_shield()
        session.lock_shields(my_report.iloc[0]["option"], opp_report.iloc[0]["option"])
        my_report, opp_report = session.swords_reports
        session.lock_swords(my_report.iloc[0]["option"], opp_report.iloc[0]["option"])
        my_report, opp_report = session.accept_reports
        session.lock_accept(my_report.iloc[0]["option"], opp_report.iloc[0]["option"])
        at.session_state["stage"] = "done"
        at.run()

        nueva_partida = next((b for b in at.button if b.label == "Empezar una nueva partida"), None)
        assert nueva_partida is not None
        nueva_partida.click().run()

        assert not at.exception
        assert "model" in at.session_state, "loading a new match should not discard the loaded predictions"
