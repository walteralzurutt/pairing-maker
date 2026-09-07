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


def test_editing_matrix_and_applying_rebuilds_model_and_resets_round(sample_raw_df):
    """Editing a cell in the escudo data_editor and clicking 'Aplicar
    cambios' must rebuild the model with the edited value and restart the
    round, without needing to touch the source spreadsheet."""
    at = _load_app(sample_raw_df)
    with patch.object(sheets_module, "load_matrix_from_url", lambda url: sample_raw_df):
        assert at.session_state["escudo"].loc["Death Guard", "Marines"] == pytest.approx(11.666667)

        # Simulate editing row 0 ("Death Guard"), column "Marines" -- this is
        # the internal shape st.data_editor's widget state actually takes;
        # AppTest has no dedicated data_editor interaction API to drive this
        # through a higher-level call.
        at.session_state["escudo_editor"] = {"edited_rows": {0: {"Marines": 18.0}},
                                              "added_rows": [], "deleted_rows": []}
        apply_btn = next(b for b in at.button if b.label == "Aplicar cambios a la matriz")
        apply_btn.click().run()

        assert not at.exception
        assert at.session_state["escudo"].loc["Death Guard", "Marines"] == 18.0
        assert at.session_state["session"].model.escudo_df.loc["Death Guard", "Marines"] == 18.0
        assert at.session_state["session"].round_number == 1  # round was reset, not carried over


def test_reloading_a_different_sheet_clears_stale_editor_state(sample_raw_df):
    """Regression/characterization: st.data_editor resets its cached edit
    state when the underlying source data changes, even under the same
    widget key -- so after a fresh 'Cargar predicciones' reload, clicking
    "Aplicar cambios" again (without touching the editor) must NOT
    silently reapply an edit made against the previous load.
    """
    at = _load_app(sample_raw_df)
    with patch.object(sheets_module, "load_matrix_from_url", lambda url: sample_raw_df):
        at.session_state["escudo_editor"] = {"edited_rows": {0: {"Marines": 15.0}},
                                              "added_rows": [], "deleted_rows": []}
        apply_btn = next(b for b in at.button if b.label == "Aplicar cambios a la matriz")
        apply_btn.click().run()
        assert at.session_state["escudo"].loc["Death Guard", "Marines"] == 15.0  # edit applied

        at.button[0].click().run()  # "Cargar / actualizar predicciones" again, no further edits

        # click Apply again without touching the editor -- if the widget's
        # stale state survived the reload, this silently reapplies 15.0
        apply_btn = next(b for b in at.button if b.label == "Aplicar cambios a la matriz")
        apply_btn.click().run()

        assert not at.exception
        assert at.session_state["escudo"].loc["Death Guard", "Marines"] == pytest.approx(11.666667), (
            "stale data_editor state from the previous sheet load leaked into the fresh reload"
        )
