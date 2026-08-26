"""
Pairing formats: each module here implements one team-size-specific
pairing algorithm on top of the shared pairing_engine.zero_sum solver
and pairing_engine.imputation matrix prep.

A format module is expected to expose:
  - TEAM_SIZE: int
  - build_model(escudo_df, espada_df, descarte_df) -> a model object
  - a LivePairingSession class that walks through that format's real
    decision sequence stage by stage, each recommend_*() returning
    (my_report, opp_report) DataFrames for the UI to render

four_player is the only format so far (ported from the original
notebook). When the N-player version is ready, add it as e.g.
n_player.py following the same shape and register it below --
app.py should only need to pick a different entry from FORMATS,
keyed by team size.
"""

from . import four_player

FORMATS = {
    four_player.TEAM_SIZE: four_player,
}


def get_format(team_size: int):
    try:
        return FORMATS[team_size]
    except KeyError:
        raise ValueError(
            f"No pairing format registered for team size {team_size}. "
            f"Available: {sorted(FORMATS)}"
        )
