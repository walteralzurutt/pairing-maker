"""
Pairing formats: each module here implements one pairing algorithm on top
of the shared pairing_engine.zero_sum solver, pairing_engine.imputation
matrix prep, and pairing_engine.report ranking/tie-break helpers.

A format module is expected to expose:
  - build_model(escudo_df, espada_df, descarte_df) -> a model object
  - a LivePairingSession class that walks through that format's real
    decision sequence stage by stage, each recommend_*() returning
    (my_report, opp_report) DataFrames for the UI to render

n_player is the sole format: a recursive shield/swords/accept engine
that handles any team size >= 3 (ported from a colleague's generalized
notebook, replacing the earlier four_player-only implementation). If a
genuinely different ruleset shows up later, reintroduce a registry keyed
by whatever actually distinguishes them at that point.
"""

from . import n_player


def get_format():
    return n_player
