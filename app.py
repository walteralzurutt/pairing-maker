import pandas as pd
import streamlit as st

from pairing_engine import sheets
from pairing_engine.imputation import matrices_from_raw_sheet
from pairing_engine.formats import get_format
from pairing_engine.formats.four_player import insight_sentences, summarize_matches

st.set_page_config(page_title="Pairing Maker", page_icon="⚔️", layout="wide")

DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "10pjgA_WKI0N6SeAQ9iuWs3MfKAN-KqRAABP77sWH2bo/edit?gid=2038464823"
)

st.title("Pairing Maker")
st.caption("Live pairing recommendations for competitive 40k team tournaments.")


def _reset():
    for key in ("model", "escudo", "espada", "descarte", "was_imputed",
                "my_team", "session", "stage"):
        st.session_state.pop(key, None)


def _fmt_option(opt):
    return " + ".join(opt) if isinstance(opt, tuple) else str(opt)


def _show_reports(my_label, my_report, opp_label, opp_report):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{my_label}**")
        st.dataframe(my_report, hide_index=True, use_container_width=True)
        top = my_report.iloc[0]
        st.success(f"Recommended: **{_fmt_option(top['option'])}** "
                   f"(equilibrium weight {top['equilibrium_weight']:.0%})")
        mixers = my_report[my_report["equilibrium_weight"] > 0]
        if len(mixers) > 1:
            st.warning("Equilibrium mixes across multiple options here -- no single "
                       "always-correct pick. Rotate across matches roughly in these "
                       "proportions:\n\n" + "\n".join(
                           f"- {_fmt_option(r['option'])}: {r['equilibrium_weight']:.0%}"
                           for _, r in mixers.iterrows()))
        for s in insight_sentences(my_report):
            st.caption(s)
    with col2:
        st.markdown(f"**{opp_label}** _(prediction -- confirm once revealed)_")
        st.dataframe(opp_report, hide_index=True, use_container_width=True)
        top = opp_report.iloc[0]
        st.info(f"Likely pick if they play rationally: **{_fmt_option(top['option'])}** "
                f"(equilibrium weight {top['equilibrium_weight']:.0%})")


# ---------------------------------------------------------------------
# Step 0: load predictions from the shared spreadsheet
# ---------------------------------------------------------------------

st.header("1. Load predictions")
sheet_url = st.text_input("Google Sheet URL (Matriz Simple tab)", value=DEFAULT_SHEET_URL)

if st.button("Load / refresh predictions", type="primary"):
    try:
        raw = sheets.load_matrix_from_url(sheet_url)
        escudo, espada, descarte, was_imputed = matrices_from_raw_sheet(raw)
        fmt = get_format(len(descarte.index))
        model = fmt.build_model(escudo, espada, descarte)
    except Exception as exc:
        st.error(f"Couldn't load/build the model: {exc}")
    else:
        _reset()
        st.session_state.model = model
        st.session_state.escudo, st.session_state.espada, st.session_state.descarte = escudo, espada, descarte
        st.session_state.was_imputed = was_imputed
        st.success(f"Loaded {len(model.team1_players)} vs {len(model.team2_players)} players.")

if "model" not in st.session_state:
    st.stop()

model = st.session_state.model
escudo, espada, descarte = st.session_state.escudo, st.session_state.espada, st.session_state.descarte

with st.expander("Predictions matrix (escudo / espada / descarte, and which cells were auto-derived)"):
    st.markdown("**descarte** (baseline)")
    st.dataframe(descarte, use_container_width=True)
    st.markdown("**escudo** (map-boosted)")
    st.dataframe(escudo, use_container_width=True)
    st.markdown("**espada** (map-penalized)")
    st.dataframe(espada, use_container_width=True)
    st.markdown("**auto-derived from map dependency** (True = not hand-entered)")
    st.dataframe(st.session_state.was_imputed, use_container_width=True)

st.caption(f"Model-expected score if both sides play optimally: "
           f"{model.expected_team1_score:.1f} - {model.expected_team2_score:.1f} (out of 80)")

# ---------------------------------------------------------------------
# Step 1: which side is us?
# ---------------------------------------------------------------------

st.header("2. Which side is your team?")
team_choice = st.radio(
    "Your team",
    options=["team1", "team2"],
    format_func=lambda t: f"{', '.join(model.team1_players)}" if t == "team1"
    else f"{', '.join(model.team2_players)}",
    key="my_team",
)

if st.session_state.get("session") is None or st.session_state.get("_session_team") != team_choice:
    st.session_state.session = get_format(len(model.team1_players)).LivePairingSession(
        model, escudo, espada, descarte, my_team=team_choice)
    st.session_state._session_team = team_choice
    st.session_state.stage = "shield"

session = st.session_state.session

# ---------------------------------------------------------------------
# Stage 1: shield
# ---------------------------------------------------------------------

st.header("3. Stage 1 -- Shield")
my_report, opp_report = session.recommend_shield()
_show_reports("Your options", my_report, "Opponent (predicted)", opp_report)

if st.session_state.stage == "shield":
    c1, c2, c3 = st.columns([2, 2, 1])
    my_shield = c1.selectbox("Your actual shield", session.my_players,
                              index=session.my_players.index(my_report.iloc[0]["option"]))
    opp_shield = c2.selectbox("Opponent's revealed shield", session.opp_players,
                               index=session.opp_players.index(opp_report.iloc[0]["option"]))
    if c3.button("Lock shields ->", type="primary"):
        session.lock_shields(my_shield, opp_shield)
        st.session_state.stage = "throw"
        st.rerun()

if st.session_state.stage in ("throw", "espada", "done"):
    st.info(f"Shields locked -- you: **{session.my_shield}**, opponent: **{session.opp_shield}**")

    # -------------------------------------------------------------
    # Stage 2: throw
    # -------------------------------------------------------------
    st.header("4. Stage 2 -- Throw")
    my_report, opp_report = session.throw_reports
    _show_reports("Your options", my_report, "Opponent (predicted)", opp_report)

    if st.session_state.stage == "throw":
        c1, c2, c3 = st.columns([2, 2, 1])
        my_throw = c1.selectbox("Your actual throw", my_report["option"].tolist(),
                                 format_func=_fmt_option)
        opp_throw = c2.selectbox("Opponent's revealed throw", opp_report["option"].tolist(),
                                  format_func=_fmt_option)
        if c3.button("Lock throws ->", type="primary"):
            session.lock_throws(my_throw, opp_throw)
            st.session_state.stage = "espada"
            st.rerun()

if st.session_state.stage in ("espada", "done"):
    st.info(f"Throws locked -- you: **{_fmt_option(session.my_throw)}**, "
            f"opponent: **{_fmt_option(session.opp_throw)}**")

    # -------------------------------------------------------------
    # Stage 3: espada
    # -------------------------------------------------------------
    st.header("5. Stage 3 -- Espada")
    my_report, opp_report = session.espada_reports
    _show_reports("Your shield's options", my_report, "Opponent's shield (predicted)", opp_report)

    if st.session_state.stage == "espada":
        c1, c2, c3 = st.columns([2, 2, 1])
        my_pick = c1.selectbox("Who did your shield actually pick?", my_report["option"].tolist())
        opp_pick = c2.selectbox("Who did the opponent's shield actually pick?", opp_report["option"].tolist())
        if c3.button("Finalize", type="primary"):
            session.finalize(my_espada_pick=my_pick, opp_espada_pick=opp_pick)
            st.session_state.stage = "done"
            st.rerun()

if st.session_state.stage == "done":
    st.header("6. Final pairings")
    result = session.result
    summary = summarize_matches(result["matches"], session.my_team, result["my_total"], result["opp_total"])
    st.dataframe(summary, hide_index=True, use_container_width=True)
    if result["my_total"] > result["opp_total"]:
        st.success(f"**You: {result['my_total']:.0f} / 80** vs Opponent: {result['opp_total']:.0f} / 80")
    elif result["my_total"] < result["opp_total"]:
        st.warning(f"You: {result['my_total']:.0f} / 80 vs **Opponent: {result['opp_total']:.0f} / 80**")
    else:
        st.info(f"You: {result['my_total']:.0f} / 80 vs Opponent: {result['opp_total']:.0f} / 80 -- tied")

if st.session_state.stage != "shield":
    if st.button("Start a new match"):
        _reset()
        st.rerun()
