import pandas as pd
import streamlit as st

from pairing_engine import sheets
from pairing_engine.imputation import matrices_from_raw_sheet
from pairing_engine.formats import get_format
from pairing_engine.formats.four_player import (
    insight_sentences,
    summarize_matches,
    COLUMN_LABELS_ES,
    SUMMARY_COLUMN_LABELS_ES,
)

st.set_page_config(page_title="Pairing Maker", page_icon="⚔️", layout="wide")

DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "10pjgA_WKI0N6SeAQ9iuWs3MfKAN-KqRAABP77sWH2bo/edit?gid=2038464823"
)

MY_TEAM = "team1"  # el equipo de las filas siempre es el nuestro

st.title("Pairing Maker")
st.caption("Recomendaciones de emparejamiento en vivo, basadas en teoría de juegos, para torneos competitivos de equipos de 40k.")


def _reset():
    for key in ("model", "escudo", "espada", "descarte", "was_imputed", "session", "stage"):
        st.session_state.pop(key, None)


def _fmt_option(opt):
    return " y ".join(opt) if isinstance(opt, tuple) else str(opt)


def _display_report(report: pd.DataFrame) -> pd.DataFrame:
    return report.rename(columns=COLUMN_LABELS_ES)


def _show_reports(my_label, my_report, opp_label, opp_report):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{my_label}**")
        st.dataframe(_display_report(my_report), hide_index=True, use_container_width=True)
        top = my_report.iloc[0]
        st.success(f"Recomendado: **{_fmt_option(top['option'])}** "
                   f"(peso de equilibrio {top['equilibrium_weight']:.0%})")
        mixers = my_report[my_report["equilibrium_weight"] > 0]
        if len(mixers) > 1:
            st.warning("El equilibrio se mezcla entre varias opciones acá -- no hay una única "
                       "elección siempre correcta. Alterná entre partidas más o menos en estas "
                       "proporciones:\n\n" + "\n".join(
                           f"- {_fmt_option(r['option'])}: {r['equilibrium_weight']:.0%}"
                           for _, r in mixers.iterrows()))
        for s in insight_sentences(my_report):
            st.caption(s)
    with col2:
        st.markdown(f"**{opp_label}** _(predicción -- confirmar cuando se revele)_")
        st.dataframe(_display_report(opp_report), hide_index=True, use_container_width=True)
        top = opp_report.iloc[0]
        st.info(f"Elección probable si juegan racionalmente: **{_fmt_option(top['option'])}** "
                f"(peso de equilibrio {top['equilibrium_weight']:.0%})")


# ---------------------------------------------------------------------
# Paso 0: cargar las predicciones desde la planilla compartida
# ---------------------------------------------------------------------

st.header("1. Cargar predicciones")
sheet_url = st.text_input("URL de la planilla de Google (pestaña Matriz Simple)", value=DEFAULT_SHEET_URL)

if st.button("Cargar / actualizar predicciones", type="primary"):
    try:
        raw = sheets.load_matrix_from_url(sheet_url)
        escudo, espada, descarte, was_imputed = matrices_from_raw_sheet(raw)
        fmt = get_format(len(descarte.index))
        model = fmt.build_model(escudo, espada, descarte)
    except Exception as exc:
        st.error(f"No se pudo cargar/armar el modelo: {exc}")
    else:
        _reset()
        st.session_state.model = model
        st.session_state.escudo, st.session_state.espada, st.session_state.descarte = escudo, espada, descarte
        st.session_state.was_imputed = was_imputed
        st.success(f"Se cargaron {len(model.team1_players)} vs {len(model.team2_players)} jugadores.")

if "model" not in st.session_state:
    st.stop()

model = st.session_state.model
escudo, espada, descarte = st.session_state.escudo, st.session_state.espada, st.session_state.descarte

st.caption(f"Nuestro equipo (filas de la planilla): {', '.join(model.team1_players)} "
           f"— Rival (columnas): {', '.join(model.team2_players)}")

with st.expander("Matriz de predicciones (escudo / espada / descarte, y qué celdas se derivaron automáticamente)"):
    st.markdown("**descarte** (base)")
    st.dataframe(descarte, use_container_width=True)
    st.markdown("**escudo** (con ventaja de mapa)")
    st.dataframe(escudo, use_container_width=True)
    st.markdown("**espada** (con penalización de mapa)")
    st.dataframe(espada, use_container_width=True)
    st.markdown("**derivado automáticamente según dependencia de mapa** (True = no ingresado a mano)")
    st.dataframe(st.session_state.was_imputed, use_container_width=True)

st.caption(f"Puntaje esperado por el modelo si ambos equipos juegan de forma óptima: "
           f"{model.expected_team1_score:.1f} - {model.expected_team2_score:.1f} (sobre 80)")

if "session" not in st.session_state:
    st.session_state.session = get_format(len(model.team1_players)).LivePairingSession(
        model, escudo, espada, descarte, my_team=MY_TEAM)
    st.session_state.stage = "shield"

session = st.session_state.session

# ---------------------------------------------------------------------
# Etapa 1: escudo
# ---------------------------------------------------------------------

st.header("2. Etapa 1 -- Escudo")
my_report, opp_report = session.recommend_shield()
_show_reports("Nuestras opciones", my_report, "Rival (predicción)", opp_report)

if st.session_state.stage == "shield":
    c1, c2, c3 = st.columns([2, 2, 1])
    my_shield = c1.selectbox("Nuestro escudo real", session.my_players,
                              index=session.my_players.index(my_report.iloc[0]["option"]))
    opp_shield = c2.selectbox("Escudo revelado del rival", session.opp_players,
                               index=session.opp_players.index(opp_report.iloc[0]["option"]))
    if c3.button("Fijar escudos ->", type="primary"):
        session.lock_shields(my_shield, opp_shield)
        st.session_state.stage = "throw"
        st.rerun()

if st.session_state.stage in ("throw", "espada", "done"):
    st.info(f"Escudos fijados -- nosotros: **{session.my_shield}**, rival: **{session.opp_shield}**")

    # -------------------------------------------------------------
    # Etapa 2: lanzamiento
    # -------------------------------------------------------------
    st.header("3. Etapa 2 -- Lanzamiento")
    my_report, opp_report = session.throw_reports
    _show_reports("Nuestras opciones", my_report, "Rival (predicción)", opp_report)

    if st.session_state.stage == "throw":
        c1, c2, c3 = st.columns([2, 2, 1])
        my_throw = c1.selectbox("Nuestro lanzamiento real", my_report["option"].tolist(),
                                 format_func=_fmt_option)
        opp_throw = c2.selectbox("Lanzamiento revelado del rival", opp_report["option"].tolist(),
                                  format_func=_fmt_option)
        if c3.button("Fijar lanzamientos ->", type="primary"):
            session.lock_throws(my_throw, opp_throw)
            st.session_state.stage = "espada"
            st.rerun()

if st.session_state.stage in ("espada", "done"):
    st.info(f"Lanzamientos fijados -- nosotros: **{_fmt_option(session.my_throw)}**, "
            f"rival: **{_fmt_option(session.opp_throw)}**")

    # -------------------------------------------------------------
    # Etapa 3: espada
    # -------------------------------------------------------------
    st.header("4. Etapa 3 -- Espada")
    my_report, opp_report = session.espada_reports
    _show_reports("Opciones de nuestro escudo", my_report, "Escudo rival (predicción)", opp_report)

    if st.session_state.stage == "espada":
        c1, c2, c3 = st.columns([2, 2, 1])
        my_pick = c1.selectbox("¿A quién eligió realmente nuestro escudo?", my_report["option"].tolist())
        opp_pick = c2.selectbox("¿A quién eligió realmente el escudo rival?", opp_report["option"].tolist())
        if c3.button("Finalizar", type="primary"):
            session.finalize(my_espada_pick=my_pick, opp_espada_pick=opp_pick)
            st.session_state.stage = "done"
            st.rerun()

if st.session_state.stage == "done":
    st.header("5. Emparejamientos finales")
    result = session.result
    summary = summarize_matches(result["matches"], session.my_team, result["my_total"], result["opp_total"])
    st.dataframe(summary.rename(columns=SUMMARY_COLUMN_LABELS_ES), hide_index=True, use_container_width=True)
    if result["my_total"] > result["opp_total"]:
        st.success(f"**Nosotros: {result['my_total']:.0f} / 80** vs Rival: {result['opp_total']:.0f} / 80")
    elif result["my_total"] < result["opp_total"]:
        st.warning(f"Nosotros: {result['my_total']:.0f} / 80 vs **Rival: {result['opp_total']:.0f} / 80**")
    else:
        st.info(f"Nosotros: {result['my_total']:.0f} / 80 vs Rival: {result['opp_total']:.0f} / 80 -- empate")

if st.session_state.stage != "shield":
    if st.button("Empezar una nueva partida"):
        _reset()
        st.rerun()
