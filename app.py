import pandas as pd
import streamlit as st

from pairing_engine import sheets
from pairing_engine.imputation import matrices_from_raw_sheet
from pairing_engine.formats import get_format
from pairing_engine.formats.n_player import SUMMARY_COLUMN_LABELS_ES, WORST_CASE_COLUMN_LABELS_ES, summarize_matches
from pairing_engine.report import insight_sentences, COLUMN_LABELS_ES

st.set_page_config(page_title="Pairing Maker", page_icon="⚔️", layout="wide")

DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "10pjgA_WKI0N6SeAQ9iuWs3MfKAN-KqRAABP77sWH2bo/edit?gid=2038464823"
)

MY_TEAM = "team1"  # el equipo de las filas siempre es el nuestro
NO_LO_SE = "No lo sé"

st.title("Pairing Maker")
st.caption("Recomendaciones de emparejamiento en vivo, basadas en teoría de juegos, para torneos competitivos de equipos de 40k.")


def _reset_session():
    for key in ("session", "stage", "known_opponent_shield_choice"):
        st.session_state.pop(key, None)


def _reset_all():
    for key in ("model", "escudo", "espada", "descarte", "was_imputed"):
        st.session_state.pop(key, None)
    _reset_session()


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
            st.warning("El equilibrio se mezcla entre varias opciones aquí -- no hay una única "
                       "elección siempre correcta. Alterna entre partidas más o menos en estas "
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
        fmt = get_format()
        model = fmt.build_model(escudo, espada, descarte)
    except Exception as exc:
        st.error(f"No se pudo cargar/armar el modelo: {exc}")
    else:
        _reset_all()
        st.session_state.model = model
        st.session_state.escudo, st.session_state.espada, st.session_state.descarte = escudo, espada, descarte
        st.session_state.was_imputed = was_imputed
        st.success(f"Se cargaron {len(model.team1_players)} vs {len(model.team2_players)} jugadores.")

if "model" not in st.session_state:
    st.stop()

model = st.session_state.model
escudo, espada, descarte = st.session_state.escudo, st.session_state.espada, st.session_state.descarte
team_size = len(model.team1_players)
total_pool = 20 * team_size

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
           f"{model.expected_team1_score:.1f} - {model.expected_team2_score:.1f} (sobre {total_pool})")

if "session" not in st.session_state:
    st.session_state.session = get_format().LivePairingSession(model, my_team=MY_TEAM)
    st.session_state.stage = "shield"

session = st.session_state.session

# ---------------------------------------------------------------------
# Rondas ya completadas (si estamos en la ronda 2+, o el partido terminó)
# ---------------------------------------------------------------------

if session.history:
    rondas_completadas = session.round_number - (0 if st.session_state.stage == "done" else 1)
    with st.expander(f"Rondas completadas ({rondas_completadas})", expanded=(st.session_state.stage == "done")):
        historial = summarize_matches(session.history, session.my_team)
        st.dataframe(historial.rename(columns=SUMMARY_COLUMN_LABELS_ES), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------
# Ronda en curso
# ---------------------------------------------------------------------

if st.session_state.stage != "done":
    st.header(f"Ronda {session.round_number} -- Quedan {len(session.my_players)} jugadores por equipo")

    # ---- Etapa 1: escudo -------------------------------------------------
    st.subheader("Etapa 1 -- Escudo")

    if st.session_state.stage == "shield":
        known_opponent_shield = None
        if session.round_number == 1:
            choice = st.selectbox(
                "Si sabes con certeza cuál será el escudo rival, selecciónalo",
                [NO_LO_SE] + session.opp_players, key="known_opponent_shield_choice",
            )
            known_opponent_shield = None if choice == NO_LO_SE else choice
        my_report, opp_report = session.recommend_shield(known_opponent_shield=known_opponent_shield)
    else:
        my_report, opp_report = session.shield_reports

    _show_reports("Nuestras opciones", my_report, "Rival (predicción)", opp_report)

    if session.worst_case is not None:
        with st.expander("Escenario del peor caso"):
            st.caption(
                "La tabla de arriba asume que el rival juega su mezcla óptima habitual, sin saber "
                "qué escudo vamos a elegir. Esta otra tabla muestra el escenario contrario: para "
                "cada escudo que podríamos elegir, ¿qué pasaría si el rival ya conociera esa "
                "elección de antemano y respondiera exactamente con la opción que más nos "
                "perjudica? Así vemos cuánto podemos perder en el peor de los casos, no solo el "
                "resultado promedio esperado."
            )
            st.dataframe(session.worst_case.rename(columns=WORST_CASE_COLUMN_LABELS_ES),
                         hide_index=True, use_container_width=True)

    if st.session_state.stage == "shield":
        c1, c2, c3 = st.columns([2, 2, 1])
        my_shield = c1.selectbox("Nuestro escudo real", session.my_players,
                                  index=session.my_players.index(my_report.iloc[0]["option"]))
        opp_shield = c2.selectbox("Escudo revelado del rival", session.opp_players,
                                   index=session.opp_players.index(opp_report.iloc[0]["option"]))
        if c3.button("Fijar escudos ->", type="primary"):
            session.lock_shields(my_shield, opp_shield)
            st.session_state.stage = "swords"
            st.rerun()

    if st.session_state.stage in ("swords", "accept"):
        st.info(f"Escudos fijados -- nosotros: **{session.my_shield}**, rival: **{session.opp_shield}**")

        # ---- Etapa 2: espadas ------------------------------------------
        st.subheader("Etapa 2 -- Espadas")
        my_report, opp_report = session.swords_reports
        _show_reports("Nuestras opciones", my_report, "Rival (predicción)", opp_report)

        if st.session_state.stage == "swords":
            c1, c2, c3 = st.columns([2, 2, 1])
            my_swords = c1.selectbox("Nuestras espadas reales", my_report["option"].tolist(),
                                      format_func=_fmt_option)
            opp_swords = c2.selectbox("Espadas reveladas del rival", opp_report["option"].tolist(),
                                       format_func=_fmt_option)
            if c3.button("Fijar espadas ->", type="primary"):
                session.lock_swords(my_swords, opp_swords)
                st.session_state.stage = "accept"
                st.rerun()

    if st.session_state.stage == "accept":
        st.info(f"Espadas fijadas -- nosotros: **{_fmt_option(session.my_swords)}**, "
                f"rival: **{_fmt_option(session.opp_swords)}**")

        # ---- Etapa 3: aceptación --------------------------------------
        st.subheader("Etapa 3 -- Aceptación")
        my_report, opp_report = session.accept_reports
        _show_reports("Opciones de nuestro escudo", my_report, "Escudo rival (predicción)", opp_report)

        if session.deviation_explanation:
            st.warning(f"⚠️ {session.deviation_explanation}")

        c1, c2, c3 = st.columns([2, 2, 1])
        my_pick = c1.selectbox("¿A quién eligió realmente nuestro escudo?", my_report["option"].tolist())
        opp_pick = c2.selectbox("¿A quién eligió realmente el escudo rival?", opp_report["option"].tolist())
        if c3.button("Fijar aceptación ->", type="primary"):
            status = session.lock_accept(my_pick, opp_pick)
            st.session_state.stage = "done" if status == "done" else "shield"
            st.rerun()

# ---------------------------------------------------------------------
# Resultado final
# ---------------------------------------------------------------------

if st.session_state.stage == "done":
    st.header("Emparejamientos finales")
    result = session.result
    summary = summarize_matches(result["matches"], session.my_team)
    st.dataframe(summary.rename(columns=SUMMARY_COLUMN_LABELS_ES), hide_index=True, use_container_width=True)
    pool = result["total_pool"]
    if result["my_total"] > result["opp_total"]:
        st.success(f"**Nosotros: {result['my_total']:.0f} / {pool:.0f}** vs Rival: {result['opp_total']:.0f} / {pool:.0f}")
    elif result["my_total"] < result["opp_total"]:
        st.warning(f"Nosotros: {result['my_total']:.0f} / {pool:.0f} vs **Rival: {result['opp_total']:.0f} / {pool:.0f}**")
    else:
        st.info(f"Nosotros: {result['my_total']:.0f} / {pool:.0f} vs Rival: {result['opp_total']:.0f} / {pool:.0f} -- empate")

if session.history:
    if st.button("Empezar una nueva partida"):
        _reset_session()
        st.rerun()
