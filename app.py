import streamlit as st

st.set_page_config(
    page_title="Pairing Maker",
    page_icon="⚔️",
    layout="wide",
)

st.title("Pairing Maker")
st.caption("Live pairing recommendations for competitive 40k team tournaments.")

st.info(
    "This is the scaffold. Next steps: port the algorithm into "
    "pairing_engine/algorithm.py, wire up pairing_engine/sheets.py to your "
    "team's spreadsheet, and build out the pairing workflow below."
)
