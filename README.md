# Pairing Maker

Game-theory-driven pairing recommendations for competitive Warhammer 40k team tournaments, built as a Streamlit app.

## Layout

- `app.py` — Streamlit UI entry point.
- `pairing_engine/algorithm.py` — the core pairing algorithm (ported from the original notebook), independent of any UI/IO code.
- `pairing_engine/sheets.py` — Google Sheets integration for pulling in player predictions.
- `assets/factions/` — faction logo images used in the UI.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml  # then fill in your service account + sheet URL
streamlit run app.py
```

## Deployment

Push to GitHub and deploy via [Streamlit Community Cloud](https://streamlit.io/cloud) (free tier). Set the contents of `.streamlit/secrets.toml` in the app's Secrets panel rather than committing it.
