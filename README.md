# Pairing Maker

Game-theory-driven pairing recommendations for competitive Warhammer 40k team tournaments, built as a Streamlit app.

## Layout

- `app.py` — Streamlit UI entry point; drives the live shield/throw/espada walkthrough for one match.
- `pairing_engine/zero_sum.py` — generic zero-sum game LP solver, shared by every pairing format.
- `pairing_engine/imputation.py` — parses the "Matriz Simple" spreadsheet tab and derives the full escudo/espada/descarte matrices.
- `pairing_engine/sheets.py` — fetches a tab of the shared Google Sheet via its public CSV export URL.
- `pairing_engine/formats/four_player.py` — the 4-player-specific pairing algorithm (ported from the original notebook). A future N-player algorithm goes in a sibling module here, registered in `pairing_engine/formats/__init__.py`.
- `assets/factions/` — faction logo images used in the UI.

## Spreadsheet input

The app reads the "Matriz Simple" tab of your team's shared Google Sheet directly via its public CSV export URL — no credentials or Google Cloud setup needed. This requires the sheet to be shared as **"Anyone with the link → Viewer"**.

That's a real privacy trade-off: since the whole point of the model is that predictions are hidden information, anyone with the sheet's URL (not just your team) could read your predictions before a match. Keep the link off any public channel your opponents might see, and consider switching to a private, credentialed integration (e.g. a Google service account) if that risk matters more than the extra setup.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Push to GitHub and deploy via [Streamlit Community Cloud](https://streamlit.io/cloud) (free tier).
