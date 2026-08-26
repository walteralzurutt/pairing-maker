"""
Google Sheets integration: pulls player predictions from the shared
team spreadsheet using a service account (credentials come from
Streamlit secrets, see .streamlit/secrets.toml.example).
"""

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


def get_client(service_account_info: dict) -> gspread.Client:
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    return gspread.authorize(creds)


def load_predictions(client: gspread.Client, sheet_url: str, worksheet_name: str) -> pd.DataFrame:
    sheet = client.open_by_url(sheet_url).worksheet(worksheet_name)
    records = sheet.get_all_records()
    return pd.DataFrame(records)
