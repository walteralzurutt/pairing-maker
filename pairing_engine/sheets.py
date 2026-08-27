"""
Google Sheets integration: pulls the "Matriz Simple" tab from the team's
shared spreadsheet via its public CSV export URL. The sheet must be
shared as "Anyone with the link -> Viewer" for this to work -- see the
README for the privacy trade-off that implies (predictions are only as
hidden as that URL).
"""

from __future__ import annotations

import re
from urllib.parse import urlparse, parse_qs

import pandas as pd

_ID_RE = re.compile(r"/spreadsheets/d/([a-zA-Z0-9-_]+)")
_PUBLISHED_RE = re.compile(r"/spreadsheets/d/e/")


def parse_sheet_url(sheet_url: str) -> tuple[str, str | None]:
    """Extract (spreadsheet_id, gid) from a normal "shared link" Google
    Sheets URL, e.g. https://docs.google.com/spreadsheets/d/<id>/edit?gid=<gid>#gid=<gid>.
    gid is None if the URL doesn't specify a tab (falls back to the first sheet).
    """
    if _PUBLISHED_RE.search(sheet_url):
        raise ValueError(
            "This looks like a 'Publish to web' link, which isn't supported. Use the normal "
            "Share link instead (Share -> General access -> Anyone with the link -> Viewer, "
            "then copy the link from the Share dialog)."
        )
    match = _ID_RE.search(sheet_url)
    if not match:
        raise ValueError(f"Couldn't find a spreadsheet ID in {sheet_url!r}")
    spreadsheet_id = match.group(1)

    parsed = urlparse(sheet_url)
    gid = parse_qs(parsed.query).get("gid", [None])[0]
    if gid is None and parsed.fragment:
        gid = parse_qs(parsed.fragment).get("gid", [None])[0]
    return spreadsheet_id, gid


def load_matrix(spreadsheet_id: str, gid: str | None = None) -> pd.DataFrame:
    """Fetch one tab of a public Google Sheet as a raw DataFrame (first row
    as column headers, no index set) -- ready to hand to
    pairing_engine.imputation.matrices_from_raw_sheet().
    """
    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv"
    if gid is not None:
        url += f"&gid={gid}"
    try:
        return pd.read_csv(url)
    except Exception as exc:
        raise RuntimeError(
            f"Couldn't fetch the sheet as CSV ({exc}). Make sure it's shared as "
            f"\"Anyone with the link -> Viewer\" and the URL/tab is correct."
        ) from exc


def load_matrix_from_url(sheet_url: str) -> pd.DataFrame:
    """Convenience: parse a full Google Sheets URL and fetch that tab directly."""
    spreadsheet_id, gid = parse_sheet_url(sheet_url)
    return load_matrix(spreadsheet_id, gid)
