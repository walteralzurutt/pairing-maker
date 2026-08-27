import pytest

from pairing_engine.sheets import parse_sheet_url


def test_parse_sheet_url_normal_link():
    """Regression: the normal 'Share' edit-link format still parses correctly."""
    url = ("https://docs.google.com/spreadsheets/d/"
           "10pjgA_WKI0N6SeAQ9iuWs3MfKAN-KqRAABP77sWH2bo/edit?gid=2038464823#gid=2038464823")
    spreadsheet_id, gid = parse_sheet_url(url)
    assert spreadsheet_id == "10pjgA_WKI0N6SeAQ9iuWs3MfKAN-KqRAABP77sWH2bo"
    assert gid == "2038464823"


def test_parse_sheet_url_publish_to_web_raises_clear_error():
    """Google's 'Publish to web' link format (/d/e/<id>/pubhtml) must not be
    silently mis-parsed into a bogus id ('e') -- it should raise a clear,
    actionable error instead.
    """
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQsomeFakeId1234/pubhtml?gid=456"
    with pytest.raises(ValueError, match="[Pp]ublish"):
        parse_sheet_url(url)
