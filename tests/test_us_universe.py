from src.data_collection.us_universe import SIZE_ID_BY_INDEX, _normalize_ticker


def test_normalize_ticker_converts_dot_to_dash():
    assert _normalize_ticker("BRK.B") == "BRK-B"
    assert _normalize_ticker("BF.B") == "BF-B"


def test_normalize_ticker_leaves_plain_tickers_unchanged():
    assert _normalize_ticker("AAPL") == "AAPL"


def test_size_id_mapping_matches_claude_md_spec():
    # CLAUDE.md: S&P500=대형(2), S&P400=중형(1), S&P600=소형(0)
    assert SIZE_ID_BY_INDEX["sp500"] == 2
    assert SIZE_ID_BY_INDEX["sp400"] == 1
    assert SIZE_ID_BY_INDEX["sp600"] == 0
