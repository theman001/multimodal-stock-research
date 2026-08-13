import pandas as pd
import pytest

from src.data_collection.event_mapping import (
    _dart_api_key,
    _match_kr_tickers,
    _match_us_tickers,
    _parse_dart_corp_code_xml,
    _parse_sec_company_tickers,
    _sec_user_agent,
)


def test_parse_sec_company_tickers_normalizes_to_upper():
    raw = {"0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple Inc."}}
    df = _parse_sec_company_tickers(raw)
    assert df.iloc[0]["ticker"] == "AAPL"
    assert df.iloc[0]["cik"] == 320193


def test_match_us_tickers_exact_match():
    lookup = pd.DataFrame([{"ticker": "AAPL", "cik": 320193, "title": "Apple Inc."}])
    result, missing = _match_us_tickers(["AAPL"], lookup)
    assert missing == []
    assert result.iloc[0]["cik"] == 320193


def test_match_us_tickers_absorbs_dash_dot_difference():
    """SEC는 'BRK-B'를 'BRK.B'로 표기하는 경우가 있다 — 양방향으로 흡수해야 함."""
    lookup = pd.DataFrame([{"ticker": "BRK.B", "cik": 1067983, "title": "Berkshire Hathaway"}])
    result, missing = _match_us_tickers(["BRK-B"], lookup)
    assert missing == []
    assert result.iloc[0]["cik"] == 1067983


def test_match_us_tickers_reports_missing_without_raising():
    lookup = pd.DataFrame([{"ticker": "AAPL", "cik": 320193, "title": "Apple Inc."}])
    result, missing = _match_us_tickers(["AAPL", "NOPE"], lookup)
    assert missing == ["NOPE"]
    assert len(result) == 1


def test_match_us_tickers_applies_manual_override_for_reused_ticker():
    """'P'는 SEC 현재 스냅샷에서 Pandora와 무관한 회사를 가리킨다 — 수동 override로
    강제 교체해야 하며, 자동 매칭 결과를 그대로 쓰면 안 된다."""
    lookup = pd.DataFrame([{"ticker": "P", "cik": 1474432, "title": "Everpure, Inc."}])
    result, missing = _match_us_tickers(["P"], lookup)
    assert missing == []
    assert result.iloc[0]["cik"] == 1230276  # Pandora Media, Inc.


def test_match_us_tickers_override_also_replaces_title_not_just_cik():
    """cik만 override하고 title을 자동매칭 결과로 남기면 한 행 안에서 cik와
    title이 서로 다른 회사를 가리키는 자기모순이 생긴다(round 2 재검토에서 발견)."""
    lookup = pd.DataFrame([{"ticker": "P", "cik": 1474432, "title": "Everpure, Inc."}])
    result, _ = _match_us_tickers(["P"], lookup)
    assert result.iloc[0]["title"] == "Pandora Media, Inc."


def test_parse_dart_corp_code_xml_drops_unlisted_entries():
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <result>
        <list><corp_code>00126380</corp_code><corp_name>\xec\x82\xbc\xec\x84\xb1\xec\xa0\x84\xec\x9e\x90</corp_name><stock_code>005930</stock_code><modify_date>20260101</modify_date></list>
        <list><corp_code>00999999</corp_code><corp_name>\xeb\xb9\x84\xec\x83\x81\xec\x9e\xa5\xeb\xb2\x95\xec\x9d\xb8</corp_name><stock_code></stock_code><modify_date>20260101</modify_date></list>
    </result>"""
    df = _parse_dart_corp_code_xml(xml)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "005930"
    assert df.iloc[0]["corp_code"] == "00126380"


def test_match_kr_tickers_reports_missing_without_raising():
    all_df = pd.DataFrame([{"ticker": "005930", "corp_code": "00126380", "corp_name": "삼성전자"}])
    result, missing = _match_kr_tickers(["005930", "999999"], all_df)
    assert missing == ["999999"]
    assert len(result) == 1


def test_match_kr_tickers_falls_back_to_common_stock_code_for_preferred_shares():
    """005935(삼성전자우)는 corpCode.xml에 따로 없고 005930(삼성전자) corp_code를 공유한다."""
    all_df = pd.DataFrame([{"ticker": "005930", "corp_code": "00126380", "corp_name": "삼성전자"}])
    result, missing = _match_kr_tickers(["005935"], all_df)
    assert missing == []
    assert result.iloc[0]["corp_code"] == "00126380"


def test_match_kr_tickers_does_not_guess_when_common_code_also_missing():
    all_df = pd.DataFrame([{"ticker": "005930", "corp_code": "00126380", "corp_name": "삼성전자"}])
    result, missing = _match_kr_tickers(["999995"], all_df)
    assert missing == ["999995"]
    assert result.empty


def test_sec_user_agent_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="SEC_EDGAR_USER_AGENT"):
        _sec_user_agent()


def test_sec_user_agent_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "test-agent test@example.com")
    assert _sec_user_agent() == "test-agent test@example.com"


def test_dart_api_key_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.delenv("DART_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DART_API_KEY"):
        _dart_api_key()


def test_dart_api_key_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("DART_API_KEY", "test-key-123")
    assert _dart_api_key() == "test-key-123"
