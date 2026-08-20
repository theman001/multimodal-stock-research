from unittest.mock import patch

import pandas as pd
import pytest

from src.data_collection.events_kr import _parse_list_response, fetch_kr_disclosures


def test_parse_list_response_no_data_status_returns_empty_not_error():
    payload = {"status": "013", "message": "조회된 데이타가 없습니다."}
    df = _parse_list_response(payload, ticker="005930", pblntf_ty="B")
    assert df.empty


def test_parse_list_response_ok_status_attaches_ticker_and_pblntf_ty():
    """pblntf_ty는 DART 응답 필드가 아니라 요청 파라미터라 응답에 없다 —
    호출부에서 넘긴 값을 직접 태깅해야 한다."""
    payload = {
        "status": "000",
        "list": [
            {"corp_code": "00126380", "report_nm": "주요사항보고서", "rcept_no": "1", "rcept_dt": "20200101", "flr_nm": "삼성전자"}
        ],
    }
    df = _parse_list_response(payload, ticker="005930", pblntf_ty="B")
    assert (df["ticker"] == "005930").all()
    assert (df["pblntf_ty"] == "B").all()
    assert len(df) == 1


def test_parse_list_response_real_error_status_raises():
    """트래픽 초과/인증키 오류 같은 진짜 에러를 '공시 없음'으로 착각하면 안 된다."""
    payload = {"status": "020", "message": "요청 제한을 초과하였습니다."}
    with pytest.raises(RuntimeError):
        _parse_list_response(payload, ticker="005930", pblntf_ty="B")


def _list_page_response(rows: list[dict]) -> dict:
    return {"status": "000", "message": "정상", "total_page": 1, "list": rows}


def _empty_page_response() -> dict:
    return {"status": "013", "message": "조회된 데이타가 없습니다.", "total_page": 0, "list": []}


def test_fetch_kr_disclosures_no_cache_fetches_from_start_date(tmp_path):
    with patch("src.data_collection.events_kr._fetch_list_page") as mock_fetch:
        mock_fetch.side_effect = [
            _list_page_response([{"corp_code": "00126380", "report_nm": "r1", "rcept_no": "1", "rcept_dt": "20200105", "flr_nm": "f"}]),
            _empty_page_response(),
        ]
        result = fetch_kr_disclosures(corp_code="00126380", ticker="005930", raw_dir=tmp_path, start_date="20150101")

    # 두 pblntf_ty(B, I) 각각 bgn_de=start_date로 호출됐는지 확인
    called_bgn_des = {call.args[2] for call in mock_fetch.call_args_list}
    assert called_bgn_des == {"20150101"}
    assert len(result) == 1
    assert (tmp_path / "events" / "disclosures_005930.parquet").exists()


def test_fetch_kr_disclosures_incremental_uses_cached_max_rcept_dt(tmp_path):
    cache_path = tmp_path / "events" / "disclosures_005930.parquet"
    cache_path.parent.mkdir(parents=True)
    cached = pd.DataFrame(
        [{"ticker": "005930", "corp_code": "00126380", "pblntf_ty": "B", "report_nm": "r1", "rcept_no": "1", "rcept_dt": "20200105", "flr_nm": "f"}]
    )
    cached.to_parquet(cache_path, index=False)

    with patch("src.data_collection.events_kr._fetch_list_page") as mock_fetch:
        mock_fetch.side_effect = [
            _list_page_response([{"corp_code": "00126380", "report_nm": "r2", "rcept_no": "2", "rcept_dt": "20200210", "flr_nm": "f"}]),
            _empty_page_response(),
        ]
        result = fetch_kr_disclosures(corp_code="00126380", ticker="005930", raw_dir=tmp_path, start_date="20150101")

    called_bgn_des = {call.args[2] for call in mock_fetch.call_args_list}
    assert called_bgn_des == {"20200105"}  # start_date가 아니라 캐시 마지막 rcept_dt부터 재조회
    assert sorted(result["rcept_no"]) == ["1", "2"]


def test_fetch_kr_disclosures_incremental_dedups_by_rcept_no(tmp_path):
    cache_path = tmp_path / "events" / "disclosures_005930.parquet"
    cache_path.parent.mkdir(parents=True)
    cached = pd.DataFrame(
        [{"ticker": "005930", "corp_code": "00126380", "pblntf_ty": "B", "report_nm": "r1", "rcept_no": "1", "rcept_dt": "20200105", "flr_nm": "f"}]
    )
    cached.to_parquet(cache_path, index=False)

    with patch("src.data_collection.events_kr._fetch_list_page") as mock_fetch:
        # 재조회 시 캐시에 이미 있는 공시(rcept_no=1)가 그대로 다시 오는 경우(당일 재조회 시 흔함)
        mock_fetch.side_effect = [
            _list_page_response([{"corp_code": "00126380", "report_nm": "r1", "rcept_no": "1", "rcept_dt": "20200105", "flr_nm": "f"}]),
            _empty_page_response(),
        ]
        result = fetch_kr_disclosures(corp_code="00126380", ticker="005930", raw_dir=tmp_path)

    assert len(result) == 1


def test_fetch_kr_disclosures_incremental_false_returns_cache_without_network_call(tmp_path):
    cache_path = tmp_path / "events" / "disclosures_005930.parquet"
    cache_path.parent.mkdir(parents=True)
    cached = pd.DataFrame(
        [{"ticker": "005930", "corp_code": "00126380", "pblntf_ty": "B", "report_nm": "r1", "rcept_no": "1", "rcept_dt": "20200105", "flr_nm": "f"}]
    )
    cached.to_parquet(cache_path, index=False)

    with patch("src.data_collection.events_kr._fetch_list_page") as mock_fetch:
        result = fetch_kr_disclosures(corp_code="00126380", ticker="005930", raw_dir=tmp_path, incremental=False)

    mock_fetch.assert_not_called()
    assert len(result) == 1


def test_fetch_kr_disclosures_force_ignores_cache_and_refetches_from_start_date(tmp_path):
    cache_path = tmp_path / "events" / "disclosures_005930.parquet"
    cache_path.parent.mkdir(parents=True)
    cached = pd.DataFrame(
        [{"ticker": "005930", "corp_code": "00126380", "pblntf_ty": "B", "report_nm": "r1", "rcept_no": "1", "rcept_dt": "20200105", "flr_nm": "f"}]
    )
    cached.to_parquet(cache_path, index=False)

    with patch("src.data_collection.events_kr._fetch_list_page") as mock_fetch:
        mock_fetch.side_effect = [
            _list_page_response([{"corp_code": "00126380", "report_nm": "r9", "rcept_no": "99", "rcept_dt": "20200105", "flr_nm": "f"}]),
            _empty_page_response(),
        ]
        result = fetch_kr_disclosures(
            corp_code="00126380", ticker="005930", raw_dir=tmp_path, start_date="20150101", force=True
        )

    called_bgn_des = {call.args[2] for call in mock_fetch.call_args_list}
    assert called_bgn_des == {"20150101"}
    assert list(result["rcept_no"]) == ["99"]  # force는 캐시와 병합하지 않고 전체 대체
