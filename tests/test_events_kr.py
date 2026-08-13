import pytest

from src.data_collection.events_kr import _parse_list_response


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
