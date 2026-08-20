"""KISBroker 유닛테스트 — 실제 KIS API 호출 없이 requests.Session.get/post를
mock으로 대체한다(plan/10 §C-3: "브로커 어댑터 테스트는 실제 API 호출 금지",
CLAUDE.md "실행 통제": requirements.txt에 없는 외부 네트워크 호출은 승인 필요).

Alpaca와 달리 place_market_order()/is_market_open() 하나가 여러 번(토큰->
해시키->주문, 또는 토큰->휴장일조회) 연쇄 호출을 하므로, get/post를 URL
접미사로 분기하는 dispatcher를 매 테스트에서 구성한다."""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from src.live.broker.base import OrderResult
from src.live.broker.kis import KISBroker, LIVE_BASE_URL, PAPER_BASE_URL, _is_within_market_hours
from src.live.observation import HeldPosition


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


_TOKEN_PAYLOAD = {"access_token": "tok-1", "token_type": "Bearer", "expires_in": 86400}
_HASHKEY_PAYLOAD = {"HASH": "hash-1"}
_BALANCE_PAYLOAD = {
    "rt_cd": "0",
    "msg1": "정상처리 되었습니다.",
    "output1": [
        {"pdno": "005930", "hldg_qty": "10", "pchs_avg_pric": "70000"},
        {"pdno": "000660", "hldg_qty": "0", "pchs_avg_pric": "0"},
    ],
    "output2": [{"dnca_tot_amt": "1000000", "tot_evlu_amt": "1700000"}],
}
_ORDER_PAYLOAD = {"rt_cd": "0", "msg1": "정상처리 되었습니다.", "output": {"ODNO": "order-1", "ORD_TMD": "090001"}}
_HOLIDAY_TRADING_DAY_PAYLOAD = {"rt_cd": "0", "msg1": "정상", "output": [{"bass_dt": "20260820", "opnd_yn": "Y"}]}
_HOLIDAY_CLOSED_DAY_PAYLOAD = {"rt_cd": "0", "msg1": "정상", "output": [{"bass_dt": "20260815", "opnd_yn": "N"}]}


def _post_dispatcher(order_payload=None, hashkey_payload=None, token_payload=None):
    order_payload = order_payload or _ORDER_PAYLOAD
    hashkey_payload = hashkey_payload or _HASHKEY_PAYLOAD
    token_payload = token_payload or _TOKEN_PAYLOAD

    def _dispatch(url, *args, **kwargs):
        if url.endswith("/oauth2/tokenP"):
            return _FakeResponse(token_payload)
        if url.endswith("/uapi/hashkey"):
            return _FakeResponse(hashkey_payload)
        if url.endswith("/uapi/domestic-stock/v1/trading/order-cash"):
            return _FakeResponse(order_payload)
        raise AssertionError(f"예상치 못한 POST 호출: {url}")

    return _dispatch


def _get_dispatcher(balance_payload=None, holiday_payload=None):
    balance_payload = balance_payload or _BALANCE_PAYLOAD
    holiday_payload = holiday_payload or _HOLIDAY_TRADING_DAY_PAYLOAD

    def _dispatch(url, *args, **kwargs):
        if url.endswith("/uapi/domestic-stock/v1/trading/inquire-balance"):
            return _FakeResponse(balance_payload)
        if url.endswith("/uapi/domestic-stock/v1/quotations/chk-holiday"):
            return _FakeResponse(holiday_payload)
        raise AssertionError(f"예상치 못한 GET 호출: {url}")

    return _dispatch


@pytest.fixture
def kis_env(monkeypatch):
    monkeypatch.setenv("KIS_PAPER_APP_KEY", "paper-key")
    monkeypatch.setenv("KIS_PAPER_APP_SECRET", "paper-secret")
    monkeypatch.setenv("KIS_PAPER_CANO", "12345678")
    monkeypatch.setenv("KIS_PAPER_ACNT_PRDT_CD", "01")
    monkeypatch.setenv("KIS_LIVE_APP_KEY", "live-key")
    monkeypatch.setenv("KIS_LIVE_APP_SECRET", "live-secret")
    monkeypatch.setenv("KIS_LIVE_CANO", "87654321")
    monkeypatch.setenv("KIS_LIVE_ACNT_PRDT_CD", "01")


def test_missing_credentials_raises_clear_error(monkeypatch):
    monkeypatch.delenv("KIS_PAPER_APP_KEY", raising=False)
    monkeypatch.delenv("KIS_PAPER_APP_SECRET", raising=False)
    monkeypatch.delenv("KIS_PAPER_CANO", raising=False)
    monkeypatch.delenv("KIS_PAPER_ACNT_PRDT_CD", raising=False)
    with pytest.raises(RuntimeError, match="KIS_PAPER"):
        KISBroker(mode="paper")


def test_paper_mode_uses_paper_url(kis_env):
    broker = KISBroker(mode="paper")
    assert broker.base_url == PAPER_BASE_URL
    assert broker.app_key == "paper-key"
    assert broker.cano == "12345678"


def test_live_mode_uses_live_url_and_keys(kis_env):
    """paper/live는 KIS에서 서로 다른 앱키·계좌 쌍이다 — 모드가 바뀌면 도메인과
    자격증명이 함께 바뀌어야 한다(paper 키로 live 도메인을 부르는 사고 방지)."""
    broker = KISBroker(mode="live")
    assert broker.base_url == LIVE_BASE_URL
    assert broker.app_key == "live-key"
    assert broker.cano == "87654321"


def test_default_mode_is_paper(kis_env):
    """CLAUDE.md Phase 4 핵심 결정 1번 — 기본값은 절대 live=True가 아니어야 함."""
    broker = KISBroker()
    assert broker.mode == "paper"
    assert broker.base_url == PAPER_BASE_URL


def test_get_cash_and_get_nav_parse_balance_response(kis_env, monkeypatch):
    monkeypatch.setattr(requests.Session, "post", MagicMock(side_effect=_post_dispatcher()))
    monkeypatch.setattr(requests.Session, "get", MagicMock(side_effect=_get_dispatcher()))
    broker = KISBroker(mode="paper")
    assert broker.get_cash() == pytest.approx(1_000_000.0)
    assert broker.get_nav() == pytest.approx(1_700_000.0)


def test_get_positions_parses_and_filters_zero_qty(kis_env, monkeypatch):
    monkeypatch.setattr(requests.Session, "post", MagicMock(side_effect=_post_dispatcher()))
    monkeypatch.setattr(requests.Session, "get", MagicMock(side_effect=_get_dispatcher()))
    broker = KISBroker(mode="paper")
    positions = broker.get_positions()
    assert positions == {"005930": HeldPosition(qty=10.0, avg_entry_price=70000.0)}
    assert "000660" not in positions  # hldg_qty=0인 과거 보유 종목은 제외


def test_access_token_is_cached_across_calls(kis_env, monkeypatch):
    mock_post = MagicMock(side_effect=_post_dispatcher())
    monkeypatch.setattr(requests.Session, "post", mock_post)
    monkeypatch.setattr(requests.Session, "get", MagicMock(side_effect=_get_dispatcher()))
    broker = KISBroker(mode="paper")

    broker.get_cash()
    broker.get_nav()

    token_calls = [c for c in mock_post.call_args_list if c.args[0].endswith("/oauth2/tokenP")]
    assert len(token_calls) == 1  # 두 번째 호출은 캐시된 토큰을 재사용해야 함


def test_access_token_refetched_after_expiry(kis_env, monkeypatch):
    mock_post = MagicMock(side_effect=_post_dispatcher())
    monkeypatch.setattr(requests.Session, "post", mock_post)
    monkeypatch.setattr(requests.Session, "get", MagicMock(side_effect=_get_dispatcher()))
    broker = KISBroker(mode="paper")

    broker.get_cash()
    broker._token_expires_at = 0.0  # 만료된 것처럼 강제
    broker.get_cash()

    token_calls = [c for c in mock_post.call_args_list if c.args[0].endswith("/oauth2/tokenP")]
    assert len(token_calls) == 2


def test_place_market_order_success_with_integer_qty(kis_env, monkeypatch):
    monkeypatch.setattr(requests.Session, "post", MagicMock(side_effect=_post_dispatcher()))
    broker = KISBroker(mode="paper")

    result = broker.place_market_order("005930", "buy", qty=10.0)

    assert isinstance(result, OrderResult)
    assert result.order_id == "order-1"
    assert result.status == "accepted"
    assert result.submitted_qty == pytest.approx(10.0)
    assert result.submitted_notional is None


def test_place_market_order_sends_integer_qty_string_and_hashkey_header(kis_env, monkeypatch):
    mock_post = MagicMock(side_effect=_post_dispatcher())
    monkeypatch.setattr(requests.Session, "post", mock_post)
    broker = KISBroker(mode="paper")

    broker.place_market_order("005930", "sell", qty=3.0)

    order_call = next(c for c in mock_post.call_args_list if c.args[0].endswith("/order-cash"))
    assert order_call.kwargs["json"]["ORD_QTY"] == "3"
    assert order_call.kwargs["json"]["PDNO"] == "005930"
    assert order_call.kwargs["headers"]["hashkey"] == "hash-1"
    assert order_call.kwargs["headers"]["tr_id"] == "VTTC0801U"  # 모의투자 매도


def test_place_market_order_rejects_notional(kis_env):
    """KIS는 notional 기반 주문 API가 없다 — 조용히 변환하지 않고 명시적으로
    거부해야 execute_kr.py가 이 제약을 놓치지 않는다."""
    broker = KISBroker(mode="paper")
    with pytest.raises(NotImplementedError, match="notional"):
        broker.place_market_order("005930", "buy", notional=100000.0)


def test_place_market_order_rejects_missing_qty(kis_env):
    broker = KISBroker(mode="paper")
    with pytest.raises(ValueError, match="qty"):
        broker.place_market_order("005930", "buy")


@pytest.mark.parametrize("bad_qty", [0.0, -1.0, float("nan"), float("inf")])
def test_place_market_order_rejects_non_positive_or_non_finite_qty(kis_env, bad_qty):
    broker = KISBroker(mode="paper")
    with pytest.raises(ValueError, match="유한한 양수"):
        broker.place_market_order("005930", "buy", qty=bad_qty)


def test_place_market_order_rejects_fractional_qty(kis_env):
    """CLAUDE.md/plan/10 — KIS는 정수 주식 수량만 허용(소수점 매매 불가)."""
    broker = KISBroker(mode="paper")
    with pytest.raises(ValueError, match="정수"):
        broker.place_market_order("005930", "buy", qty=3.5)


def test_place_market_order_business_error_raises_with_message(kis_env, monkeypatch):
    """KIS는 HTTP 200이어도 rt_cd!='0'이면 비즈니스 로직 실패다 — HTTP 상태코드만
    보면 이 실패(잔고부족 등)를 놓친다."""
    bad_order = {"rt_cd": "1", "msg1": "주문가능금액을 초과하였습니다.", "output": {}}
    monkeypatch.setattr(requests.Session, "post", MagicMock(side_effect=_post_dispatcher(order_payload=bad_order)))
    broker = KISBroker(mode="paper")
    with pytest.raises(RuntimeError, match="주문가능금액을 초과"):
        broker.place_market_order("005930", "buy", qty=10.0)


def test_http_error_includes_response_body_detail(kis_env, monkeypatch):
    monkeypatch.setattr(
        requests.Session, "post", MagicMock(return_value=_FakeResponse({"msg1": "인증 실패"}, status_code=403))
    )
    broker = KISBroker(mode="paper")
    with pytest.raises(requests.HTTPError, match="인증 실패"):
        broker.get_cash()


def test_is_market_open_false_on_holiday(kis_env, monkeypatch):
    monkeypatch.setattr(requests.Session, "post", MagicMock(side_effect=_post_dispatcher()))
    monkeypatch.setattr(
        requests.Session,
        "get",
        MagicMock(side_effect=_get_dispatcher(holiday_payload=_HOLIDAY_CLOSED_DAY_PAYLOAD)),
    )
    broker = KISBroker(mode="paper")
    assert broker.is_market_open() is False


def test_is_market_open_checks_time_window_when_trading_day(kis_env, monkeypatch):
    """휴장일이 아니면 시간 경계 판단(_is_within_market_hours)에 위임한다 —
    실제 반환값은 테스트 실행 시각에 따라 달라지므로 bool 타입만 확인하고,
    경계값 자체는 시각을 직접 주입할 수 있는 _is_within_market_hours로 검증한다."""
    monkeypatch.setattr(requests.Session, "post", MagicMock(side_effect=_post_dispatcher()))
    monkeypatch.setattr(requests.Session, "get", MagicMock(side_effect=_get_dispatcher()))
    broker = KISBroker(mode="paper")
    assert isinstance(broker.is_market_open(), bool)


@pytest.mark.parametrize(
    "hour,minute,expected",
    [
        (9, 0, True),
        (15, 30, True),
        (12, 0, True),
        (8, 59, False),
        (15, 31, False),
        (0, 0, False),
    ],
)
def test_is_within_market_hours_boundaries(hour, minute, expected):
    ts = pd.Timestamp("2026-08-20", tz="Asia/Seoul").replace(hour=hour, minute=minute)
    assert _is_within_market_hours(ts) is expected
