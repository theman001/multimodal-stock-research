"""AlpacaBroker 유닛테스트 — 실제 Alpaca API 호출 없이 requests.Session.get/post를
mock으로 대체한다(plan/10 §C-3: "브로커 어댑터 테스트는 실제 API 호출 금지",
CLAUDE.md "실행 통제": requirements.txt에 없는 외부 네트워크 호출은 승인 필요)."""
from unittest.mock import MagicMock

import pytest
import requests

from src.live.broker.alpaca import AlpacaBroker, LIVE_BASE_URL, PAPER_BASE_URL
from src.live.broker.base import OrderResult
from src.live.observation import HeldPosition


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


@pytest.fixture
def paper_env(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER_API_KEY_ID", "paper-key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "paper-secret")
    monkeypatch.setenv("ALPACA_LIVE_API_KEY_ID", "live-key")
    monkeypatch.setenv("ALPACA_LIVE_SECRET_KEY", "live-secret")


def test_missing_credentials_raises_clear_error(monkeypatch):
    monkeypatch.delenv("ALPACA_PAPER_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ALPACA_PAPER"):
        AlpacaBroker(mode="paper")


def test_paper_mode_uses_paper_url_and_keys(paper_env):
    broker = AlpacaBroker(mode="paper")
    assert broker.base_url == PAPER_BASE_URL
    assert broker._session.headers["APCA-API-KEY-ID"] == "paper-key"
    assert broker._session.headers["APCA-API-SECRET-KEY"] == "paper-secret"


def test_live_mode_uses_live_url_and_keys(paper_env):
    """paper/live는 Alpaca에서 서로 다른 키 쌍이다 — 모드가 바뀌면 URL과 키가
    함께 바뀌어야 한다(paper 키로 live 엔드포인트를 부르는 사고 방지)."""
    broker = AlpacaBroker(mode="live")
    assert broker.base_url == LIVE_BASE_URL
    assert broker._session.headers["APCA-API-KEY-ID"] == "live-key"
    assert broker._session.headers["APCA-API-SECRET-KEY"] == "live-secret"


def test_default_mode_is_paper(paper_env):
    """CLAUDE.md Phase 4 핵심 결정 1번 — 기본값은 절대 live=True가 아니어야 함."""
    broker = AlpacaBroker()
    assert broker.mode == "paper"
    assert broker.base_url == PAPER_BASE_URL


def test_get_positions_parses_long_and_short_with_correct_sign(paper_env, monkeypatch):
    payload = [
        {"symbol": "AAPL", "qty": "10", "avg_entry_price": "150.0", "side": "long"},
        {"symbol": "TSLA", "qty": "5", "avg_entry_price": "200.0", "side": "short"},
    ]
    mock_get = MagicMock(return_value=_FakeResponse(payload))
    monkeypatch.setattr(requests.Session, "get", mock_get)

    broker = AlpacaBroker(mode="paper")
    positions = broker.get_positions()

    assert positions["AAPL"] == HeldPosition(qty=10.0, avg_entry_price=150.0)
    assert positions["TSLA"] == HeldPosition(qty=-5.0, avg_entry_price=200.0)
    # mock이 URL과 무관하게 같은 payload를 돌려주므로, URL 자체를 확인하지
    # 않으면 엉뚱한 엔드포인트(예: /account)를 불러도 이 테스트는 그냥
    # 통과해버린다 — 실제로 맞는 엔드포인트를 불렀는지까지 검증한다
    # (review-loop 4차 재검토로 발견, "테스트 자체의 신뢰성" 관점).
    assert mock_get.call_args.args[0].endswith("/v2/positions")


def test_get_positions_forces_sign_from_side_field_even_if_qty_string_already_negative(paper_env, monkeypatch):
    """Alpaca API 버전에 따라 qty 문자열 자체의 부호 관례가 다를 수 있다는
    불확실성에 기대지 않고, side 필드로 부호를 직접 강제한다 — qty 문자열이
    이미 음수로 와도(예: 다른 버전의 API) 결과가 일관돼야 한다."""
    payload = [{"symbol": "TSLA", "qty": "-5", "avg_entry_price": "200.0", "side": "short"}]
    monkeypatch.setattr(requests.Session, "get", MagicMock(return_value=_FakeResponse(payload)))

    broker = AlpacaBroker(mode="paper")
    positions = broker.get_positions()

    assert positions["TSLA"] == HeldPosition(qty=-5.0, avg_entry_price=200.0)


def test_get_cash_parses_account_response(paper_env, monkeypatch):
    mock_get = MagicMock(return_value=_FakeResponse({"cash": "1234.56", "portfolio_value": "9999.0"}))
    monkeypatch.setattr(requests.Session, "get", mock_get)
    broker = AlpacaBroker(mode="paper")
    assert broker.get_cash() == pytest.approx(1234.56)
    assert mock_get.call_args.args[0].endswith("/v2/account")


def test_get_nav_parses_portfolio_value(paper_env, monkeypatch):
    mock_get = MagicMock(return_value=_FakeResponse({"cash": "1234.56", "portfolio_value": "9999.0"}))
    monkeypatch.setattr(requests.Session, "get", mock_get)
    broker = AlpacaBroker(mode="paper")
    assert broker.get_nav() == pytest.approx(9999.0)
    assert mock_get.call_args.args[0].endswith("/v2/account")


def test_is_market_open_parses_clock_response(paper_env, monkeypatch):
    mock_get = MagicMock(return_value=_FakeResponse({"is_open": True}))
    monkeypatch.setattr(requests.Session, "get", mock_get)
    broker = AlpacaBroker(mode="paper")
    assert broker.is_market_open() is True
    assert mock_get.call_args.args[0].endswith("/v2/clock")


def test_place_market_order_with_notional(paper_env, monkeypatch):
    mock_post = MagicMock(
        return_value=_FakeResponse(
            {"id": "order-1", "status": "accepted", "filled_qty": "0", "filled_avg_price": None}
        )
    )
    monkeypatch.setattr(requests.Session, "post", mock_post)

    broker = AlpacaBroker(mode="paper")
    result = broker.place_market_order("AAPL", "buy", notional=500.0)

    assert isinstance(result, OrderResult)
    assert result.order_id == "order-1"
    assert result.status == "accepted"
    assert result.submitted_notional == 500.0
    assert result.submitted_qty is None
    sent_body = mock_post.call_args.kwargs["json"]
    assert sent_body["notional"] == "500.0"
    assert "qty" not in sent_body
    assert mock_post.call_args.args[0].endswith("/v2/orders")


def test_place_market_order_with_qty(paper_env, monkeypatch):
    mock_post = MagicMock(
        return_value=_FakeResponse({"id": "order-2", "status": "filled", "filled_qty": "3", "filled_avg_price": "101.5"})
    )
    monkeypatch.setattr(requests.Session, "post", mock_post)

    broker = AlpacaBroker(mode="paper")
    result = broker.place_market_order("TSLA", "sell", qty=3.0)

    assert result.filled_qty == pytest.approx(3.0)
    assert result.filled_avg_price == pytest.approx(101.5)
    sent_body = mock_post.call_args.kwargs["json"]
    assert sent_body["qty"] == "3.0"
    assert "notional" not in sent_body


@pytest.mark.parametrize("qty,notional", [(None, None), (1.0, 100.0)])
def test_place_market_order_requires_exactly_one_of_qty_or_notional(paper_env, qty, notional):
    broker = AlpacaBroker(mode="paper")
    with pytest.raises(ValueError, match="qty/notional"):
        broker.place_market_order("AAPL", "buy", qty=qty, notional=notional)


@pytest.mark.parametrize("bad_notional", [0.0, -1.0, float("nan"), float("inf")])
def test_place_market_order_rejects_non_positive_or_non_finite_notional(paper_env, bad_notional):
    """0 이하/NaN/Inf인 notional을 그대로 브로커에 보내면 애매한 에러로
    거부되거나(디버깅 어려움), 극단적으로는 반올림 후 사실상 $0 주문이
    나갈 수 있다 — 호출자 실수를 여기서 먼저 막아야 한다(review-loop
    2차 재검토로 발견)."""
    broker = AlpacaBroker(mode="paper")
    with pytest.raises(ValueError, match="유한한 양수"):
        broker.place_market_order("AAPL", "buy", notional=bad_notional)


def test_http_error_propagates(paper_env, monkeypatch):
    monkeypatch.setattr(requests.Session, "get", MagicMock(return_value=_FakeResponse({}, status_code=500)))
    broker = AlpacaBroker(mode="paper")
    with pytest.raises(requests.HTTPError):
        broker.get_cash()


def test_http_error_includes_response_body_detail(paper_env, monkeypatch):
    """resp.raise_for_status()만 쓰면 상태 코드만 보이고 Alpaca가 실패
    사유로 돌려주는 JSON message는 사라진다 — 실주문 실패의 원인(잔고
    부족, 잘못된 심볼 등)을 알 수 없게 되는 게 위험하다(review-loop 1차
    재검토로 발견)."""
    monkeypatch.setattr(
        requests.Session,
        "post",
        MagicMock(return_value=_FakeResponse({"code": 40310000, "message": "insufficient buying power"}, status_code=403)),
    )
    broker = AlpacaBroker(mode="paper")
    with pytest.raises(requests.HTTPError, match="insufficient buying power"):
        broker.place_market_order("AAPL", "buy", notional=100.0)


def test_get_positions_translates_alpaca_dot_symbol_to_our_dash_ticker(paper_env, monkeypatch):
    """이 저장소의 실제 120종목 유니버스에 'BRK-B'(yfinance 관례, 대시)가
    있는데 Alpaca API는 같은 종목을 'BRK.B'(점)로 돌려준다 — 변환 없이
    쓰면 이 포지션이 어떤 유니버스 티커와도 안 맞아 observation.py의
    unknown_tickers 검증에 걸린다(review-loop 5차 재검토로 발견, 실제
    data/checkpoints/rl_ticker_universe.json 확인으로 검증됨)."""
    payload = [{"symbol": "BRK.B", "qty": "2", "avg_entry_price": "300.0", "side": "long"}]
    monkeypatch.setattr(requests.Session, "get", MagicMock(return_value=_FakeResponse(payload)))

    broker = AlpacaBroker(mode="paper")
    positions = broker.get_positions()

    assert "BRK-B" in positions
    assert "BRK.B" not in positions


def test_place_market_order_translates_our_dash_ticker_to_alpaca_dot_symbol(paper_env, monkeypatch):
    mock_post = MagicMock(return_value=_FakeResponse({"id": "order-5", "status": "accepted"}))
    monkeypatch.setattr(requests.Session, "post", mock_post)

    broker = AlpacaBroker(mode="paper")
    result = broker.place_market_order("BRK-B", "buy", notional=100.0)

    sent_body = mock_post.call_args.kwargs["json"]
    assert sent_body["symbol"] == "BRK.B"
    # OrderResult.ticker는 우리 표준 표기를 그대로 유지해야 한다(호출자가
    # 넘긴 값과 응답의 ticker가 일치해야 나중에 대조하기 쉽다).
    assert result.ticker == "BRK-B"


def test_get_positions_rejects_unrecognized_side_value(paper_env, monkeypatch):
    """side가 없거나 'long'/'short'가 아니면 조용히 long 취급하는 게 가장
    위험하다 — 진짜 숏 포지션을 롱으로 잘못 분류하면 손익 방향이 통째로
    반대로 계산된다. 모르는 값은 막아야 한다(review-loop 1차 재검토로
    발견)."""
    payload = [{"symbol": "AAPL", "qty": "10", "avg_entry_price": "150.0", "side": "unknown"}]
    monkeypatch.setattr(requests.Session, "get", MagicMock(return_value=_FakeResponse(payload)))

    broker = AlpacaBroker(mode="paper")
    with pytest.raises(ValueError, match="side"):
        broker.get_positions()


def test_place_market_order_rounds_notional_to_cents(paper_env, monkeypatch):
    """allocation.py의 equal_share는 그냥 나눗셈이라 333.33333333333337처럼
    부동소수점 정밀도가 그대로 남는 값이 나오기 쉽다 — 달러 금액을 센트
    단위로 반올림해서 보내야 한다(review-loop 2차 재검토로 발견)."""
    mock_post = MagicMock(return_value=_FakeResponse({"id": "order-4", "status": "accepted"}))
    monkeypatch.setattr(requests.Session, "post", mock_post)

    broker = AlpacaBroker(mode="paper")
    result = broker.place_market_order("AAPL", "buy", notional=333.33333333333337)

    sent_body = mock_post.call_args.kwargs["json"]
    assert sent_body["notional"] == "333.33"
    # OrderResult에도 원본이 아니라 실제로 전송한(반올림된) 값이 남아야
    # 한다 — 아니면 "기록상 신청 금액"과 "실제 전송 금액"이 달라져 나중에
    # 리포트를 볼 때 혼란스럽다(review-loop 3차 재검토로 발견).
    assert result.submitted_notional == pytest.approx(333.33)


def test_place_market_order_rejects_notional_that_rounds_down_to_zero(paper_env):
    """notional=0.004는 반올림 전엔 양수라 원본값만 검증하면 통과하지만,
    센트로 반올림하면 "0.00"이 돼 사실상 $0 주문이 나간다 — 검증은 반올림
    이후 값으로 해야 한다(review-loop 3차 재검토로 발견, 직전 라운드
    자신의 수정에 남아있던 구멍)."""
    broker = AlpacaBroker(mode="paper")
    with pytest.raises(ValueError, match="반올림 후"):
        broker.place_market_order("AAPL", "buy", notional=0.004)


def test_place_market_order_passes_through_client_order_id(paper_env, monkeypatch):
    """호출자(향후 execute_us.py)가 안전하게 재시도하려면 client_order_id를
    지정할 수 있어야 한다 — 이 계층은 재시도하지 않지만, 그 재시도를 안전하게
    만드는 고리는 열어둬야 한다(review-loop 1차 재검토로 발견)."""
    mock_post = MagicMock(return_value=_FakeResponse({"id": "order-3", "status": "accepted"}))
    monkeypatch.setattr(requests.Session, "post", mock_post)

    broker = AlpacaBroker(mode="paper")
    broker.place_market_order("AAPL", "buy", notional=100.0, client_order_id="decide-2026-08-20-AAPL")

    sent_body = mock_post.call_args.kwargs["json"]
    assert sent_body["client_order_id"] == "decide-2026-08-20-AAPL"
