"""execute_kr.py 유닛테스트 — test_live_execute_us.py와 동일 구조(공유
안전장치 검증), market 관련 부분 + KIS 고유 사정(notional->정수 qty 변환,
get_current_price() 조회) 부분만 KR로 교체·추가. 실제 브로커 API 호출 없이
가짜 BrokerAdapter를 쓴다."""
import json

import pytest

import src.live.execute_kr as execute_kr_module
from src.live.broker.base import BrokerAdapter, OrderResult
from src.live.decide_kr import decision_path
from src.live.execute_kr import run_execute
from src.live.observation import HeldPosition
from src.live.safety import LiveState, load_state, save_state

DEFAULT_PRICE = 10.0  # notional=40.0짜리 테스트 주문이 qty=4로 깔끔하게 나뉘도록


class _FakeBroker(BrokerAdapter):
    def __init__(self, positions=None, cash=1000.0, nav=1000.0, market_open=True, prices=None, price_errors=None):
        super().__init__(mode="paper")
        self._positions = positions or {}
        self._cash = cash
        self._nav = nav
        self._market_open = market_open
        self._prices = prices or {}
        self._price_errors = price_errors or {}
        self.orders_submitted: list[tuple] = []
        self.price_lookups: list[str] = []

    def get_positions(self):
        return dict(self._positions)

    def get_cash(self):
        return self._cash

    def get_nav(self):
        return self._nav

    def get_current_price(self, ticker):
        self.price_lookups.append(ticker)
        if ticker in self._price_errors:
            raise self._price_errors[ticker]
        return self._prices.get(ticker, DEFAULT_PRICE)

    def place_market_order(self, ticker, side, qty=None, notional=None, client_order_id=None):
        self.orders_submitted.append((ticker, side, qty, notional))
        price = self._prices.get(ticker, DEFAULT_PRICE)
        if side == "buy":
            self._positions[ticker] = HeldPosition(qty=(qty or 1.0), avg_entry_price=price)
            self._cash -= (qty or 0.0) * price
        else:
            self._positions.pop(ticker, None)
            self._cash += (qty or 0.0) * price
        return OrderResult(ticker=ticker, side=side, status="filled", filled_qty=qty, filled_avg_price=price)

    def is_market_open(self):
        return self._market_open

    def count_open_orders(self):
        return 0  # 이 페이크는 place_market_order 가 즉시 체결로 동작

    def wait_until_orders_settle(self, timeout_s=180.0, poll_s=5.0):
        # 테스트에선 실제 대기 없이 폴링 로직만 검증(타임아웃도 짧게)
        return super().wait_until_orders_settle(timeout_s=0.05, poll_s=0.001)


def _write_decision(data_root, target_date, orders):
    path = decision_path(data_root, target_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_date": str(target_date.date()) if hasattr(target_date, "date") else str(target_date),
        "decided_at": "2024-01-01T07:00:00+09:00",
        "nav": 1000.0,
        "cash": 1000.0,
        "suppress_buys": False,
        "orders": orders,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


import pandas as pd

TARGET_DATE = pd.Timestamp("2024-03-01")


def test_run_execute_does_nothing_when_market_closed(tmp_path):
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "005930", "side": "buy", "notional": 40.0, "market_id": 1}])
    broker = _FakeBroker(market_open=False)

    results = run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert results == []
    assert broker.orders_submitted == []


def test_run_execute_skips_when_already_executed_today(tmp_path):
    """좁은 폴링 cron 윈도우 안에서 같은 날 여러 번 실행될 수 있다 — 이미
    체결이 끝났으면(last_executed_date==오늘) 다시 제출하면 안 된다."""
    save_state(
        tmp_path,
        "kr",
        LiveState(
            positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0,
            updated_at="2024-01-01", last_executed_date=str(TARGET_DATE.date()),
        ),
    )
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "005930", "side": "buy", "notional": 40.0, "market_id": 1}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)

    results = run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert results == []
    assert broker.orders_submitted == []


def test_run_execute_does_not_skip_on_first_ever_run(tmp_path):
    """load_or_bootstrap_state()는 state.json이 없으면 그 순간을 updated_at으로
    찍어 즉시 저장한다 — 이 updated_at을 '오늘 이미 체결함'의 근거로 쓰면
    그날의 첫(유일한) 실행 자체를 건너뛰게 되는 버그가 있었다(execute_us.py
    구현 중 실제로 발견, last_executed_date라는 별도 필드로 분리해 수정).
    state.json이 아예 없는 상태(최초 배포 직후)에서 정상적으로 주문이
    나가야 한다."""
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "005930", "side": "buy", "notional": 40.0, "market_id": 1}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)

    results = run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert len(results) == 1
    assert broker.orders_submitted == [("005930", "buy", 4.0, None)]  # 40.0 // 10.0 = 4주


def test_run_execute_submits_buy_orders_as_integer_qty_using_current_price(tmp_path):
    """KIS는 notional 주문이 없다 — notional을 get_current_price()로 조회한
    체결가로 나눠 정수 수량으로 바꿔서 제출해야 한다(Alpaca는 notional을
    그대로 넘겼지만 KR은 qty를 계산해서 넘김)."""
    save_state(tmp_path, "kr", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "005930", "side": "buy", "notional": 45.0, "market_id": 1}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0, prices={"005930": 10.0})

    results = run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert len(results) == 1
    # 45.0 // 10.0 = 4주(floor — notional을 초과해서 사지 않는 안전한 방향)
    assert broker.orders_submitted == [("005930", "buy", 4.0, None)]
    assert "005930" in broker.price_lookups


def test_run_execute_skips_buy_when_notional_below_one_share_price(tmp_path):
    """이 티커에 배분된 notional이 1주 가격에도 못 미치면(고가주 + 소액
    배분) 주문 없이 조용히 건너뛰어야 한다 — Alpaca는 소수점 매매가 되니
    이 케이스 자체가 없었지만 KIS는 정수 수량만 되므로 새로 생기는 경로다."""
    save_state(tmp_path, "kr", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "005930", "side": "buy", "notional": 40.0, "market_id": 1}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0, prices={"005930": 1_000_000.0})  # 고가주

    results = run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert results == []
    assert broker.orders_submitted == []


def test_run_execute_one_buy_price_lookup_failure_does_not_block_other_orders(tmp_path):
    """get_current_price()는 execute_us.py에는 없던 종목별 추가 네트워크
    호출이다 — 한 종목의 조회 실패가 아직 처리 안 된 다른 매수는 물론,
    포지션을 줄이는 무관한 매도까지 막으면 안 된다(SELL을 항상 허용하는
    킬스위치 철학과 반대 방향이라 review-loop 1차 재검토로 발견)."""
    save_state(
        tmp_path,
        "kr",
        LiveState(
            positions={"000660": HeldPosition(qty=5.0, avg_entry_price=100.0)},
            cash=1000.0,
            nav=1500.0,
            nav_anchor=1500.0,
            updated_at="2024-01-01",
        ),
    )
    _write_decision(
        tmp_path,
        TARGET_DATE,
        [
            {"ticker": "005930", "side": "buy", "notional": 40.0, "market_id": 1},  # 현재가 조회 실패
            {"ticker": "003545", "side": "buy", "notional": 40.0, "market_id": 1},  # 정상 매수
            {"ticker": "000660", "side": "sell", "notional": None, "market_id": 1},  # 무관한 매도
        ],
    )
    broker = _FakeBroker(
        positions={"000660": HeldPosition(qty=5.0, avg_entry_price=100.0)},
        cash=1000.0,
        nav=1500.0,
        price_errors={"005930": RuntimeError("KIS API 일시 장애")},
    )

    calls = []
    original = execute_kr_module.send_notification
    execute_kr_module.send_notification = lambda text, level="info": calls.append((text, level)) or True
    try:
        results = run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)
    finally:
        execute_kr_module.send_notification = original

    submitted_tickers = {t for t, *_ in broker.orders_submitted}
    assert submitted_tickers == {"003545", "000660"}  # 실패한 005930만 제외, 나머지는 정상 제출
    assert len(results) == 2
    warning_texts = [text for text, level in calls if level == "warning"]
    assert any("005930" in text for text in warning_texts)


def test_run_execute_sells_using_current_broker_qty_not_price_lookup(tmp_path):
    """SELL 주문은 notional=None(전량 청산) — 실제 수량은 브로커 계좌 조회로
    결정한다(allocation.py 설계 그대로, Alpaca와 동일). get_current_price()를
    호출할 필요가 없다(가격 조회는 BUY의 notional->qty 변환에만 필요)."""
    save_state(
        tmp_path,
        "kr",
        LiveState(
            positions={"005930": HeldPosition(qty=7.0, avg_entry_price=100.0)},
            cash=0.0,
            nav=700.0,
            nav_anchor=700.0,
            updated_at="2024-01-01",
        ),
    )
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "005930", "side": "sell", "notional": None, "market_id": 1}])
    broker = _FakeBroker(positions={"005930": HeldPosition(qty=7.0, avg_entry_price=100.0)}, cash=0.0, nav=700.0)

    run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert broker.orders_submitted == [("005930", "sell", 7.0, None)]
    assert broker.price_lookups == []


def test_run_execute_skips_sell_when_no_longer_held(tmp_path):
    save_state(tmp_path, "kr", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "005930", "side": "sell", "notional": None, "market_id": 1}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)  # 이미 청산돼 없음

    results = run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert results == []
    assert broker.orders_submitted == []


def test_run_execute_filters_out_non_kr_orders(tmp_path):
    save_state(tmp_path, "kr", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(
        tmp_path,
        TARGET_DATE,
        [
            {"ticker": "005930", "side": "buy", "notional": 50.0, "market_id": 1},  # KR
            {"ticker": "AAPL", "side": "buy", "notional": 50.0, "market_id": 0},  # US — 걸러져야 함
        ],
    )
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)

    run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    submitted_tickers = {t for t, *_ in broker.orders_submitted}
    assert submitted_tickers == {"005930"}


def test_run_execute_updates_state_after_execution_preserving_nav_anchor(tmp_path):
    save_state(tmp_path, "kr", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "005930", "side": "buy", "notional": 40.0, "market_id": 1}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)

    run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    new_state = load_state(tmp_path, "kr")
    assert new_state.nav_anchor == pytest.approx(1000.0)  # 배포 기준선은 절대 안 바뀜
    assert "005930" in new_state.positions
    assert new_state.cash == pytest.approx(960.0)  # 1000 - 4주*10.0(체결가)


def test_run_execute_rejects_live_mode(tmp_path):
    with pytest.raises(ValueError, match="live"):
        run_execute(mode="live", data_root=tmp_path, target_date=TARGET_DATE)


def test_run_execute_rejects_injected_live_broker_even_if_mode_string_says_paper(tmp_path):
    """mode="paper" 가드는 mode 문자열 인자만 본다 — broker=로 실제로는 live
    모드인 브로커 인스턴스를 주입하면서 mode="paper"만 넘기면 그 가드를
    그대로 우회할 수 있었다(execute_us.py review-loop 2차 재검토로 발견된
    것과 동일한 우회 경로라 처음부터 반영)."""

    class _FakeLiveBroker(_FakeBroker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mode = "live"  # 실제로는 live 모드인 척하는 브로커

    broker = _FakeLiveBroker(positions={}, cash=1000.0, nav=1000.0)
    with pytest.raises(ValueError, match="broker.mode"):
        run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)
    assert broker.orders_submitted == []


def test_run_execute_raises_on_missing_decision_file(tmp_path):
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)
    with pytest.raises(FileNotFoundError):
        run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)


def test_run_execute_raises_on_reconciliation_mismatch(tmp_path):
    save_state(tmp_path, "kr", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "005930", "side": "buy", "notional": 40.0, "market_id": 1}])
    broker = _FakeBroker(positions={}, cash=100.0, nav=100.0)  # 예상 밖으로 현금이 크게 다름

    with pytest.raises(RuntimeError, match="재구성 불일치"):
        run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)


def test_run_execute_blocked_by_concurrent_lock(tmp_path):
    from src.live.safety import acquire_run_lock

    save_state(tmp_path, "kr", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)

    with acquire_run_lock(tmp_path, "execute_kr"):
        with pytest.raises(RuntimeError, match="이미 진행 중"):
            run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)


def test_run_execute_sends_no_notification_on_routine_no_op_polls(tmp_path):
    """좁은 폴링 윈도우 안에서 시장이 닫혀 있거나 이미 오늘 체결이 끝난
    경우는 반복되는 정상 상태다 — 매번 알리면 스팸이 되므로 알림을 보내면
    안 된다."""
    calls = []
    original = execute_kr_module.send_notification
    execute_kr_module.send_notification = lambda text, level="info": calls.append((text, level)) or True
    try:
        _write_decision(tmp_path, TARGET_DATE, [{"ticker": "005930", "side": "buy", "notional": 40.0, "market_id": 1}])
        broker = _FakeBroker(market_open=False)
        run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)
    finally:
        execute_kr_module.send_notification = original

    assert calls == []


def test_run_execute_sends_info_notification_after_actually_executing(tmp_path):
    save_state(tmp_path, "kr", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "005930", "side": "buy", "notional": 40.0, "market_id": 1}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)

    calls = []
    original = execute_kr_module.send_notification
    execute_kr_module.send_notification = lambda text, level="info": calls.append((text, level)) or True
    try:
        run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)
    finally:
        execute_kr_module.send_notification = original

    assert len(calls) == 1
    text, level = calls[0]
    assert level == "info"
    assert "execute 완료" in text


def test_run_execute_sends_error_notification_on_reconciliation_failure_and_still_raises(tmp_path):
    save_state(tmp_path, "kr", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "005930", "side": "buy", "notional": 40.0, "market_id": 1}])
    broker = _FakeBroker(positions={}, cash=100.0, nav=100.0)  # 큰 괴리 -> 재구성 실패

    calls = []
    original = execute_kr_module.send_notification
    execute_kr_module.send_notification = lambda text, level="info": calls.append((text, level)) or True
    try:
        with pytest.raises(RuntimeError, match="재구성 불일치"):
            run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)
    finally:
        execute_kr_module.send_notification = original

    assert len(calls) == 1
    text, level = calls[0]
    assert level == "error"
    assert "execute_kr 실패" in text


def test_run_execute_respects_order_cap_even_if_decision_file_is_stale(tmp_path):
    """decide_kr.py가 아침에 이미 캡을 적용했지만, 저녁 체결 시점에 NAV가
    바뀌었을 수 있으므로 execute_kr.py도 독립적으로 다시 캡을 검증해야
    한다."""
    save_state(tmp_path, "kr", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    # 하드캡(NAV의 10%)을 넘는 주문을 결정 파일에 인위적으로 넣음(아침엔 NAV가 훨씬 컸다고 가정)
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "005930", "side": "buy", "notional": 500.0, "market_id": 1}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)  # NAV의 10% = 100 < 500

    run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert broker.orders_submitted == []  # 캡 초과로 차단돼야 함
