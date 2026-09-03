"""execute_us.py 유닛테스트 — 실제 브로커 API 호출 없이 가짜 BrokerAdapter를 쓴다."""
import json

import pytest

import src.live.execute_us as execute_us_module
from src.live.broker.base import BrokerAdapter, OrderResult
from src.live.decide_us import decision_path
from src.live.execute_us import run_execute
from src.live.observation import HeldPosition
from src.live.safety import LiveState, load_state, save_state


class _FakeBroker(BrokerAdapter):
    def __init__(self, positions=None, cash=1000.0, nav=1000.0, market_open=True, open_orders_seq=None):
        super().__init__(mode="paper")
        self._positions = positions or {}
        self._cash = cash
        self._nav = nav
        self._market_open = market_open
        # 폴링 호출마다 하나씩 소비되는 미체결 주문 수 시퀀스(마지막 값은 반복).
        # None 이면 즉시 체결(0)로 동작 — 대부분의 테스트가 이 경우.
        self._open_orders_seq = list(open_orders_seq) if open_orders_seq is not None else None
        self.orders_submitted: list[tuple] = []
        self.settle_polls = 0

    def get_positions(self):
        return dict(self._positions)

    def get_cash(self):
        return self._cash

    def get_nav(self):
        return self._nav

    def place_market_order(self, ticker, side, qty=None, notional=None, client_order_id=None):
        self.orders_submitted.append((ticker, side, qty, notional))
        if side == "buy":
            self._positions[ticker] = HeldPosition(qty=(qty or 1.0), avg_entry_price=100.0)
            self._cash -= notional or 0.0
        else:
            self._positions.pop(ticker, None)
            self._cash += (qty or 0.0) * 100.0
        return OrderResult(ticker=ticker, side=side, status="filled", filled_qty=qty, filled_avg_price=100.0)

    def is_market_open(self):
        return self._market_open

    def count_open_orders(self):
        self.settle_polls += 1
        if self._open_orders_seq is None:
            return 0
        if len(self._open_orders_seq) > 1:
            return self._open_orders_seq.pop(0)
        return self._open_orders_seq[0]

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
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "AAPL", "side": "buy", "notional": 40.0, "market_id": 0}])
    broker = _FakeBroker(market_open=False)

    results = run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert results == []
    assert broker.orders_submitted == []


def test_run_execute_skips_when_already_executed_today(tmp_path):
    """넓은 폴링 cron 윈도우 안에서 같은 날 여러 번 실행될 수 있다 — 이미
    체결이 끝났으면(last_executed_date==오늘) 다시 제출하면 안 된다."""
    save_state(
        tmp_path,
        "us",
        LiveState(
            positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0,
            updated_at="2024-01-01", last_executed_date=str(TARGET_DATE.date()),
        ),
    )
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "AAPL", "side": "buy", "notional": 40.0, "market_id": 0}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)

    results = run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert results == []
    assert broker.orders_submitted == []


def test_run_execute_does_not_skip_on_first_ever_run(tmp_path):
    """load_or_bootstrap_state()는 state.json이 없으면 그 순간을 updated_at으로
    찍어 즉시 저장한다 — 이 updated_at을 '오늘 이미 체결함'의 근거로 쓰면
    그날의 첫(유일한) 실행 자체를 건너뛰게 되는 버그가 있었다(구현 중 실제로
    발견, last_executed_date라는 별도 필드로 분리해 수정). state.json이
    아예 없는 상태(최초 배포 직후)에서 정상적으로 주문이 나가야 한다."""
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "AAPL", "side": "buy", "notional": 40.0, "market_id": 0}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)

    results = run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert len(results) == 1
    assert broker.orders_submitted == [("AAPL", "buy", None, 40.0)]


def test_run_execute_submits_buy_orders(tmp_path):
    save_state(tmp_path, "us", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "AAPL", "side": "buy", "notional": 40.0, "market_id": 0}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)

    results = run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert len(results) == 1
    assert broker.orders_submitted == [("AAPL", "buy", None, 40.0)]


def test_run_execute_sells_using_current_broker_qty_not_notional(tmp_path):
    """SELL 주문은 notional=None(전량 청산) — 실제 수량은 브로커 계좌 조회로
    결정해야 한다(allocation.py 설계 그대로)."""
    save_state(
        tmp_path,
        "us",
        LiveState(
            positions={"AAPL": HeldPosition(qty=7.0, avg_entry_price=100.0)},
            cash=0.0,
            nav=700.0,
            nav_anchor=700.0,
            updated_at="2024-01-01",
        ),
    )
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "AAPL", "side": "sell", "notional": None, "market_id": 0}])
    broker = _FakeBroker(positions={"AAPL": HeldPosition(qty=7.0, avg_entry_price=100.0)}, cash=0.0, nav=700.0)

    run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert broker.orders_submitted == [("AAPL", "sell", 7.0, None)]


def test_run_execute_skips_sell_when_no_longer_held(tmp_path):
    save_state(tmp_path, "us", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "AAPL", "side": "sell", "notional": None, "market_id": 0}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)  # 이미 청산돼 없음

    results = run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert results == []
    assert broker.orders_submitted == []


def test_run_execute_filters_out_non_us_orders(tmp_path):
    save_state(tmp_path, "us", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(
        tmp_path,
        TARGET_DATE,
        [
            {"ticker": "AAPL", "side": "buy", "notional": 50.0, "market_id": 0},  # US
            {"ticker": "005930", "side": "buy", "notional": 50.0, "market_id": 1},  # KR — 걸러져야 함
        ],
    )
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)

    run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    submitted_tickers = {t for t, *_ in broker.orders_submitted}
    assert submitted_tickers == {"AAPL"}


def test_run_execute_updates_state_after_execution_preserving_nav_anchor(tmp_path):
    save_state(tmp_path, "us", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "AAPL", "side": "buy", "notional": 40.0, "market_id": 0}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)

    run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    new_state = load_state(tmp_path, "us")
    assert new_state.nav_anchor == pytest.approx(1000.0)  # 배포 기준선은 절대 안 바뀜
    assert "AAPL" in new_state.positions
    assert new_state.cash == pytest.approx(960.0)  # 1000 - 40(주문 notional)


def test_run_execute_waits_for_fills_before_snapshotting_state(tmp_path):
    """시장가라도 개장 정각엔 즉시 체결되지 않는다 — 체결 전에 state 를
    저장하면 다음날 재구성 체크가 100% 불일치로 오판한다(2026-08-31 실제
    발생: cash 100000 vs 0.36). 주문 제출 후 미체결이 0이 될 때까지 폴링한
    뒤에 스냅샷해야 한다."""
    save_state(tmp_path, "us", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "AAPL", "side": "buy", "notional": 40.0, "market_id": 0}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0, open_orders_seq=[1, 1, 0])

    run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert broker.settle_polls >= 3  # 0을 볼 때까지 실제로 폴링했다


def test_run_execute_warns_when_orders_do_not_settle_in_time(tmp_path):
    """제한시간 내 미체결이 남으면 경고 알림을 보낸다 — state 스냅샷이
    부정확할 수 있고 다음 실행의 재구성 체크가 막힐 수 있으므로."""
    save_state(tmp_path, "us", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "AAPL", "side": "buy", "notional": 40.0, "market_id": 0}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0, open_orders_seq=[2])  # 영원히 2건 미체결

    calls = []
    original = execute_us_module.send_notification
    execute_us_module.send_notification = lambda text, level="info": calls.append((text, level))
    try:
        run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)
    finally:
        execute_us_module.send_notification = original

    assert any(level == "warning" and "미체결" in text for text, level in calls)
    # 미체결이 남았으면 state 를 pending 으로 표시해야 한다 — 이후 폴이 재동기화
    assert load_state(tmp_path, "us").pending_settle is True


def test_subsequent_poll_resyncs_state_after_delayed_fills(tmp_path):
    """execute 가 미체결로 끝나 state.pending_settle=True 로 저장된 뒤, 10분
    간격 다음 폴은 재트레이드는 안 하되 브로커에서 정확한 상태를 다시 읽어
    맞춘다(2026-09-02: SELL 2건이 3분 뒤 체결돼 현금 275->6201, 다음날
    decide 재구성 6% 괴리로 정지한 걸 방지)."""
    save_state(
        tmp_path, "us",
        LiveState(
            positions={}, cash=275.19, nav=99170.0, nav_anchor=100000.0,
            updated_at="2026-09-02", last_executed_date=str(TARGET_DATE.date()), pending_settle=True,
        ),
    )
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "AAPL", "side": "buy", "notional": 40.0, "market_id": 0}])
    # 이제 미체결 0, 계좌엔 지연 체결분이 반영됨
    broker = _FakeBroker(
        positions={"XOM": HeldPosition(qty=24.0, avg_entry_price=159.9)}, cash=6201.73, nav=99176.07
    )

    calls = []
    original = execute_us_module.send_notification
    execute_us_module.send_notification = lambda text, level="info": calls.append((text, level)) or True
    try:
        results = run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)
    finally:
        execute_us_module.send_notification = original

    assert results == [] and broker.orders_submitted == []  # 재트레이드 안 함
    new_state = load_state(tmp_path, "us")
    assert new_state.pending_settle is False
    assert new_state.cash == pytest.approx(6201.73)
    assert new_state.nav_anchor == pytest.approx(100000.0)  # 기준선 보존
    assert any("재동기화" in t for t, _ in calls)


def test_run_execute_rejects_live_mode(tmp_path):
    with pytest.raises(ValueError, match="live"):
        run_execute(mode="live", data_root=tmp_path, target_date=TARGET_DATE)


def test_run_execute_rejects_injected_live_broker_even_if_mode_string_says_paper(tmp_path):
    """mode="paper" 가드는 mode 문자열 인자만 본다 — broker=로 실제로는 live
    모드인 브로커 인스턴스를 주입하면서 mode="paper"만 넘기면 그 가드를
    그대로 우회할 수 있었다(review-loop 2차 재검토로 발견)."""

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
    save_state(tmp_path, "us", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "AAPL", "side": "buy", "notional": 40.0, "market_id": 0}])
    broker = _FakeBroker(positions={}, cash=100.0, nav=100.0)  # 예상 밖으로 현금이 크게 다름

    with pytest.raises(RuntimeError, match="재구성 불일치"):
        run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)


def test_run_execute_blocked_by_concurrent_lock(tmp_path):
    from src.live.safety import acquire_run_lock

    save_state(tmp_path, "us", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)

    with acquire_run_lock(tmp_path, "execute_us"):
        with pytest.raises(RuntimeError, match="이미 진행 중"):
            run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)


def test_run_execute_sends_no_notification_on_routine_no_op_polls(tmp_path):
    """넓은 폴링 윈도우 안에서 시장이 닫혀 있거나 이미 오늘 체결이 끝난
    경우는 10분마다 반복되는 정상 상태다 — 매번 알리면 스팸이 되므로 알림을
    보내면 안 된다."""
    calls = []
    original = execute_us_module.send_notification
    execute_us_module.send_notification = lambda text, level="info": calls.append((text, level)) or True
    try:
        _write_decision(tmp_path, TARGET_DATE, [{"ticker": "AAPL", "side": "buy", "notional": 40.0, "market_id": 0}])
        broker = _FakeBroker(market_open=False)
        run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)
    finally:
        execute_us_module.send_notification = original

    assert calls == []


def test_run_execute_sends_info_notification_after_actually_executing(tmp_path):
    save_state(tmp_path, "us", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "AAPL", "side": "buy", "notional": 40.0, "market_id": 0}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)

    calls = []
    original = execute_us_module.send_notification
    execute_us_module.send_notification = lambda text, level="info": calls.append((text, level)) or True
    try:
        run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)
    finally:
        execute_us_module.send_notification = original

    assert [lvl for _, lvl in calls] == ["info", "info"]  # 시작 핑 + 완료
    assert any("execute 시작" in t for t, _ in calls)
    assert any("execute 완료" in t for t, _ in calls)


def test_run_execute_sends_error_notification_on_reconciliation_failure_and_still_raises(tmp_path):
    save_state(tmp_path, "us", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "AAPL", "side": "buy", "notional": 40.0, "market_id": 0}])
    broker = _FakeBroker(positions={}, cash=100.0, nav=100.0)  # 큰 괴리 -> 재구성 실패

    calls = []
    original = execute_us_module.send_notification
    execute_us_module.send_notification = lambda text, level="info": calls.append((text, level)) or True
    try:
        with pytest.raises(RuntimeError, match="재구성 불일치"):
            run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)
    finally:
        execute_us_module.send_notification = original

    assert len(calls) == 1
    text, level = calls[0]
    assert level == "error"
    assert "execute_us 실패" in text


def test_run_execute_respects_order_cap_even_if_decision_file_is_stale(tmp_path):
    """decide_us.py가 아침에 이미 캡을 적용했지만, 저녁 체결 시점에 NAV가
    바뀌었을 수 있으므로 execute_us.py도 독립적으로 다시 캡을 검증해야
    한다."""
    save_state(tmp_path, "us", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    # 하드캡(NAV의 10%)을 넘는 주문을 결정 파일에 인위적으로 넣음(아침엔 NAV가 훨씬 컸다고 가정)
    _write_decision(tmp_path, TARGET_DATE, [{"ticker": "AAPL", "side": "buy", "notional": 500.0, "market_id": 0}])
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)  # NAV의 10% = 100 < 500

    run_execute(mode="paper", data_root=tmp_path, target_date=TARGET_DATE, broker=broker)

    assert broker.orders_submitted == []  # 캡 초과로 차단돼야 함
