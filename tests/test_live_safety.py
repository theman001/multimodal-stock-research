import pytest

from src.live.allocation import TargetOrder
from src.live.observation import HeldPosition
from src.live.safety import (
    LiveState,
    acquire_run_lock,
    check_daily_loss_kill_switch,
    check_reconciliation,
    enforce_order_caps,
    load_or_bootstrap_state,
    load_state,
    save_state,
)


# ---------------------------------------------------------------------- #
# state.json 저장/로드/부트스트랩
# ---------------------------------------------------------------------- #


def test_load_state_returns_none_when_missing(tmp_path):
    assert load_state(tmp_path) is None


def test_save_and_load_state_round_trip(tmp_path):
    state = LiveState(
        positions={"AAPL": HeldPosition(qty=10.0, avg_entry_price=150.0)},
        cash=500.0,
        nav=2000.0,
        nav_anchor=1800.0,
        updated_at="2026-08-20T00:00:00+00:00",
    )
    save_state(tmp_path, state)
    loaded = load_state(tmp_path)

    assert loaded.positions == state.positions
    assert loaded.cash == state.cash
    assert loaded.nav == state.nav
    assert loaded.nav_anchor == state.nav_anchor


def test_save_and_load_state_round_trips_last_executed_date(tmp_path):
    state = LiveState(positions={}, cash=100.0, nav=100.0, nav_anchor=100.0, last_executed_date="2026-08-20")
    save_state(tmp_path, state)
    loaded = load_state(tmp_path)
    assert loaded.last_executed_date == "2026-08-20"


def test_load_state_backward_compatible_with_missing_last_executed_date_field(tmp_path):
    """last_executed_date 필드가 생기기 전에 저장된 state.json도(그 필드가
    아예 없는 JSON) 깨지지 않고 None으로 읽혀야 한다."""
    import json

    path = tmp_path / "live" / "state.json"
    path.parent.mkdir(parents=True)
    old_payload = {"positions": {}, "cash": 1.0, "nav": 1.0, "nav_anchor": 1.0, "updated_at": "2024-01-01"}
    path.write_text(json.dumps(old_payload), encoding="utf-8")

    loaded = load_state(tmp_path)
    assert loaded.last_executed_date is None


def test_bootstrap_creates_state_with_nav_anchor_equal_to_current_nav(tmp_path):
    positions = {"AAPL": HeldPosition(qty=1.0, avg_entry_price=100.0)}
    state = load_or_bootstrap_state(tmp_path, positions, current_cash=900.0, current_nav=1000.0)

    assert state.nav_anchor == pytest.approx(1000.0)
    assert state.nav == pytest.approx(1000.0)
    assert load_state(tmp_path) is not None  # 즉시 저장됐어야 함


def test_bootstrap_does_not_overwrite_existing_nav_anchor(tmp_path):
    """nav_anchor는 최초 배포 시점에만 정해지고 이후 절대 다시 계산되면
    안 된다 — observation.py의 nav_ratio 정의(배포 시점 기준)와 어긋난다."""
    first = load_or_bootstrap_state(tmp_path, {}, current_cash=1000.0, current_nav=1000.0)
    assert first.nav_anchor == pytest.approx(1000.0)

    # 다음 날 NAV가 바뀐 상태로 다시 호출해도 nav_anchor는 그대로여야 함
    second = load_or_bootstrap_state(tmp_path, {}, current_cash=1200.0, current_nav=1200.0)
    assert second.nav_anchor == pytest.approx(1000.0)


# ---------------------------------------------------------------------- #
# 중복 실행 방지 락
# ---------------------------------------------------------------------- #


def test_run_lock_blocks_concurrent_acquisition(tmp_path):
    with acquire_run_lock(tmp_path, "decide_us"):
        with pytest.raises(RuntimeError, match="이미 진행 중"):
            with acquire_run_lock(tmp_path, "decide_us"):
                pass


def test_run_lock_releases_after_context_exits(tmp_path):
    with acquire_run_lock(tmp_path, "decide_us"):
        pass
    with acquire_run_lock(tmp_path, "decide_us"):
        pass  # 두 번째도 정상적으로 획득돼야 함(첫 번째가 이미 해제됨)


def test_run_lock_different_names_do_not_conflict(tmp_path):
    with acquire_run_lock(tmp_path, "decide_us"):
        with acquire_run_lock(tmp_path, "execute_us"):
            pass  # 서로 다른 스크립트 락은 독립적이어야 함


def test_run_lock_reclaims_stale_lock_from_dead_process(tmp_path):
    """이전 실행이 크래시로 락 파일만 남기고 죽었으면(PID가 이미 죽음),
    그 락이 이후 모든 실행을 영구히 막아버리면 안 된다."""
    lock_path = tmp_path / "live" / "decide_us.lock"
    lock_path.parent.mkdir(parents=True)
    # 존재하지 않을 만큼 큰 PID를 흉내(실제로 살아있지 않을 값)
    dead_pid = 2**30
    lock_path.write_text(str(dead_pid), encoding="utf-8")

    with acquire_run_lock(tmp_path, "decide_us"):
        pass  # stale lock을 재사용해 정상적으로 획득돼야 함


def test_run_lock_releases_even_if_body_raises(tmp_path):
    with pytest.raises(ValueError):
        with acquire_run_lock(tmp_path, "decide_us"):
            raise ValueError("작업 중 실패")
    with acquire_run_lock(tmp_path, "decide_us"):
        pass  # 예외가 나도 락은 풀려야 함


# ---------------------------------------------------------------------- #
# 재구성(reconciliation) 체크
# ---------------------------------------------------------------------- #


def test_reconciliation_passes_when_state_matches():
    state = LiveState(positions={"AAPL": HeldPosition(qty=10.0, avg_entry_price=150.0)}, cash=500.0, nav=2000.0)
    check_reconciliation(
        current_positions={"AAPL": HeldPosition(qty=10.0, avg_entry_price=150.0)},
        current_cash=500.0,
        last_state=state,
    )  # 예외 없이 통과해야 함


def test_reconciliation_raises_on_unexpected_cash_drift():
    state = LiveState(positions={}, cash=1000.0, nav=2000.0)
    with pytest.raises(RuntimeError, match="현금 재구성 불일치"):
        check_reconciliation(current_positions={}, current_cash=1000.0 - 500.0, last_state=state)  # NAV의 25% 이탈


def test_reconciliation_raises_on_unexpected_position_change():
    state = LiveState(positions={"AAPL": HeldPosition(qty=10.0, avg_entry_price=150.0)}, cash=500.0, nav=2000.0)
    with pytest.raises(RuntimeError, match="AAPL 포지션 재구성 불일치"):
        check_reconciliation(
            current_positions={"AAPL": HeldPosition(qty=25.0, avg_entry_price=150.0)},  # 수동 개입 등으로 수량이 달라짐
            current_cash=500.0,
            last_state=state,
        )


def test_reconciliation_detects_unexpected_new_position_not_in_last_state():
    state = LiveState(positions={}, cash=2000.0, nav=2000.0)
    with pytest.raises(RuntimeError, match="TSLA 포지션 재구성 불일치"):
        check_reconciliation(
            current_positions={"TSLA": HeldPosition(qty=5.0, avg_entry_price=200.0)},
            current_cash=2000.0,
            last_state=state,
        )


def test_reconciliation_raises_on_non_finite_current_cash():
    """NaN/Inf인 current_cash를 그냥 통과시키면 cash_drift_ratio가 NaN이 되고
    'NaN > tolerance'는 항상 False라 재구성 체크가 조용히 무력화된다."""
    state = LiveState(positions={}, cash=1000.0, nav=2000.0)
    with pytest.raises(RuntimeError, match="유한하지 않아"):
        check_reconciliation(current_positions={}, current_cash=float("nan"), last_state=state)


def test_reconciliation_raises_on_non_finite_position_qty():
    state = LiveState(positions={}, cash=1000.0, nav=2000.0)
    with pytest.raises(RuntimeError, match="유한하지 않아"):
        check_reconciliation(
            current_positions={"AAPL": HeldPosition(qty=float("nan"), avg_entry_price=100.0)},
            current_cash=1000.0,
            last_state=state,
        )


def test_reconciliation_skipped_when_last_nav_non_positive():
    """last_state.nav<=0은 비정상 기준선이라 비율 비교 자체가 무의미하다 —
    0으로 나누기를 피하고 조용히 통과시킨다(다른 방어선이 이런 상태를
    먼저 잡아내야 하는 영역)."""
    state = LiveState(positions={}, cash=0.0, nav=0.0)
    check_reconciliation(current_positions={"AAPL": HeldPosition(qty=100.0, avg_entry_price=1.0)}, current_cash=99999.0, last_state=state)


# ---------------------------------------------------------------------- #
# 일일 손실 킬스위치
# ---------------------------------------------------------------------- #


def test_kill_switch_triggers_on_large_drop():
    assert check_daily_loss_kill_switch(current_nav=940.0, last_known_nav=1000.0, threshold=0.05) is True  # -6%


def test_kill_switch_does_not_trigger_on_small_drop():
    assert check_daily_loss_kill_switch(current_nav=980.0, last_known_nav=1000.0, threshold=0.05) is False  # -2%


def test_kill_switch_does_not_trigger_on_gain():
    assert check_daily_loss_kill_switch(current_nav=1100.0, last_known_nav=1000.0, threshold=0.05) is False


def test_kill_switch_skipped_when_no_baseline():
    assert check_daily_loss_kill_switch(current_nav=1000.0, last_known_nav=0.0) is False


def test_kill_switch_triggers_fail_closed_on_non_finite_current_nav():
    """NaN인 current_nav를 그대로 비교하면 'NaN >= threshold'가 항상 False라
    킬스위치가 조용히 안 걸린다 — 값이 이상하면 차단(True)이 안전한 방향."""
    assert check_daily_loss_kill_switch(current_nav=float("nan"), last_known_nav=1000.0) is True
    assert check_daily_loss_kill_switch(current_nav=float("inf"), last_known_nav=1000.0) is True


# ---------------------------------------------------------------------- #
# 주문 상한(정책 캡 재확인 + 하드 캡)
# ---------------------------------------------------------------------- #


def test_enforce_order_caps_passes_orders_within_policy_cap():
    orders = [TargetOrder(ticker="AAPL", side="buy", notional=50.0)]  # NAV(1000)*0.05=50, 딱 경계
    result = enforce_order_caps(orders, nav=1000.0, max_position_weight=0.05)
    assert len(result) == 1


def test_enforce_order_caps_blocks_order_exceeding_policy_cap():
    """allocation.py가 정상이면 절대 안 나올 상황이지만, 방어적 이중 체크로
    막아야 한다(실제로 주문을 차단하는지 확인하는 핵심 테스트)."""
    orders = [TargetOrder(ticker="AAPL", side="buy", notional=200.0)]  # NAV(1000)*0.05=50 초과
    result = enforce_order_caps(orders, nav=1000.0, max_position_weight=0.05)
    assert result == []


def test_enforce_order_caps_blocks_order_exceeding_hard_cap_even_if_policy_cap_is_loose():
    """max_position_weight를 실수로 크게 넘겨도(예: 설정 오류) 하드 상한이
    독립적인 마지막 방어선 역할을 해야 한다."""
    orders = [TargetOrder(ticker="AAPL", side="buy", notional=150.0)]  # NAV(1000)의 15% > 하드캡 10%
    result = enforce_order_caps(orders, nav=1000.0, max_position_weight=0.5)  # 정책 캡은 느슨하게(50%) 설정
    assert result == []


def test_enforce_order_caps_never_blocks_sell_orders():
    orders = [TargetOrder(ticker="AAPL", side="sell", notional=None)]
    result = enforce_order_caps(orders, nav=1000.0)
    assert len(result) == 1


def test_enforce_order_caps_preserves_valid_orders_alongside_blocked_ones():
    orders = [
        TargetOrder(ticker="AAPL", side="buy", notional=30.0),  # 통과
        TargetOrder(ticker="TSLA", side="buy", notional=500.0),  # 차단
        TargetOrder(ticker="MSFT", side="sell", notional=None),  # 통과
    ]
    result = enforce_order_caps(orders, nav=1000.0, max_position_weight=0.05)
    assert {o.ticker for o in result} == {"AAPL", "MSFT"}


def test_enforce_order_caps_blocks_all_buys_on_non_finite_nav():
    """NaN인 nav로 캡을 계산하면 hard_cap/policy_cap도 NaN이 되고
    'notional > NaN'은 항상 False라 이 함수가 아무것도 못 거르게 된다 —
    BUY는 전부 차단, SELL은 그대로 통과가 안전한 방향."""
    orders = [
        TargetOrder(ticker="AAPL", side="buy", notional=10.0),
        TargetOrder(ticker="MSFT", side="sell", notional=None),
    ]
    result = enforce_order_caps(orders, nav=float("nan"))
    assert [o.ticker for o in result] == ["MSFT"]
