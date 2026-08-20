"""Phase 4 공유 — 무인 자동매매의 최소 방어선(plan/10 §B-5).

이 모듈의 체크는 전부 "주문을 막는" 방향으로만 작동한다 — allocation.py의
배분 로직이 이미 올바르다고 믿지 않고 방어적으로 이중 확인한다. 상태
(`LiveState`)는 `${DATA_ROOT}/live/state_{track}.json`에 원자적으로 저장한다
(`progress_log.json`과 동일한 tmp-then-rename 패턴, CLAUDE.md "진행 로그
프로토콜" 참고).

**`track`은 필수 파라미터다("us"/"kr").** plan/10 §B-5 원안의 pseudocode는
단일 `${DATA_ROOT}/live/state.json`이었지만(마치 decide_us.py 도입 전
`${DATA_ROOT}/live/decisions/{date}.json`이 시장 구분 없던 것과 동일한
설계 공백), KR/US가 물리적으로 분리된 계좌(원화 KIS vs 달러 Alpaca,
Phase 4 핵심 결정 4번)를 각자 독립적으로 운영한다는 전제 자체가 두 트랙이
같은 state 파일을 공유하면 안 된다는 뜻이다 — 공유하면 한쪽 트랙의
NAV/현금/포지션(KRW)이 다른 쪽(USD)의 재구성 체크·nav_anchor·킬스위치
기준선을 그대로 덮어써서, 두 트랙 모두 매 실행마다 재구성 불일치로
막히거나(통화 단위 자체가 다른 숫자를 비교) 킬스위치가 값이 안 맞는 채로
엉뚱하게 발동한다. decide_us.py/execute_us.py가 이미 락 이름("decide_us"/
"execute_us")과 결정 파일명("us_{date}.json")에 적용한 것과 동일한 원칙을
state 파일에도 적용한다(KR 트랙 세션이 이 모듈을 그대로 재사용할 예정이라
그 시점에 실제로 충돌하기 전에 여기서 막음 — 다른 세션과의 간섭 여부를
점검하다 발견).

기본값은 어디에도 `live=True`가 없다(Phase 4 핵심 결정 1번) — 이 모듈
자체는 브로커를 직접 호출하지 않으므로 mode 파라미터가 없다.
"""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.live.allocation import TargetOrder
from src.live.observation import HeldPosition
from src.rl.trading_env import MAX_POSITION_WEIGHT

# allocation.py가 이미 MAX_POSITION_WEIGHT(0.05)를 지키므로 이 값은 그보다
# 훨씬 큰 "그래도 이걸 넘으면 뭔가 심각하게 잘못된 것" 하드 상한이다 — 정책
# 캡 재확인과는 다른, 독립적인 안전망(plan/10 §B-5 "1회 주문 금액이 NAV
# 대비 하드 상한(예: 10%) 초과 시 차단").
HARD_ORDER_NOTIONAL_CAP_RATIO = 0.10
DAILY_LOSS_KILL_SWITCH_RATIO = 0.05  # 전일 대비 NAV가 이 비율 이상 하락하면 신규 BUY 중단
RECONCILIATION_TOLERANCE_RATIO = 0.02  # 마지막 기록 NAV 대비 이 비율 넘게 브로커 상태가 다르면 중단


@dataclass
class LiveState:
    """마지막으로 확인된(보통 execute_*.py가 실제 주문을 낸 직후) 계좌 상태 —
    재구성 체크/일일 손실 킬스위치의 기준선. `nav_anchor`는 배포(최초 부트스트랩)
    시점 NAV로 이후 절대 갱신하지 않는다(observation.py의 nav_ratio 정의와
    일치시키기 위함).

    `last_executed_date`는 `updated_at`과 별개 필드다 — `updated_at`은
    `load_or_bootstrap_state()`의 최초 부트스트랩에서도 "지금"으로 찍히므로,
    이 값을 "오늘 이미 체결했는가"의 근거로 쓰면 그날 아침 처음 부트스트랩된
    상태를 "오늘 이미 실행됨"으로 착각해 그날의 첫 실행을 건너뛰게 된다
    (execute_us.py 구현 중 실제로 발견한 버그) — `last_executed_date`는
    execute_us.py가 주문을 실제로 제출한 뒤에만 그날 날짜(KST, "YYYY-MM-DD")로
    세팅한다."""

    positions: dict[str, HeldPosition] = field(default_factory=dict)
    cash: float = 0.0
    nav: float = 0.0
    nav_anchor: float = 1.0
    updated_at: str | None = None
    last_executed_date: str | None = None


def _state_path(data_root: Path, track: str) -> Path:
    return data_root / "live" / f"state_{track}.json"


def load_state(data_root: Path, track: str) -> LiveState | None:
    path = _state_path(data_root, track)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    positions = {t: HeldPosition(**p) for t, p in payload["positions"].items()}
    return LiveState(
        positions=positions,
        cash=payload["cash"],
        nav=payload["nav"],
        nav_anchor=payload["nav_anchor"],
        updated_at=payload["updated_at"],
        # .get(): 이 필드가 생기기 전에 저장된 state.json과의 하위호환.
        last_executed_date=payload.get("last_executed_date"),
    )


def save_state(data_root: Path, track: str, state: LiveState) -> None:
    path = _state_path(data_root, track)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "positions": {t: {"qty": p.qty, "avg_entry_price": p.avg_entry_price} for t, p in state.positions.items()},
        "cash": state.cash,
        "nav": state.nav,
        "nav_anchor": state.nav_anchor,
        "updated_at": state.updated_at,
        "last_executed_date": state.last_executed_date,
    }
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_or_bootstrap_state(
    data_root: Path, track: str, current_positions: dict[str, HeldPosition], current_cash: float, current_nav: float
) -> LiveState:
    """`state_{track}.json`이 있으면 그대로 읽어 반환한다(절대 여기서 갱신하지
    않음 — 갱신은 execute_*.py가 실제 주문 이후에만 한다). 없으면(최초 배포)
    지금 이 순간을 기준선으로 새로 만들어 즉시 저장한다 — nav_anchor는 이때
    한 번만 정해지고 이후 다시는 안 바뀐다."""
    state = load_state(data_root, track)
    if state is not None:
        return state
    state = LiveState(
        positions=current_positions,
        cash=current_cash,
        nav=current_nav,
        nav_anchor=current_nav,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    save_state(data_root, track, state)
    return state


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 존재는 하는데 시그널 권한이 없음 -> 살아있다고 보수적으로 간주
    return True


@contextmanager
def acquire_run_lock(data_root: Path, name: str):
    """중복 실행 방지 — `${DATA_ROOT}/live/{name}.lock`에 현재 PID를 남긴다.
    기존 락이 있어도 그 PID가 이미 죽어 있으면(예: 이전 실행이 크래시로
    락 파일만 남기고 죽음) stale lock으로 판단해 재사용한다 — 그렇지 않으면
    한 번의 비정상 종료가 이후 모든 실행을 영구히 막아버린다."""
    lock_path = data_root / "live" / f"{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            existing_pid = int(lock_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            existing_pid = None
        if existing_pid is not None and _pid_alive(existing_pid):
            raise RuntimeError(f"{name} 실행이 이미 진행 중(PID {existing_pid}, lock={lock_path})")

    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def check_reconciliation(
    current_positions: dict[str, HeldPosition],
    current_cash: float,
    last_state: LiveState,
    tolerance: float = RECONCILIATION_TOLERANCE_RATIO,
) -> None:
    """브로커가 보고한 현재 포지션/현금이 마지막으로 기록한 상태와 크게
    다르면(수동 개입/부분체결/기업행동 등) 주문 없이 막는다(예외 발생).
    `last_state.nav<=0`(비정상 기준선)이면 비율 비교가 무의미하므로 건너뛴다."""
    if not np.isfinite(current_cash):
        # NaN/Inf인 current_cash를 그냥 통과시키면 아래 cash_drift_ratio가
        # NaN이 되고, "NaN > tolerance"는 항상 False라 이 함수의 존재
        # 이유(재구성 불일치를 걸러내는 것) 자체가 조용히 무력화된다 —
        # 이 함수는 안전장치라 애매하면 통과가 아니라 차단이 맞는 방향이다
        # (review-loop 4차 재검토로 발견, enforce_order_caps/kill switch에도
        # 같은 문제가 있어 함께 수정).
        raise RuntimeError(f"current_cash가 유한하지 않아 재구성 확인 불가: {current_cash!r}")
    for ticker, pos in current_positions.items():
        if not np.isfinite(pos.qty):
            raise RuntimeError(f"{ticker}의 현재 수량이 유한하지 않아 재구성 확인 불가: {pos.qty!r}")

    if last_state.nav <= 0:
        return

    cash_drift_ratio = abs(current_cash - last_state.cash) / last_state.nav
    if cash_drift_ratio > tolerance:
        raise RuntimeError(
            f"현금 재구성 불일치: 마지막 기록 {last_state.cash:.2f}, 현재 {current_cash:.2f} "
            f"(NAV 대비 {cash_drift_ratio:.2%} 차이, 허용치 {tolerance:.2%})"
        )

    all_tickers = set(current_positions) | set(last_state.positions)
    for ticker in all_tickers:
        cur = current_positions.get(ticker)
        prev = last_state.positions.get(ticker)
        cur_qty = cur.qty if cur is not None else 0.0
        prev_qty = prev.qty if prev is not None else 0.0
        if abs(cur_qty - prev_qty) > 1e-6:
            raise RuntimeError(
                f"{ticker} 포지션 재구성 불일치: 마지막 기록 qty={prev_qty}, 현재 qty={cur_qty}"
            )


def check_daily_loss_kill_switch(
    current_nav: float, last_known_nav: float, threshold: float = DAILY_LOSS_KILL_SWITCH_RATIO
) -> bool:
    """전일 대비 NAV가 threshold 이상 하락했으면 True(신규 BUY를 중단해야
    함)를 반환한다. SELL은 이 킬스위치의 대상이 아니다(포지션 정리는 손실
    상황에서 오히려 허용해야 하는 방향). `last_known_nav<=0`(기준선 없음 —
    최초 실행 등)이면 판단할 수 없으므로 False(중단 안 함)를 반환한다."""
    if not np.isfinite(current_nav):
        # NaN인 current_nav를 그대로 비교하면 "NaN >= threshold"가 항상
        # False라 킬스위치가 조용히 안 걸린다 — 이 함수는 안전장치라 값이
        # 이상하면 "모르니까 통과"가 아니라 "모르니까 차단"이 맞는 방향이다
        # (review-loop 4차 재검토로 발견).
        return True
    if last_known_nav <= 0:
        return False
    drop_ratio = (last_known_nav - current_nav) / last_known_nav
    return drop_ratio >= threshold


def enforce_order_caps(
    orders: list[TargetOrder], nav: float, max_position_weight: float = MAX_POSITION_WEIGHT
) -> list[TargetOrder]:
    """BUY 주문의 notional이 두 상한(정책 캡 재확인 + 하드 상한) 안에 있는지
    검증한다. allocation.py가 이미 `max_position_weight`를 지키므로 정상
    상황에서는 이 체크가 실제로 걸릴 일이 없어야 한다 — 그래도 방어적으로
    다시 확인한다는 게 이 함수의 취지다. 상한을 넘는 주문은 줄이지 않고
    통째로 제외한다(부분 체결로 배분 의도가 왜곡되는 것보다 안전).
    """
    if not np.isfinite(nav):
        # NaN인 nav로 캡을 계산하면 hard_cap/policy_cap도 NaN이 되고,
        # "notional > NaN"은 항상 False라 이 함수(캡을 강제하는 게 유일한
        # 존재 이유)가 조용히 아무것도 안 거르는 상태가 된다 — BUY는 전부
        # 차단하는 게 안전한 방향이다(review-loop 4차 재검토로 발견).
        return [o for o in orders if o.side != "buy"]

    hard_cap = HARD_ORDER_NOTIONAL_CAP_RATIO * nav
    policy_cap = max_position_weight * nav
    # 부동소수점 비교 흔들림 방지용 여유(allocation.py의 min()이 만드는 값과
    # 이론상 정확히 같아야 하지만, 자릿수 오차로 아주 근소하게 넘는 경우까지
    # 걸러내면 안 됨).
    epsilon = 1e-9

    safe_orders = []
    for order in orders:
        if order.side != "buy":
            safe_orders.append(order)
            continue
        if order.notional > policy_cap + epsilon or order.notional > hard_cap + epsilon:
            continue
        safe_orders.append(order)
    return safe_orders
