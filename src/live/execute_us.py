"""Phase 4 US 트랙 — decide_us.py가 저장한 오늘자 결정을 실제로 체결한다.

plan/10 §B-6/§C-2. US 장 개장 시각(22:30~23:30 KST, DST에 따라 다름)에
cron으로 돈다. 정확한 DST 경계를 손으로 관리하는 cron 두 줄 대신,
`AlpacaBroker.is_market_open()`으로 실행 시점에 직접 확인한다 — 그래서
cron을 조금 더 넓은 창(예: 22:00~23:50 KST, 매 10분)으로 걸어놓아도 장이
실제로 열리기 전까지는 조용히 아무 것도 안 하고 끝난다(DST 전환 두 번의
수동 cron 갱신이 필요 없어짐).

**넓은 폴링 윈도우의 함정**: 그 창 안에서 여러 번 실행되므로(예: 10분마다),
한 번 체결에 성공한 뒤 같은 날 또 실행되면 그대로 두면 같은 결정을 또
제출해 주문이 중복된다 — `state.last_executed_date`(execute_us.py가 성공적으로
주문 사이클을 마친 뒤에만 그날 날짜로 세팅, `state.updated_at`과는 다른
필드 — `updated_at`은 최초 부트스트랩에서도 "지금"으로 찍혀서 그 값을 이
판단에 쓰면 그날의 첫 실행 자체를 건너뛰는 버그가 났었다, `safety.py`의
`LiveState.last_executed_date` docstring 참고)이 이미 오늘(KST) 날짜면
멱등적으로 건너뛴다.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import get_data_root
from src.live.allocation import TargetOrder
from src.live.broker.alpaca import AlpacaBroker
from src.live.broker.base import BrokerAdapter, OrderResult
from src.live.decide_us import MARKET_ID_US, decision_path
from src.live.notify import send_notification
from src.live.safety import LiveState, acquire_run_lock, enforce_order_caps, load_or_bootstrap_state, save_state


def _kst_today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()


def _load_us_orders(path: Path) -> list[TargetOrder]:
    import json

    if not path.exists():
        raise FileNotFoundError(f"오늘자 결정 파일이 없음: {path} — decide_us.py가 먼저 실행됐는지 확인할 것.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        TargetOrder(ticker=o["ticker"], side=o["side"], notional=o["notional"])
        for o in payload["orders"]
        if o["market_id"] == MARKET_ID_US
    ]


def run_execute(
    mode: Literal["paper", "live"] = "paper",
    data_root: Path | None = None,
    target_date: pd.Timestamp | str | None = None,
    broker: BrokerAdapter | None = None,
) -> list[OrderResult]:
    """오늘자 결정 파일(US분만 필터링)을 읽어 브로커에 제출한다. `mode`
    기본값은 반드시 `"paper"`다(CLAUDE.md Phase 4 핵심 결정 1번) — `"live"`를
    실제로 넘기는 건 이번 세션 범위 밖이라 여기서 명시적으로 막는다(코드
    경로 자체는 `AlpacaBroker(mode="live")`로 존재하되, "안 부르기로 한
    약속"에만 의존하지 않기 위함).
    """
    if mode != "paper":
        raise ValueError(
            f"mode={mode!r}는 이번 세션 범위 밖 — live 자동집행은 별도 승인 후에만 "
            "활성화한다(CLAUDE.md Phase 4 핵심 결정 1번)."
        )

    data_root = data_root or get_data_root()
    target_date = pd.Timestamp(target_date) if target_date is not None else _kst_today()
    broker = broker or AlpacaBroker(mode=mode)
    if broker.mode != "paper":
        # 바로 위의 mode!="paper" 가드는 mode 문자열 인자만 본다 — broker=를
        # 통해 실제로는 live 모드로 만들어진 브로커 인스턴스를 주입하면서
        # mode="paper"만 넘기면(예: run_execute(mode="paper",
        # broker=AlpacaBroker(mode="live"))) 그 가드를 그대로 우회해 실거래
        # 브로커가 호출될 수 있었다 — 실제로 쓰이는 broker 객체 자체의
        # mode도 확인해야 한다(review-loop 2차 재검토로 발견).
        raise ValueError(
            f"broker.mode={broker.mode!r}가 paper가 아님 — live 자동집행은 별도 승인 후에만 "
            "활성화한다(CLAUDE.md Phase 4 핵심 결정 1번)."
        )

    # 시장이 닫혀 있거나(넓은 폴링 윈도우 중 대부분) 오늘 이미 체결이 끝난
    # 경우는 10분마다 조용히 반복되는 정상 상태라 매번 알리면 스팸이 된다 —
    # 그래서 executed=True는 실제로 체결 사이클을 밟기 시작한 뒤에만 세팅하고,
    # 아래 완료 알림은 executed일 때만 보낸다(예외는 어느 단계에서 나든 항상
    # 알림 — except 블록 참고).
    executed = False
    results: list[OrderResult] = []
    try:
        with acquire_run_lock(data_root, "execute_us"):
            if not broker.is_market_open():
                return []  # 아직 장이 안 열림 — 넓은 cron 윈도우 전제, 조용히 종료

            state = load_or_bootstrap_state(data_root, "us", broker.get_positions(), broker.get_cash(), broker.get_nav())
            if state.last_executed_date == str(target_date.date()):
                # 넓은 폴링 윈도우 안에서 이미 오늘(KST) 체결이 끝났다 — 그대로
                # 다시 제출하면 주문이 중복된다(모듈 docstring 참고). state.updated_at은
                # load_or_bootstrap_state()의 최초 부트스트랩에서도 "지금"으로 찍히므로
                # 이 판단에 쓰면 안 된다(safety.py의 LiveState.last_executed_date
                # docstring 참고 — 실제로 그렇게 짰다가 그날의 첫 실행을 건너뛰는
                # 버그를 냈었음).
                return []

            executed = True
            current_positions = broker.get_positions()
            current_cash = broker.get_cash()
            nav = broker.get_nav()
            from src.live.safety import check_reconciliation

            # decide_us.py가 그날 아침 이미 한 번 확인했지만, 결정 시각과 체결
            # 시각 사이(최대 16시간 이상)에 수동 개입 등으로 상태가 또 바뀌었을
            # 수 있다 — 실제로 돈이 움직이기 직전이므로 한 번 더 확인한다.
            check_reconciliation(current_positions, current_cash, state)

            orders = _load_us_orders(decision_path(data_root, target_date))
            safe_orders = enforce_order_caps(orders, nav=nav)

            # 스케줄 시작 핑 — 하루 1번(장 개장 확인 + 아직 미체결일 때만 여기
            # 도달). "시작"만 오고 "완료"/"실패"가 없으면 집행 중 조용히 멈춘 것.
            n_buy = sum(1 for o in safe_orders if o.side == "buy")
            n_sell = sum(1 for o in safe_orders if o.side == "sell")
            send_notification(
                f"[US] execute 시작({target_date.date()}): 매수 {n_buy}건/매도 {n_sell}건 집행", level="info"
            )

            for order in safe_orders:
                if order.side == "sell":
                    pos = current_positions.get(order.ticker)
                    if pos is None or pos.qty == 0:
                        continue  # 이미 청산됐거나 애초에 없음(재구성 통과했으니 정상 범위)
                    # allocation.py가 숏 포지션의 SELL은 이미 no-op으로 걸러내므로
                    # (경고만 남김) 여기 도달하는 건 전부 롱 청산이다 — qty>0.
                    result = broker.place_market_order(order.ticker, "sell", qty=pos.qty)
                else:
                    result = broker.place_market_order(order.ticker, "buy", notional=order.notional)
                results.append(result)

            # 시장가라도 개장 정각 opening cross 중엔 즉시 체결되지 않는다 —
            # 여기서 안 기다리면 아래 get_positions()/get_cash()가 체결 전
            # 스냅샷을 잡아 state 에 저장되고, 다음날 재구성 체크가 100%
            # 불일치로 오판한다(2026-08-31 실제 발생). 주문을 실제로 낸
            # 경우에만 기다린다.
            if results:
                unsettled = broker.wait_until_orders_settle()
                if unsettled:
                    send_notification(
                        f"[US] execute({target_date.date()}): {unsettled}건이 제한시간 내 미체결 — "
                        "state 스냅샷이 부정확할 수 있음(다음 실행의 재구성 체크가 막힐 수 있음)",
                        level="warning",
                    )

            final_positions = broker.get_positions()
            final_cash = broker.get_cash()
            final_nav = broker.get_nav()
            save_state(
                data_root,
                "us",
                LiveState(
                    positions=final_positions,
                    cash=final_cash,
                    nav=final_nav,
                    nav_anchor=state.nav_anchor,  # 배포 시점 기준선은 절대 다시 계산하지 않음
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    # 실제 제출한 주문이 0건(전부 no-op)이어도 "오늘 체결 사이클을
                    # 완료했다"는 사실 자체는 기록한다 — 그래야 같은 날 나머지
                    # 폴링에서 반복적으로 재구성 체크/락 경합을 하지 않는다.
                    last_executed_date=str(target_date.date()),
                ),
            )
    except Exception as e:
        # 락은 이미 해제된 뒤다(acquire_run_lock의 finally가 with 블록을 빠져나가며
        # 먼저 실행됨). 재구성 불일치를 포함한 모든 실패 경로가 여기로 모인다
        # (plan/10 §B-5: "재구성 불일치하면... 알림 후 중단").
        send_notification(f"[US] execute_us 실패({target_date.date()}): {e!r}", level="error")
        raise

    if executed:
        n_buy = sum(1 for r in results if r.side == "buy")
        n_sell = sum(1 for r in results if r.side == "sell")
        send_notification(f"[US] execute 완료({target_date.date()}): 매수 {n_buy}건/매도 {n_sell}건 체결", level="info")

    return results


if __name__ == "__main__":
    run_execute()
