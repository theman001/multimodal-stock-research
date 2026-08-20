"""Phase 4 KR 트랙 — decide_kr.py가 저장한 오늘자 결정을 실제로 체결한다.

plan/10 §B-6/§C-2, execute_us.py와 동일 구조(공유 안전장치 재사용, market
관련 부분만 KR로 교체) + KR 고유 사정 하나가 추가된다: **KIS는 notional
(금액) 기반 주문 API가 없다**(broker/kis.py 모듈 docstring 참고) — BUY
주문은 `allocation.py`가 만든 notional을 그대로 못 쓰고, 이 파일이
`KISBroker.get_current_price()`로 체결 직전 시세를 조회해 정수 수량으로
바꾼 뒤 제출한다(2026-08-20 사용자 결정 — 가격 소스는 관측 시점의 종가가
아니라 실시간 조회). SELL은 US와 동일하게 브로커가 보고하는 현재 보유
수량을 그대로 쓴다(원래도 notional이 아니라 qty 기반이었음).

KR장은 09:00 KST 정시 개장, DST 없음(미국과 달리 개장 시각이 연중 고정) —
그래도 US와 동일하게 `KISBroker.is_market_open()` + `state.last_executed_date`
패턴을 그대로 재사용한다. cron이 정확히 09:00에 한 번만 도는 게 아니라
09:00~09:55 사이 여러 번(예: 5분마다, cron_kr.txt 참고) 도는 좁은 폴링 윈도우로 설정해도,
이 패턴 덕분에 크론 시각이 실제 개장 처리 지연(휴장일 오판, 시스템 시계
오차 등)과 정확히 안 맞아도 안전하고, 중복 체결도 막아준다.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import get_data_root
from src.live.allocation import TargetOrder
from src.live.broker.base import BrokerAdapter, OrderResult
from src.live.broker.kis import KISBroker
from src.live.decide_kr import MARKET_ID_KR, decision_path
from src.live.notify import send_notification
from src.live.safety import LiveState, acquire_run_lock, enforce_order_caps, load_or_bootstrap_state, save_state


def _kst_today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()


def _load_kr_orders(path: Path) -> list[TargetOrder]:
    import json

    if not path.exists():
        raise FileNotFoundError(f"오늘자 결정 파일이 없음: {path} — decide_kr.py가 먼저 실행됐는지 확인할 것.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        TargetOrder(ticker=o["ticker"], side=o["side"], notional=o["notional"])
        for o in payload["orders"]
        if o["market_id"] == MARKET_ID_KR
    ]


def run_execute(
    mode: Literal["paper", "live"] = "paper",
    data_root: Path | None = None,
    target_date: pd.Timestamp | str | None = None,
    broker: BrokerAdapter | None = None,
) -> list[OrderResult]:
    """오늘자 결정 파일(KR분만 필터링)을 읽어 브로커에 제출한다. `mode`
    기본값은 반드시 `"paper"`다(CLAUDE.md Phase 4 핵심 결정 1번) — `"live"`를
    실제로 넘기는 건 이번 세션 범위 밖이라 여기서 명시적으로 막는다(코드
    경로 자체는 `KISBroker(mode="live")`로 존재하되, "안 부르기로 한
    약속"에만 의존하지 않기 위함).
    """
    if mode != "paper":
        raise ValueError(
            f"mode={mode!r}는 이번 세션 범위 밖 — live 자동집행은 별도 승인 후에만 "
            "활성화한다(CLAUDE.md Phase 4 핵심 결정 1번)."
        )

    data_root = data_root or get_data_root()
    target_date = pd.Timestamp(target_date) if target_date is not None else _kst_today()
    broker = broker or KISBroker(mode=mode)
    if broker.mode != "paper":
        # mode!="paper" 가드는 mode 문자열 인자만 본다 — broker=를 통해 실제로는
        # live 모드로 만들어진 브로커 인스턴스를 주입하면서 mode="paper"만 넘기면
        # 그 가드를 그대로 우회해 실거래 브로커가 호출될 수 있다 — execute_us.py
        # review-loop 2차 재검토로 발견된 것과 동일한 우회 경로라 처음부터 반영.
        raise ValueError(
            f"broker.mode={broker.mode!r}가 paper가 아님 — live 자동집행은 별도 승인 후에만 "
            "활성화한다(CLAUDE.md Phase 4 핵심 결정 1번)."
        )

    # 시장이 닫혀 있거나(좁은 폴링 윈도우 중 대부분) 오늘 이미 체결이 끝난
    # 경우는 정상 상태라 매번 알리면 스팸이 된다 — 그래서 executed=True는
    # 실제로 체결 사이클을 밟기 시작한 뒤에만 세팅하고, 아래 완료 알림은
    # executed일 때만 보낸다(예외는 어느 단계에서 나든 항상 알림 — except 참고).
    executed = False
    results: list[OrderResult] = []
    try:
        with acquire_run_lock(data_root, "execute_kr"):
            if not broker.is_market_open():
                return []  # 아직 장이 안 열렸거나 휴장일 — 조용히 종료

            state = load_or_bootstrap_state(data_root, "kr", broker.get_positions(), broker.get_cash(), broker.get_nav())
            if state.last_executed_date == str(target_date.date()):
                # 폴링 윈도우 안에서 이미 오늘(KST) 체결이 끝났다 — 그대로 다시
                # 제출하면 주문이 중복된다(모듈 docstring 참고). state.updated_at은
                # load_or_bootstrap_state()의 최초 부트스트랩에서도 "지금"으로 찍히므로
                # 이 판단에 쓰면 안 된다(safety.py의 LiveState.last_executed_date
                # docstring 참고).
                return []

            executed = True
            current_positions = broker.get_positions()
            current_cash = broker.get_cash()
            nav = broker.get_nav()
            from src.live.safety import check_reconciliation

            # decide_kr.py가 그날 아침 이미 한 번 확인했지만, 결정 시각과 체결
            # 시각 사이에 수동 개입 등으로 상태가 또 바뀌었을 수 있다 — 실제로
            # 돈이 움직이기 직전이므로 한 번 더 확인한다.
            check_reconciliation(current_positions, current_cash, state)

            orders = _load_kr_orders(decision_path(data_root, target_date))
            safe_orders = enforce_order_caps(orders, nav=nav)

            buy_failures: list[tuple[str, Exception]] = []
            for order in safe_orders:
                if order.side == "sell":
                    pos = current_positions.get(order.ticker)
                    if pos is None or pos.qty == 0:
                        continue  # 이미 청산됐거나 애초에 없음(재구성 통과했으니 정상 범위)
                    # allocation.py가 숏 포지션의 SELL은 이미 no-op으로 걸러내므로
                    # (경고만 남김) 여기 도달하는 건 전부 롱 청산이다 — qty>0.
                    result = broker.place_market_order(order.ticker, "sell", qty=pos.qty)
                    results.append(result)
                    continue

                # BUY: KIS는 notional 주문 API가 없다(broker/kis.py 모듈
                # docstring) — 체결 직전 실시간 현재가로 정수 수량을 계산한다.
                # get_current_price()는 execute_us.py에는 없던 종목별 추가
                # 네트워크 호출이다 — 여기서 예외를 그대로 흘려보내면 이 한
                # 종목의 현재가 조회 실패가 아직 처리 안 된 다른 매수는 물론,
                # 포지션을 줄이는 무관한 매도까지 통째로 막아버린다(SELL을
                # 항상 허용하는 킬스위치 철학과 반대 방향 — review-loop 1차
                # 재검토로 발견). 이 티커 하나로 실패 반경을 좁힌다.
                try:
                    price = broker.get_current_price(order.ticker)
                    qty = int(order.notional // price)
                    if qty <= 0:
                        # 이 티커에 배분된 notional이 1주 가격에도 못 미침(고가주 +
                        # 소액 배분이 겹치면 실제로 일어난다 — Alpaca는 소수점
                        # 매매가 되니 이 케이스 자체가 없었지만 KIS는 정수 수량만
                        # 되므로 새로 생기는 경로다). 실패가 아니라 정상적인 no-op.
                        print(
                            f"[execute_kr] {order.ticker} 매수 건너뜀 — "
                            f"notional {order.notional:,.0f}원으로 1주도 못 삼(현재가 {price:,.0f}원)"
                        )
                        continue
                    result = broker.place_market_order(order.ticker, "buy", qty=float(qty))
                except Exception as e:
                    buy_failures.append((order.ticker, e))
                    continue
                results.append(result)

            if buy_failures:
                # 종목마다 개별 알림을 보내면 스팸이 되므로(장애 시 여러 종목이
                # 한꺼번에 실패하기 쉬움) 하나로 모아 한 번만 알린다 — 이미
                # 체결된 다른 주문은 정상적으로 완료됐으므로 예외를 다시 던지지
                # 않는다(re-raise하면 여기까지 온 성공한 주문들의 state 저장이
                # 막혀버림).
                detail = "; ".join(f"{t}: {e!r}" for t, e in buy_failures)
                send_notification(f"[KR] 매수 {len(buy_failures)}건 처리 실패(현재가 조회/주문): {detail}", level="warning")

            final_positions = broker.get_positions()
            final_cash = broker.get_cash()
            final_nav = broker.get_nav()
            save_state(
                data_root,
                "kr",
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
        send_notification(f"[KR] execute_kr 실패({target_date.date()}): {e!r}", level="error")
        raise

    if executed:
        n_buy = sum(1 for r in results if r.side == "buy")
        n_sell = sum(1 for r in results if r.side == "sell")
        send_notification(f"[KR] execute 완료({target_date.date()}): 매수 {n_buy}건/매도 {n_sell}건 체결", level="info")

    return results


if __name__ == "__main__":
    run_execute()
