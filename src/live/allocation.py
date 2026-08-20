"""Phase 4 공유 — `TradingEnv.step()`의 BUY/SELL 배분 로직을 라이브용 순수 함수로 재구현.

`TradingEnv`(src/rl/trading_env.py)는 백테스트 시뮬레이션 장부를 직접
mutate하며 거래비용(`round_trip_cost_for_market()`)도 함께 시뮬레이션한다.
라이브 실행은 실제 브로커가 실제 체결가/수수료를 부과하므로(plan/10 §B-4:
"round_trip_cost_for_market()는 브로커가 실제로 부과하므로 라이브 코드에서
재차감 불필요"), 이 함수는 "몇 종목을 사고/팔지, 얼마씩 배분할지"라는 배분
산술(equal_share/cap/no-op 규칙, trading_env.py 189-223행)만 수치까지
동일하게 재현하고, 비용 차감은 하지 않는다 — 반환하는 BUY notional은
"비용 차감 전" 목표 배분 금액이다.

`TradingEnv.step()`은 같은 스텝 안에서 SELL을 먼저 정산해 그 현금을 BUY
배분에 쓴다(trading_env.py 189-223행: SELL 처리가 BUY 처리보다 먼저 옴,
"같은 스텝에 판 현금을 그 스텝의 BUY 배분에 쓸 수 있게" 명시) — 이 함수도
`position_values`(SELL 대상 티커의 현재 시가평가액)로 동일한
`cash_after_sells`를 계산해 같은 순서를 재현한다.

`position_values`에 숏 포지션(observation.py의 `HeldPosition.qty<0`, "롱
베이스, 숏은 전략적 허용" 결정 참고)의 음수 시가평가액이 섞여 있을 수 있다.
이 함수의 SELL 판단은 여전히 "SELL=롱 청산"이라는 전제로 짜여 있어(정책의
3지 선택 자체는 안 바꾼다는 스코프 결정), 숏에 그대로 적용하면 실제 필요한
커버링(매수)과 반대 방향 주문이 나간다 — 그래서 SELL 대상 중 시가평가액이
음수인 티커는 주문을 만들지 않고 건너뛴다(no-op, 경고만 남김). 건너뛴
티커는 `cash_after_sells` 계산에서도 제외된다(그 포지션을 오늘 정산하지
않으므로).

**명시적으로 미구현인 부분**: `TradingEnv.step()`은 SELL 처리보다도 먼저
`masked_streak > MAX_MASKED_STREAK_DAYS`(연속 마스킹 10일 초과) 포지션을
강제 청산한다(trading_env.py 185-187행). 이건 "며칠 연속 그 티커 가격이
없었는가"라는 상태를 날짜에 걸쳐 추적해야 하는데, 이 함수는 순수 함수라
그 상태를 갖고 있지 않다 — 호출자가 masked_streak를 별도로 추적해 넘기지
않는 한 이 강제청산 규칙은 라이브 경로에 아직 없다. 상시 데이터 수집이
정상이면 거의 발동 안 할 안전장치이긴 하나, 상태 추적을 요구하는
`safety.py`(다음 단계) 쪽에 자연스럽게 속하는 문제라 여기서는 구현하지
않고 이 gap을 명시적으로 남겨둔다.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.rl.trading_env import ACTION_BUY, ACTION_SELL, MAX_POSITION_WEIGHT


@dataclass
class TargetOrder:
    ticker: str
    side: Literal["buy", "sell"]
    # BUY: 목표 배분 금액(비용 차감 전, 브로커가 실제 체결 시 자체 비용을 부과함).
    # SELL: None — 전량 청산이므로 실제 수량은 실행 단계에서 브로커 계좌 조회로 결정한다.
    notional: float | None = None


def compute_target_allocations(
    action: np.ndarray,
    tickers: list[str],
    current_holdings: set[str],
    position_values: dict[str, float],
    cash: float,
    nav: float,
    valid_mask: np.ndarray,
    has_price: np.ndarray,
    max_position_weight: float = MAX_POSITION_WEIGHT,
) -> list["TargetOrder"]:
    """TradingEnv.step()과 동일한 규칙으로 오늘의 목표 주문 목록을 계산한다.

    - SELL: 보유 중 + 오늘 가격 있음 + action==SELL 인 티커 전량 청산.
    - BUY: 미보유 + valid_mask=1 + 오늘 가격 있음 + action==BUY 인 티커에
      한해, `equal_share = cash_after_sells / len(buy_tickers)`를
      `max_position_weight * nav`로 캡. 캡으로 남는 현금은 재분배하지
      않는다(trading_env.py와 동일, 보수적).
    - 보유 중 BUY, 미보유 SELL/HOLD, 보유 중 HOLD는 전부 no-op(주문 없음).
    """
    action = np.asarray(action)
    n = len(tickers)
    if not (len(action) == n and len(valid_mask) == n and len(has_price) == n):
        # 길이가 다르면 아래 numpy 불리언 연산(&)이 broadcast 에러를 내긴 하지만,
        # 어느 파라미터가 문제인지 이름을 안 알려준다 — tickers/action/valid_mask/
        # has_price는 전부 같은 순서의 티커별 배열이어야 하므로(암묵적 계약),
        # 여기서 먼저 이름 붙은 에러로 막는다(재검토로 발견, infer.py의 shape
        # 체크와 동일한 성격).
        raise ValueError(
            f"tickers({n})/action({len(action)})/valid_mask({len(valid_mask)})/"
            f"has_price({len(has_price)}) 길이가 서로 다름 — 같은 순서의 티커별 "
            "배열이어야 한다."
        )
    holding_arr = np.array([t in current_holdings for t in tickers])
    valid = np.asarray(valid_mask).astype(bool)
    priced = np.asarray(has_price).astype(bool)

    sell_mask = holding_arr & priced & (action == ACTION_SELL)
    raw_sell_idx = np.where(sell_mask)[0]

    # "청산 = SELL 주문"은 롱 전제다 — 숏(시가평가액 음수)을 청산하려면 실제로는
    # 매수(커버)해야 하는데, 그대로 side="sell"을 내면 실제 필요와 반대 방향
    # 주문이 나가 숏을 더 늘리게 된다("롱 베이스, 숏은 전략적 허용" 결정 이후
    # observation.py가 숏의 시가평가액을 음수로 정확히 반영하기 시작하면서 생긴
    # 통합 지점 위험 — 구현 중 리뷰로 발견). 처음엔 예외를 던져 막았지만(1차
    # 수정), 그러면 정책이 그 숏 티커에 SELL을 낼 때마다 나머지 119종목의
    # 결정까지 매일 통째로 막혀 "숏은 전략적으로 유지"라는 의도와 정면으로
    # 부딪힌다(재검토로 발견) — 대신 그 티커만 no-op(주문 미생성)으로
    # 건너뛰고, 경고만 남겨(숏 존재 자체의 최종 감지·알림은 safety.py의
    # 재구성 체크 몫으로 남김) 나머지는 정상 진행한다.
    short_sell_idx = {i for i in raw_sell_idx if position_values.get(tickers[i], 0.0) < 0}
    if short_sell_idx:
        warnings.warn(
            f"숏 포지션에 SELL 결정이 나와 주문을 만들지 않고 건너뜀(no-op): "
            f"{sorted(tickers[i] for i in short_sell_idx)} — 이 함수는 SELL을 "
            "'롱 청산'으로만 처리하므로 그대로 내면 실제 필요한 커버링(매수)과 "
            "반대 방향 주문이 된다.",
            stacklevel=2,
        )
    sell_idx = np.array([i for i in raw_sell_idx if i not in short_sell_idx], dtype=int)

    cash_after_sells = cash + sum(position_values.get(tickers[i], 0.0) for i in sell_idx)

    buy_mask = (~holding_arr) & valid & priced & (action == ACTION_BUY)
    buy_idx = np.where(buy_mask)[0]

    orders: list[TargetOrder] = [TargetOrder(ticker=tickers[i], side="sell") for i in sell_idx]

    if len(buy_idx) > 0:
        equal_share = cash_after_sells / len(buy_idx)
        cap = max_position_weight * nav
        # cash/nav가 정상 범위를 벗어나면(재구성 불일치 등 safety.py가 원래 걸러야
        # 할 상황) equal_share/cap이 음수가 될 수 있는데, min()이 그 음수를 그대로
        # 통과시키면 "매수 금액이 음수인 매수 주문"이 만들어진다 — 0 이하로는
        # 절대 BUY 주문을 내지 않는다(구현 중 리뷰로 발견).
        alloc = max(min(equal_share, cap), 0.0)
        if alloc > 0:
            orders.extend(TargetOrder(ticker=tickers[i], side="buy", notional=alloc) for i in buy_idx)

    return orders
