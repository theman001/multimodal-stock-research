"""Phase 4 공유 — 브로커 실행 어댑터 공통 인터페이스.

plan/10 §B-4/§C-2. "모드"는 클래스 분기가 아니라 생성자 파라미터
(`mode: Literal["paper","live"]`)로 base URL/키 쌍만 바뀌는 구조다 —
의사결정 로직(observation → infer → allocation)은 모의/실전 공용이고,
이 어댑터가 실제로 "브로커에 주문을 보낸다" 단계만 모드로 분기한다.

`get_positions()`는 `src/live/observation.py::HeldPosition`을 반환한다.
plan/10 §B-4의 원래 pseudocode는 `dict[str, float]`(수량만)이었지만,
observation.py 구현 중(Phase 4 2단계) `unrealized_return` 계산에 평단가
(`avg_entry_price`)가 반드시 필요함을 발견해 확장했다 — 수량만으로는
"보유 중"인지는 알아도 "얼마에 샀는지"를 몰라 평가손익을 계산할 방법이
없었다. 새 브로커 어댑터를 추가할 때도 이 확장된 형태를 그대로 따를 것.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from src.live.observation import HeldPosition


@dataclass
class OrderResult:
    """`place_market_order()`의 반환값. `raw`에 브로커 원본 응답을 그대로
    보존해 감사/디버깅에 쓴다(일일 리포트 — plan/10 §B-7)."""

    ticker: str
    side: Literal["buy", "sell"]
    status: str  # 브로커가 보고한 주문 상태 그대로(예: "accepted", "filled", "rejected")
    order_id: str | None = None
    submitted_qty: float | None = None
    submitted_notional: float | None = None
    filled_qty: float | None = None
    filled_avg_price: float | None = None
    raw: dict | None = None


class BrokerAdapter(ABC):
    """`mode="paper"`(기본)/`"live"` — 생성자에서 도메인/키만 분기한다(§B-4).

    **기본값은 절대 `live`가 아니어야 한다**(Phase 4 핵심 결정 1번,
    CLAUDE.md "Phase 4 구체 사양"). `live` 경로를 실제로 호출하는 코드는
    존재해야 하지만, 이번 구현 범위에서 실제로 실행하지는 않는다(별도
    승인 후).
    """

    def __init__(self, mode: Literal["paper", "live"] = "paper"):
        self.mode = mode

    @abstractmethod
    def get_positions(self) -> dict[str, HeldPosition]:
        """ticker -> HeldPosition(qty, avg_entry_price). `qty<0`은 숏
        포지션이다("롱 베이스, 숏은 전략적 허용" 결정, observation.py 모듈
        docstring 참고)."""

    @abstractmethod
    def get_cash(self) -> float: ...

    @abstractmethod
    def get_nav(self) -> float:
        """브로커가 직접 보고하는 계좌 순자산. 용도가 세 가지다.
        (1) safety.py의 재구성(reconciliation) 체크에서 "우리가 재계산한
        NAV vs 브로커가 실제로 보고하는 NAV" 괴리를 감지하는 용도 —
        observation.py는 학습 시점과 동일한 가격 소스(우리 자체 OHLCV
        파이프라인의 종가)로 NAV를 스스로 재계산해서 쓰고 이 값을 그대로
        쓰지 않는다. (2) execute_us.py가 주문 캡(`enforce_order_caps`)을
        계산할 때의 NAV로도 이 값을 쓴다 — 그 시점엔 observation.py의
        재계산 파이프라인이 이미 다 끝난 뒤(아침 decide 시점)라 새로 돌릴
        게 아니라면 손쉽게 구할 수 있는 유일한 NAV이고, 체결 직전 브로커의
        실시간 값을 쓰는 게 아침에 계산해둔 스냅샷보다 오히려 더 정확하다.
        (3) `safety.py::load_or_bootstrap_state()`가 최초 배포 시점 `nav_anchor`를
        seed할 때도 이 값을 쓴다 — 그 시점엔 observation.py가 아직 한 번도
        돈 적이 없어 재계산할 방법 자체가 없다(정상적인 최초 배포는 보유
        포지션 없이 현금뿐이라 이 값과 재계산값이 사실상 같지만, state.json이
        중간에 유실돼 기존 포지션이 있는 채로 재부트스트랩되면 두 값이
        약간 어긋날 수 있다는 점은 알려진 한계로 남겨둔다)."""

    @abstractmethod
    def place_market_order(
        self,
        ticker: str,
        side: Literal["buy", "sell"],
        qty: float | None = None,
        notional: float | None = None,
        client_order_id: str | None = None,
    ) -> OrderResult:
        """시장가 주문을 제출한다. `qty`/`notional` 중 정확히 하나만 지정한다
        (대부분의 현대 브로커 API가 이 방식을 지원 — `notional`은 목표
        금액, `qty`는 목표 수량). 재시도는 이 계층의 책임이 아니다 — 주문
        제출을 맹목적으로 재시도하면 응답 유실 시 중복 주문이 나갈 수 있어,
        재시도/멱등성 판단은 호출자(execute_*.py, safety.py)의 몫으로
        남겨둔다. `client_order_id`는 그 재시도를 안전하게 만들기 위한
        고리다 — 호출자가 매 주문에 고유값을 지정해두면, 브로커가 지원하는
        경우 같은 값으로 재제출 시 중복 주문 대신 기존 주문으로 처리된다
        (구현체가 지원하지 않으면 무시해도 됨)."""

    @abstractmethod
    def is_market_open(self) -> bool: ...

    @abstractmethod
    def count_open_orders(self) -> int:
        """아직 체결/취소/거부되지 않고 열려 있는 주문 수. `place_market_order()`는
        시장가라도 "접수"만 확인하고 체결을 보장하지 않는다 — 특히 개장 정각의
        opening cross 중에는 즉시 체결되지 않는다. execute_*.py 가 주문 제출
        직후 `get_positions()`/`get_cash()` 를 찍어 state 로 저장하면 체결 전
        스냅샷이 남고, 다음날 아침 재구성 체크가 "기록 vs 현실" 100% 불일치로
        오판한다(2026-08-31 실제 발생 — decide_us 가 이 이유로 멈춤)."""

    def wait_until_orders_settle(self, timeout_s: float = 180.0, poll_s: float = 5.0) -> int:
        """열린 주문이 모두 정리(체결/취소/거부)될 때까지 폴링한다. 남은
        미체결 주문 수를 반환 — 0이 아니면 timeout_s 안에 다 안 끝난 것.
        execute_*.py 가 최종 state 스냅샷 직전에 호출한다(모듈 docstring 참고).
        `count_open_orders()` 자체가 실패하면(브로커 조회 오류) 예외를 그대로
        올린다 — 이 시점의 조회 실패는 state 정확성에 직결되므로 삼키지 않는다."""
        deadline = time.monotonic() + timeout_s
        while True:
            n = self.count_open_orders()
            if n == 0 or time.monotonic() >= deadline:
                return n
            time.sleep(poll_s)
