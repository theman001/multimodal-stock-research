"""Phase 4 US 트랙 — Alpaca Trading API v2 브로커 어댑터.

Alpaca REST API를 `requests`로 직접 호출한다(전용 SDK 미사용 — DART/EDGAR
모듈(`event_mapping.py` 등)이 이미 raw `requests` 호출로 통일돼 있는 이
저장소 관례를 그대로 따름, 신규 의존성 불필요). 소수점 매매(fractional
shares)를 지원하므로 `notional`(목표 금액) 기반 주문을 그대로 쓸 수 있어
목표비중→수량 변환이 필요 없다(plan/10 §B-4).

**mode="paper"(기본)/"live"** — 생성자 파라미터로 도메인/키 쌍만 분기한다.
paper/live는 Alpaca에서 서로 다른 계좌·키 쌍이므로 env var 이름 자체를
모드별로 분리했다(`ALPACA_PAPER_*`/`ALPACA_LIVE_*`) — paper 키를 live
엔드포인트에, 또는 그 반대로 잘못 쓰는 사고를 원천 차단하기 위함.
**live 경로는 코드로는 존재하되 이번 세션에서 실제로 호출하지 않는다**
(CLAUDE.md "Phase 4 구체 사양" 핵심 결정 1번).

Alpaca `/v2/positions`는 숏 포지션을 `side: "short"` 필드로 명시하지만
`qty` 문자열 자체의 부호는 API 버전에 따라 관례가 달라질 수 있다 — 그
불확실성에 기대지 않고 `side` 필드로 직접 부호를 강제해서
`HeldPosition.qty<0`(observation.py "롱 베이스, 숏은 전략적 허용" 결정)과
일관되게 맞춘다.

**심볼 표기 변환**: 이 저장소의 티커 유니버스(`rl_ticker_universe.json`)는
yfinance 관례를 따라 주식 클래스를 대시로 표기한다(예: "BRK-B" — 실제로
120종목 중 이 표기가 있는 게 확인됨). Alpaca API는 같은 종목을 점 표기로
쓴다("BRK.B") — 변환 없이 그대로 쓰면 `get_positions()`가 돌려주는 키가
`rl_ticker_universe.json`의 어떤 티커와도 안 맞아 observation.py의
`unknown_tickers` 검증에 걸리고(review-loop 5차 재검토로 발견), 반대
방향으로는 `place_market_order()`에 우리 표기를 그대로 넘기면 Alpaca가
그 심볼을 못 찾아 주문이 실패한다.
"""
from __future__ import annotations

import os
from typing import Literal

import numpy as np
import requests
from dotenv import load_dotenv

from src.live.broker.base import BrokerAdapter, OrderResult
from src.live.observation import HeldPosition

load_dotenv()

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"
API_VERSION = "v2"
DEFAULT_TIMEOUT_SECONDS = 30


def _to_alpaca_symbol(ticker: str) -> str:
    """우리 표준 티커(yfinance 관례, 예: "BRK-B")를 Alpaca 심볼 표기(예:
    "BRK.B")로 변환한다. 이 저장소 유니버스에서 대시는 오직 주식 클래스
    구분에만 쓰인다(다른 용도로 쓰인 사례 없음 확인) — 그 외 티커는
    변환해도 값이 그대로다."""
    return ticker.replace("-", ".")


def _from_alpaca_symbol(symbol: str) -> str:
    """`_to_alpaca_symbol()`의 역변환 — Alpaca 응답의 심볼을 우리 표준
    티커로 되돌린다."""
    return symbol.replace(".", "-")


def _raise_for_status_with_body(resp: requests.Response) -> None:
    """resp.raise_for_status()는 상태 코드만 보여주고 응답 본문(Alpaca가 실패
    사유를 담아 보내는 JSON message)은 버린다 — 실제 돈이 오가는 주문 실패의
    원인(잔고 부족, 잘못된 심볼, 장 마감 등)을 알 수 없게 되는 게 특히
    위험하다(review-loop 1차 재검토로 발견)."""
    if resp.status_code < 400:
        return
    try:
        detail = resp.json()
    except ValueError:
        detail = resp.text
    raise requests.HTTPError(f"{resp.status_code} Alpaca API 에러: {detail}", response=resp)


def _env_credentials(mode: Literal["paper", "live"]) -> tuple[str, str]:
    prefix = "ALPACA_PAPER" if mode == "paper" else "ALPACA_LIVE"
    key_id = os.environ.get(f"{prefix}_API_KEY_ID")
    secret_key = os.environ.get(f"{prefix}_SECRET_KEY")
    if not key_id or not secret_key:
        raise RuntimeError(
            f"{prefix}_API_KEY_ID/{prefix}_SECRET_KEY 환경변수가 설정되지 않음 (.env 확인) — "
            "paper/live는 Alpaca에서 서로 다른 키 쌍이라 모드별로 별도 env var를 쓴다."
        )
    return key_id, secret_key


class AlpacaBroker(BrokerAdapter):
    def __init__(self, mode: Literal["paper", "live"] = "paper", timeout: float = DEFAULT_TIMEOUT_SECONDS):
        super().__init__(mode=mode)
        self.base_url = PAPER_BASE_URL if mode == "paper" else LIVE_BASE_URL
        key_id, secret_key = _env_credentials(mode)
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key})

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{API_VERSION}{path}"

    def get_positions(self) -> dict[str, HeldPosition]:
        resp = self._session.get(self._url("/positions"), timeout=self.timeout)
        _raise_for_status_with_body(resp)
        positions: dict[str, HeldPosition] = {}
        for p in resp.json():
            side = p.get("side")
            if side not in ("long", "short"):
                # side가 없거나 "long"/"short" 외의 값이면 else 분기로 조용히
                # long 취급하는 게 가장 위험하다 — 진짜 숏 포지션을 롱으로
                # 잘못 분류하면 관측/배분 로직이 손익 방향을 통째로 반대로
                # 계산하게 된다. 모르는 값이면 조용히 넘어가지 않고 막는다
                # (review-loop 1차 재검토로 발견).
                raise ValueError(f"{p.get('symbol')}의 side 값이 예상 밖: {side!r} (long/short만 허용)")
            qty = abs(float(p["qty"]))
            signed_qty = -qty if side == "short" else qty
            ticker = _from_alpaca_symbol(p["symbol"])
            positions[ticker] = HeldPosition(qty=signed_qty, avg_entry_price=float(p["avg_entry_price"]))
        return positions

    def get_cash(self) -> float:
        resp = self._session.get(self._url("/account"), timeout=self.timeout)
        _raise_for_status_with_body(resp)
        return float(resp.json()["cash"])

    def get_nav(self) -> float:
        resp = self._session.get(self._url("/account"), timeout=self.timeout)
        _raise_for_status_with_body(resp)
        return float(resp.json()["portfolio_value"])

    def place_market_order(
        self,
        ticker: str,
        side: Literal["buy", "sell"],
        qty: float | None = None,
        notional: float | None = None,
        client_order_id: str | None = None,
    ) -> OrderResult:
        if (qty is None) == (notional is None):
            raise ValueError(f"qty/notional 중 정확히 하나만 지정해야 함: qty={qty!r}, notional={notional!r}")

        if qty is not None:
            submitted_qty, submitted_notional = qty, None
        else:
            # allocation.py::compute_target_allocations()의 equal_share는 그냥
            # 나눗셈(cash/len(buy_idx))이라 333.33333333333337처럼 부동소수점
            # 정밀도가 그대로 남는 값이 나오기 쉽다 — 달러 금액을 센트 단위
            # 이상 정밀도로 보내는 건 의미가 없고, 브로커가 거부할 수도 있다
            # (review-loop 2차 재검토로 발견). 반올림한 값을 실제로 전송하고
            # OrderResult.submitted_notional에도 그 값을 남긴다 — 원본값을
            # 남기면 "기록상 신청 금액"과 "브로커에 실제로 보낸 금액"이
            # 달라져 나중에 리포트를 볼 때 혼란스럽다.
            submitted_qty, submitted_notional = None, round(notional, 2)

        value = submitted_qty if submitted_qty is not None else submitted_notional
        if not (np.isfinite(value) and value > 0):
            # 검증은 반드시 반올림 "이후" 값으로 해야 한다 — notional=0.004는
            # 반올림 전엔 양수라 2차 재검토에서 원본값 기준으로 막았을 땐
            # 통과했지만, 반올림 후 "0.00"이 돼 사실상 $0 주문이 나가는
            # 구멍이 그대로 남아 있었다(3차 재검토로 발견, 직전 라운드
            # 자신의 수정에 남은 문제).
            raise ValueError(f"qty/notional은 반올림 후에도 유한한 양수여야 함: {value!r}")

        body: dict = {
            "symbol": _to_alpaca_symbol(ticker),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        if submitted_qty is not None:
            body["qty"] = str(submitted_qty)
        else:
            body["notional"] = str(submitted_notional)
        if client_order_id is not None:
            # 이 계층 자체는 재시도하지 않지만(base.py 문서 참고), 호출자가
            # 나중에 안전하게 재시도하려면 Alpaca가 지원하는 client_order_id
            # 중복제출 방지를 쓸 수 있어야 한다 — 지금 매개변수를 안 열어두면
            # execute_*.py(4단계)에서 재시도 정책을 설계할 때 이 어댑터부터
            # 다시 고쳐야 한다(review-loop 1차 재검토로 발견).
            body["client_order_id"] = client_order_id

        resp = self._session.post(self._url("/orders"), json=body, timeout=self.timeout)
        _raise_for_status_with_body(resp)
        data = resp.json()

        return OrderResult(
            ticker=ticker,
            side=side,
            status=data.get("status", "unknown"),
            order_id=data.get("id"),
            submitted_qty=submitted_qty,
            submitted_notional=submitted_notional,
            filled_qty=float(data["filled_qty"]) if data.get("filled_qty") else None,
            filled_avg_price=float(data["filled_avg_price"]) if data.get("filled_avg_price") else None,
            raw=data,
        )

    def is_market_open(self) -> bool:
        resp = self._session.get(self._url("/clock"), timeout=self.timeout)
        _raise_for_status_with_body(resp)
        return bool(resp.json()["is_open"])
