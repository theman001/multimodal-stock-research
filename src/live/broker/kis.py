"""Phase 4 KR 트랙 — 한국투자증권(KIS) Open API 브로커 어댑터.

**중요 — TR_ID/응답 필드명은 재확인이 필요하다.** 이 파일은 이번 세션에서
실제 KIS API에 네트워크 호출을 한 번도 하지 않고(모의투자 앱키가 아직
`.env`에 없음) 작성됐다 — 공개된 KIS Open API 문서/커뮤니티 레퍼런스 기준
best-effort 구현이다. `_TR_ID_*` 상수, `inquire-balance`/`order-cash`/
`chk-holiday` 응답의 정확한 필드명(`dnca_tot_amt`/`tot_evlu_amt`/`opnd_yn`
등)은 실제 모의투자 앱키를 발급받아 첫 스모크 테스트를 돌릴 때 KIS
Developer 포털 문서로 반드시 재확인할 것 — 이 프로젝트가 pykrx/KRX Open
API에 대해 이미 취해온 방침("정확한 함수 시그니처는 버전에 따라 다를 수
있으니 구현 시 재확인할 것", CLAUDE.md)과 동일하게 취급한다. 다행히 실패
모드가 안전한 방향이다 — TR_ID가 틀리면 KIS가 명확한 에러 응답(`rt_cd`!=
"0")으로 주문을 거부할 뿐, 조용히 엉뚱한 주문이 나가지는 않는다.

Alpaca와의 구조적 차이 (plan/10 §B-4):
- **모의투자(`openapivts.koreainvestment.com:29443`)/실전
  (`openapi.koreainvestment.com:9443`)이 서로 다른 도메인**이자 서로 다른
  앱키/계좌쌍이다 — `KIS_PAPER_*`/`KIS_LIVE_*`로 env var를 분리해 paper
  키로 live 도메인을 부르는 사고를 원천 차단한다(Alpaca 어댑터와 동일 관례).
- **OAuth2 액세스 토큰 발급이 필요하다**(`/oauth2/tokenP`) — Alpaca는
  API 키만으로 매 요청을 인증하지만 KIS는 먼저 Bearer 토큰을 발급받아야
  하고, 발급은 유효기간이 길고(수 시간~24시간) 빈번한 재발급이 권장되지
  않으므로 인스턴스 안에서 캐싱해 재사용한다.
- **주문 요청에는 `hashkey`가 추가로 필요하다**(`/uapi/hashkey`) — 주문
  본문(body)을 먼저 이 엔드포인트에 보내 해시값을 받고, 실제 주문 요청
  헤더에 그 해시를 실어 보내야 한다(요청 위변조 방지 목적의 KIS 특유
  절차).
- **정수 주식 수량만 허용, notional(금액) 기반 주문 API가 없다** —
  CLAUDE.md "Phase 4 구체 사양" 3번, plan/10 §B-4. `allocation.py`(공유
  모듈)가 만드는 `TargetOrder`는 Alpaca(소수점 매매 지원)를 기준으로
  notional을 담는데, 이 어댑터는 notional을 받으면 조용히 변환하지 않고
  즉시 `NotImplementedError`를 던진다. notional -> 정수 수량 변환은
  `get_current_price()`(2026-08-20 사용자 결정 — `execute_kr.py`가 체결
  직전 KIS 현재가를 실시간 조회해서 씀, `observation.py`의 관측 시점 종가를
  재사용하지 않음)로 이 어댑터가 기준가 조회 자체는 제공하되, `notional //
  price` 나눗셈과 그 결과를 실제로 주문에 쓰는 결정은 여전히
  `execute_kr.py`(Phase 4 KR 4단계)의 몫이다 — `place_market_order()`는
  이미 계산된 정수 `qty`만 받는다. `get_current_price()`는 Alpaca 쪽엔
  대응 개념이 아예 없어서 공유 `BrokerAdapter` ABC가 아니라 `KISBroker`
  전용 메서드로 뒀다.
- **KIS는 HTTP 200이어도 비즈니스 로직 실패를 응답 본문의 `rt_cd`로
  알린다**(Alpaca는 HTTP 상태코드로 실패를 표현) — HTTP 레벨 에러 체크만
  하면 이 실패를 놓친다.
- KR 티커는 이미 순수 6자리 숫자 문자열이라(`rl_ticker_universe.json`
  확인) Alpaca처럼 별도 심볼 표기 변환이 필요 없다.
- 이 프로젝트의 모의투자 계좌는 일반 위탁계좌(현금 매매)를 전제한다 —
  신용/대주 같은 숏 메커니즘을 다루지 않으므로 `get_positions()`가 돌려주는
  수량은 항상 0 이상이다(Alpaca의 `side` 필드 기반 부호 처리 같은 로직이
  필요 없음).
"""
from __future__ import annotations

import os
import time
from typing import Literal

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.live.broker.base import BrokerAdapter, OrderResult
from src.live.observation import HeldPosition

load_dotenv()

PAPER_BASE_URL = "https://openapivts.koreainvestment.com:29443"
LIVE_BASE_URL = "https://openapi.koreainvestment.com:9443"
DEFAULT_TIMEOUT_SECONDS = 30
_TOKEN_EXPIRY_BUFFER_SECONDS = 60  # 만료 시각 직전 재사용해 요청 중 만료되는 경합을 피함

_BALANCE_PATH = "/uapi/domestic-stock/v1/trading/inquire-balance"
_ORDER_PATH = "/uapi/domestic-stock/v1/trading/order-cash"
_OPEN_ORDERS_PATH = "/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl"
_HASHKEY_PATH = "/uapi/hashkey"
_PRICE_PATH = "/uapi/domestic-stock/v1/quotations/inquire-price"
_TOKEN_PATH = "/oauth2/tokenP"

# TR_ID는 모의/실전이 접두사(V)로만 구분되는 게 KIS의 일반 관례다 — 재확인 필요(모듈 docstring 참고).
_TR_ID_BALANCE = {"paper": "VTTC8434R", "live": "TTTC8434R"}
_TR_ID_BUY = {"paper": "VTTC0802U", "live": "TTTC0802U"}
_TR_ID_SELL = {"paper": "VTTC0801U", "live": "TTTC0801U"}
_TR_ID_OPEN_ORDERS = {"paper": "VTTC8036R", "live": "TTTC8036R"}  # 정정취소가능주문조회 — 재확인 필요
_TR_ID_CURRENT_PRICE = "FHKST01010100"  # 조회성 TR이라 모의/실전 공통으로 추정 — 재확인 필요

_ORDER_DVSN_MARKET = "01"  # 시장가

_KST_MARKET_OPEN = (9, 0)
_KST_MARKET_CLOSE = (15, 30)


def _is_within_market_hours(now_kst: pd.Timestamp) -> bool:
    """09:00~15:30 KST(양끝 포함) 여부만 판단하는 순수 함수 — `is_market_open()`에서
    분리해 실제 현재 시각을 몰래 사용하지 않고도(pd.Timestamp.now()를 mock하지
    않고도) 경계값을 직접 테스트할 수 있게 한다."""
    open_h, open_m = _KST_MARKET_OPEN
    close_h, close_m = _KST_MARKET_CLOSE
    market_open = now_kst.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    market_close = now_kst.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    return market_open <= now_kst <= market_close


def _env_credentials(mode: Literal["paper", "live"]) -> tuple[str, str, str, str]:
    prefix = "KIS_PAPER" if mode == "paper" else "KIS_LIVE"
    app_key = os.environ.get(f"{prefix}_APP_KEY")
    app_secret = os.environ.get(f"{prefix}_APP_SECRET")
    cano = os.environ.get(f"{prefix}_CANO")
    acnt_prdt_cd = os.environ.get(f"{prefix}_ACNT_PRDT_CD")
    if not app_key or not app_secret or not cano or not acnt_prdt_cd:
        raise RuntimeError(
            f"{prefix}_APP_KEY/{prefix}_APP_SECRET/{prefix}_CANO/{prefix}_ACNT_PRDT_CD 환경변수가 "
            "설정되지 않음 (.env 확인) — paper/live는 KIS에서 서로 다른 앱키·계좌이므로 모드별로 "
            "별도 env var를 쓴다(Alpaca 어댑터와 동일 관례)."
        )
    return app_key, app_secret, cano, acnt_prdt_cd


def _raise_for_status_with_body(resp: requests.Response) -> None:
    """resp.raise_for_status()는 상태 코드만 보여주고 응답 본문(KIS가 실패
    사유를 담아 보내는 JSON message)은 버린다 — 실제 돈이 오가는 주문 실패의
    원인을 알 수 없게 되는 게 특히 위험하다(Alpaca 어댑터 review-loop 1차
    재검토에서 발견된 것과 동일한 교훈을 처음부터 반영)."""
    if resp.status_code < 400:
        return
    try:
        detail = resp.json()
    except ValueError:
        detail = resp.text
    raise requests.HTTPError(f"{resp.status_code} KIS API 에러: {detail}", response=resp)


def _check_rt_cd(data: dict) -> None:
    """KIS는 HTTP 200이어도 rt_cd!='0'이면 비즈니스 로직 실패(잔고부족/휴장일/
    잘못된 파라미터 등)다 — HTTP 상태코드만 보면 이런 실패를 놓친다."""
    rt_cd = data.get("rt_cd")
    if rt_cd != "0":
        raise RuntimeError(f"KIS API 오류 (rt_cd={rt_cd}): {data.get('msg1')}")


class KISBroker(BrokerAdapter):
    def __init__(self, mode: Literal["paper", "live"] = "paper", timeout: float = DEFAULT_TIMEOUT_SECONDS):
        super().__init__(mode=mode)
        self.base_url = PAPER_BASE_URL if mode == "paper" else LIVE_BASE_URL
        self.app_key, self.app_secret, self.cano, self.acnt_prdt_cd = _env_credentials(mode)
        self.timeout = timeout
        self._session = requests.Session()
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _ensure_access_token(self) -> str:
        now = time.time()
        if self._access_token is not None and now < self._token_expires_at:
            return self._access_token

        resp = self._session.post(
            self._url(_TOKEN_PATH),
            json={"grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret},
            timeout=self.timeout,
        )
        _raise_for_status_with_body(resp)
        data = resp.json()
        self._access_token = data["access_token"]
        expires_in = float(data.get("expires_in", 86400))
        self._token_expires_at = now + expires_in - _TOKEN_EXPIRY_BUFFER_SECONDS
        return self._access_token

    def _auth_headers(self, tr_id: str) -> dict:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._ensure_access_token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",  # 개인(retail) 계좌 전제 — 이 프로젝트 스코프
        }

    def _hashkey(self, body: dict) -> str:
        resp = self._session.post(
            self._url(_HASHKEY_PATH),
            json=body,
            headers={
                "content-type": "application/json; charset=utf-8",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
            },
            timeout=self.timeout,
        )
        _raise_for_status_with_body(resp)
        return resp.json()["HASH"]

    def _inquire_balance(self) -> dict:
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        resp = self._session.get(
            self._url(_BALANCE_PATH),
            params=params,
            headers=self._auth_headers(_TR_ID_BALANCE[self.mode]),
            timeout=self.timeout,
        )
        _raise_for_status_with_body(resp)
        data = resp.json()
        _check_rt_cd(data)
        return data

    def get_current_price(self, ticker: str) -> float:
        """`BrokerAdapter` ABC에는 없는 KIS 전용 메서드다 — Alpaca는 notional
        (금액) 기반 주문을 직접 지원해 "몇 주를 살지" 계산이 브로커 내부에서
        끝나지만, KIS는 정수 수량만 받으므로(모듈 docstring "정수 주식 수량만
        허용" 참고) `execute_kr.py`가 `allocation.py`의 notional을 정수 수량으로
        바꿀 기준가가 필요하다. 이 변환은 공유 `BrokerAdapter` 인터페이스의
        책임이 아니라(Alpaca 쪽엔 아예 무의미한 개념) `execute_kr.py`가
        `KISBroker`인 걸 알고 직접 호출하는 KR 트랙 전용 경로다(2026-08-20
        사용자 결정 — 실시간 조회, 관측 시점의 종가 재사용 아님). 체결 직전
        시세이므로 결정(07:00 KST) 시점과 실제 체결(09:00 KST) 시점 사이 갭을
        학습 시뮬레이션보다 더 최신 가격으로 좁혀준다(plan/10 §D "오버나이트
        갭" 항목과 관련)."""
        resp = self._session.get(
            self._url(_PRICE_PATH),
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker},
            headers=self._auth_headers(_TR_ID_CURRENT_PRICE),
            timeout=self.timeout,
        )
        _raise_for_status_with_body(resp)
        data = resp.json()
        _check_rt_cd(data)
        # data["output"]으로 직접 인덱싱하지 않는다 — 이 파일의 다른 메서드
        # (place_market_order/is_market_open)와 동일하게 .get()으로 방어한다.
        # 거래정지/상장폐지 종목처럼 rt_cd="0"이면서도 output이 비거나 필드가
        # 빠진 응답이 올 가능성을 배제할 수 없다(재확인 필요) — 그런 경우
        # 원인 불명의 KeyError 대신 명확한 도메인 에러로 실패해야 한다.
        stck_prpr = data.get("output", {}).get("stck_prpr")
        if stck_prpr is None:
            raise RuntimeError(f"{ticker}의 현재가 조회 응답에 stck_prpr이 없음(거래정지/상장폐지 의심): {data!r}")
        price = float(stck_prpr)
        if not (np.isfinite(price) and price > 0):
            # notional // price로 바로 나눗셈에 쓰일 값이다 — 0/NaN/Inf가
            # 여기서 안 걸리면 ZeroDivisionError나 터무니없는 qty로 이어진다.
            raise RuntimeError(f"{ticker}의 현재가 조회 결과가 비정상적임(유한한 양수 아님): {price!r}")
        return price

    def get_positions(self) -> dict[str, HeldPosition]:
        data = self._inquire_balance()
        positions: dict[str, HeldPosition] = {}
        for row in data.get("output1", []):
            qty = float(row["hldg_qty"])
            if qty == 0:
                # 전량 청산된 과거 보유 종목이 0수량 행으로 남아있는 경우를
                # 방어적으로 걸러낸다 — 실제로 이런 행이 오는지는 재확인 필요.
                continue
            positions[row["pdno"]] = HeldPosition(qty=qty, avg_entry_price=float(row["pchs_avg_pric"]))
        return positions

    def get_cash(self) -> float:
        data = self._inquire_balance()
        return float(data["output2"][0]["dnca_tot_amt"])

    def get_nav(self) -> float:
        # get_nav()는 safety.py의 재구성 체크/일일 손실 킬스위치/주문 캡 계산에
        # 직접 쓰이는 값이다(broker/base.py의 get_nav() 계약 참고) — 여기서
        # 잘못 짚으면 그 안전장치들이 잘못된 기준으로 동작한다. KIS
        # inquire-balance output2에는 이름이 비슷한 여러 필드가 있어(예:
        # `scts_evlu_amt`=유가증권평가금액만, `nass_amt`=순자산금액) `tot_evlu_amt`가
        # "현금+보유주식 평가액 합계"(원하는 진짜 NAV)인지 "보유주식 평가액만"
        # (현금 미포함)인지가 모듈 docstring의 일반적 재확인 경고보다 구체적으로
        # 검증돼야 한다 — 첫 스모크 테스트에서 get_cash()+포지션 평가액 합과
        # 대조해 반드시 확인할 것.
        data = self._inquire_balance()
        return float(data["output2"][0]["tot_evlu_amt"])

    def place_market_order(
        self,
        ticker: str,
        side: Literal["buy", "sell"],
        qty: float | None = None,
        notional: float | None = None,
        client_order_id: str | None = None,
    ) -> OrderResult:
        if notional is not None:
            raise NotImplementedError(
                "KIS는 notional(금액) 기반 주문 API가 없다 — 정수 주식 수량(qty)만 지원한다. "
                "notional -> qty 변환(체결가 조회 포함)은 이 어댑터가 암묵적으로 떠맡지 않는다 — "
                "execute_kr.py가 명시적으로 처리해야 한다."
            )
        if qty is None:
            raise ValueError("KIS는 qty가 필수다(notional 미지원)")
        if not (np.isfinite(qty) and qty > 0):
            raise ValueError(f"qty는 유한한 양수여야 함: {qty!r}")
        if qty != int(qty):
            raise ValueError(f"KIS는 정수 주식 수량만 허용한다(소수점 매매 불가): qty={qty!r}")
        qty_int = int(qty)

        body = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": ticker,
            "ORD_DVSN": _ORDER_DVSN_MARKET,
            "ORD_QTY": str(qty_int),
            "ORD_UNPR": "0",
        }
        # client_order_id: KIS order-cash에는 클라이언트 지정 멱등성 키를 실어 보낼
        # 필드가 없다(재확인 필요) — base.py 계약대로 받되 무시한다.
        tr_id = _TR_ID_BUY[self.mode] if side == "buy" else _TR_ID_SELL[self.mode]
        headers = self._auth_headers(tr_id)
        headers["hashkey"] = self._hashkey(body)

        resp = self._session.post(self._url(_ORDER_PATH), json=body, headers=headers, timeout=self.timeout)
        _raise_for_status_with_body(resp)
        data = resp.json()
        _check_rt_cd(data)
        output = data.get("output", {})

        return OrderResult(
            ticker=ticker,
            side=side,
            status="accepted",
            order_id=output.get("ODNO"),
            submitted_qty=float(qty_int),
            submitted_notional=None,
            # KIS 주문 응답은 접수 확인만 담고 즉시 체결 정보를 주지 않는다(체결
            # 여부는 별도 체결조회 API가 필요) — filled_* 없음은 알려진 한계.
            filled_qty=None,
            filled_avg_price=None,
            raw=data,
        )

    def is_market_open(self) -> bool:
        """휴장일(설/추석 등 국공휴일) 여부는 확인하지 않는다 — `chk-holiday`
        (TR_ID CTCA0903R)를 모의투자 도메인에 실제로 호출해보니 KIS가
        `rt_cd="1", msg_cd="EGW02006", msg1="모의투자 TR이 아닙니다"`로
        거부했고, KIS 공식 Postman 샘플에도 이 TR이 `[실전투자]`로만 표기돼
        있다(실전 도메인 전용으로 보임). 이걸 풀려면 실전 앱키가 있어야 하는데
        "모의투자만 먼저" 원칙과 맞지 않아 채택하지 않았다(2026-08-20 사용자
        결정). 대신 평일+장중시간(KST)만 로컬로 확인하고, 실제 국공휴일에
        `execute_kr.py`가 돌아가더라도 KIS 주문 API 자체가 명확한 에러로
        거부하는 걸 안전망으로 삼는다(조용히 잘못된 주문이 나가지 않음 —
        `_raise_for_status_with_body`가 실패 사유를 그대로 드러냄)."""
        now_kst = pd.Timestamp.now(tz="Asia/Seoul")
        if now_kst.dayofweek >= 5:  # 5=토, 6=일
            return False
        return _is_within_market_hours(now_kst)

    def count_open_orders(self) -> int:
        """정정취소가능주문조회(TR_ID VTTC8036R/TTTC8036R — 재확인 필요, 모듈
        docstring 참고)로 아직 미체결로 열려 있는 주문 수를 센다. execute_kr.py
        가 주문 제출 후 체결을 기다린 다음 state 를 저장하도록 하기 위함
        (US 트랙에서 이 절차 없이 저장했다가 다음날 재구성 체크가 오판한
        일이 있었음 — base.py `count_open_orders` docstring 참고). 페이지네이션은
        무시한다 — 이 프로젝트의 하루 주문 수(<=120)면 첫 페이지로 충분하다."""
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "INQR_DVSN_1": "0",
            "INQR_DVSN_2": "0",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        resp = self._session.get(
            self._url(_OPEN_ORDERS_PATH),
            params=params,
            headers=self._auth_headers(_TR_ID_OPEN_ORDERS[self.mode]),
            timeout=self.timeout,
        )
        _raise_for_status_with_body(resp)
        data = resp.json()
        _check_rt_cd(data)
        return len(data.get("output", []))
