"""DART(금융감독원) 공시 수집 — plan/08_phase2_news_events.md 참고.

corp_code별 주요사항보고서·수시공시(`pblntf_ty` B/I)를 `list.json`으로
가져와 raw 캐싱한다. DART 응답에는 접수 시각(시:분:초)이 없다 —
`rcept_dt`(날짜)만 제공하므로, 타임스탬프 정렬 규칙(module 2)에서 보수적으로
D+1(접수일 다음 거래일부터 반영)로 처리한다 — CLAUDE.md "Phase 2 구체 사양" 참고.

**주의**: `pblntf_ty`는 응답 `list` 배열에 없는, 요청 시에만 쓰는 필터
파라미터다(API 문서로 확인 완료 — 응답 필드는 corp_cls/corp_name/corp_code/
stock_code/report_nm/rcept_no/flr_nm/rcept_dt/rm뿐). 그래서 값 하나당
별도 요청을 보내야 하며, 응답을 받은 뒤 클라이언트에서 걸러낼 방법이 없다
(처음에 이걸 몰라서 필터 없이 전체 유형을 받아온 뒤 클라이언트에서 거르려다
실패한 적이 있음 — 존재하지 않는 컬럼으로 걸러서 사실상 무필터였음).

`list.json`은 페이지당 최대 100건, 조회기간 제한은 `corp_code` 지정 시
없음(전체 corp_code 미지정 검색만 3개월 제한 — API 문서로 확인 완료).

DART_API_KEY는 KRX Open API와 마찬가지로 opendart.fss.or.kr 무료 가입 후
발급받아야 한다.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data_collection.event_mapping import _dart_api_key

load_dotenv()

DART_LIST_URL = "https://opendart.fss.or.kr/api/list.json"
DART_REQUEST_DELAY_SECONDS = 0.2
TARGET_PBLNTF_TY = ["B", "I"]  # B=주요사항보고, I=거래소공시(수시공시) — 정기공시/지분공시 등은 제외
START_DATE = "20150101"
PAGE_COUNT = 100
_STATUS_OK = "000"
_STATUS_NO_DATA = "013"  # 조회된 데이터 없음 — 에러가 아니라 정상 케이스(공시가 없는 기간)

_KEEP_COLUMNS = ["ticker", "corp_code", "pblntf_ty", "report_nm", "rcept_no", "rcept_dt", "flr_nm"]


def _fetch_list_page(
    corp_code: str, pblntf_ty: str, bgn_de: str, end_de: str, page_no: int, api_key: str
) -> dict:
    resp = requests.get(
        DART_LIST_URL,
        params={
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "pblntf_ty": pblntf_ty,
            "bgn_de": bgn_de,
            "end_de": end_de,
            "page_no": page_no,
            "page_count": PAGE_COUNT,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_list_response(payload: dict, ticker: str, pblntf_ty: str) -> pd.DataFrame:
    """status 013(조회 결과 없음)은 정상 케이스로 빈 DataFrame을 반환한다.

    000/013 외의 상태 코드(트래픽 초과, 시스템 점검, 인증키 오류 등)는
    조용히 빈 결과로 넘기지 않고 즉시 raise한다 — 진짜 에러를 "공시 없음"으로
    착각하면 이후 피처 단계에서 결측을 오탐하게 된다.
    """
    status = payload.get("status")
    if status == _STATUS_NO_DATA:
        return pd.DataFrame(columns=_KEEP_COLUMNS)
    if status != _STATUS_OK:
        raise RuntimeError(f"DART API 오류 (status={status}): {payload.get('message')}")

    rows = payload.get("list", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=_KEEP_COLUMNS)
    df["ticker"] = ticker
    df["pblntf_ty"] = pblntf_ty  # 응답에 없는 필드 — 요청 파라미터 값을 그대로 태깅
    return df[[c for c in _KEEP_COLUMNS if c in df.columns]]


def fetch_kr_disclosures(
    corp_code: str, ticker: str, raw_dir: Path, start_date: str = START_DATE, force: bool = False
) -> pd.DataFrame:
    """corp_code의 주요사항보고+수시공시(TARGET_PBLNTF_TY) 이력을 raw 캐싱한다 (캐시 우선)."""
    cache_path = raw_dir / "events" / f"disclosures_{ticker}.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    api_key = _dart_api_key()
    end_date = pd.Timestamp.today().strftime("%Y%m%d")
    frames = []
    for pblntf_ty in TARGET_PBLNTF_TY:
        page_no = 1
        while True:
            payload = _fetch_list_page(corp_code, pblntf_ty, start_date, end_date, page_no, api_key)
            frames.append(_parse_list_response(payload, ticker, pblntf_ty))
            time.sleep(DART_REQUEST_DELAY_SECONDS)
            total_page = payload.get("total_page", 0)
            if page_no >= total_page:
                break
            page_no += 1

    frames = [f for f in frames if not f.empty]
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_KEEP_COLUMNS)
    result = result.drop_duplicates(subset="rcept_no").sort_values("rcept_dt").reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path, index=False)
    return result
