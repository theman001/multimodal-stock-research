"""SEC EDGAR 공시(8-K 중심) 수집 — plan/08_phase2_news_events.md 참고.

CIK별 전체 filing 이력을 가져와 대상 form 타입만 필터링해 raw 캐싱한다.
타임스탬프 정렬 규칙(당일/익일 배정)은 이 모듈에서 다루지 않는다 — 여기서는
`acceptanceDateTime`(초 단위)을 그대로 보존해 다음 모듈(정렬 규칙)에 넘긴다.

`submissions/CIK##########.json`의 `filings.recent`는 최근 항목만 담고
있고(회사당 최대 1000여 건), 그 이전 이력은 `filings.files`에 나열된 보조
JSON에 나뉘어 있다 — 2015-01-01부터 전체 이력을 채우려면 두 소스를 합쳐야
한다(AAPL 등 상장 이력이 긴 종목에서 실제로 확인됨).

또한 SEC의 ticker->CIK는 "현재 시점" 스냅샷이라, 지주회사 전환/합병으로 CIK가
바뀐 종목은 자동 매핑만으로는 옛 CIK 이력이 빠진다 — `event_mapping.PREDECESSOR_CIKS`에
등록된 종목은 선행 CIK의 filing도 함께 가져와 병합한다(감사 경위는 그 모듈 docstring 참고).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from src.data_collection.event_mapping import PREDECESSOR_CIKS, _sec_user_agent

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
SEC_REQUEST_DELAY_SECONDS = 0.15  # SEC fair-access 권장 10 req/s 이하
TARGET_FORMS = {"8-K"}
START_DATE = "2015-01-01"

_KEEP_COLUMNS = [
    "ticker",
    "cik",
    "form",
    "filingDate",
    "acceptanceDateTime",
    "accessionNumber",
    "primaryDocument",
    "items",
]


def _filings_block_to_df(block: dict, cik: int) -> pd.DataFrame:
    """'recent' 또는 보조 파일의 컬럼형(dict-of-list) 구조를 표로 변환한다."""
    if not block or not block.get("form"):
        return pd.DataFrame(columns=_KEEP_COLUMNS)
    df = pd.DataFrame(block)
    df["cik"] = cik
    return df


def _filter_target_filings(df: pd.DataFrame, ticker: str, start_date: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=_KEEP_COLUMNS)
    df = df[df["form"].isin(TARGET_FORMS) & (df["filingDate"] >= start_date)].copy()
    df["ticker"] = ticker
    return df[[c for c in _KEEP_COLUMNS if c in df.columns]].reset_index(drop=True)


def _fetch_all_filings_for_cik(cik: int, start_date: str, headers: dict) -> pd.DataFrame:
    """한 CIK의 'recent' + 보조 파일 전체를 합쳐 form 필터링 없이 반환한다."""
    resp = requests.get(SEC_SUBMISSIONS_URL.format(cik=cik), headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    time.sleep(SEC_REQUEST_DELAY_SECONDS)

    frames = [_filings_block_to_df(payload["filings"]["recent"], cik)]
    for extra in payload["filings"].get("files", []):
        if extra["filingTo"] < start_date:
            continue  # 이 보조 파일 전체가 우리 대상 기간보다 앞선 이력 — 스킵
        extra_resp = requests.get(
            SEC_SUBMISSIONS_FILE_URL.format(name=extra["name"]), headers=headers, timeout=30
        )
        extra_resp.raise_for_status()
        frames.append(_filings_block_to_df(extra_resp.json(), cik))
        time.sleep(SEC_REQUEST_DELAY_SECONDS)

    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_KEEP_COLUMNS)


def fetch_us_filings(
    cik: int, ticker: str, raw_dir: Path, start_date: str = START_DATE, force: bool = False
) -> pd.DataFrame:
    """CIK(+PREDECESSOR_CIKS 등록분)의 filing 이력에서 TARGET_FORMS만 남겨 raw 캐싱한다 (캐시 우선)."""
    cache_path = raw_dir / "events" / f"filings_{ticker}.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    headers = {"User-Agent": _sec_user_agent()}
    all_ciks = [cik] + PREDECESSOR_CIKS.get(ticker, [])
    frames = [_fetch_all_filings_for_cik(c, start_date, headers) for c in all_ciks]
    frames = [f for f in frames if not f.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_KEEP_COLUMNS)
    result = _filter_target_filings(combined, ticker, start_date)
    result = result.drop_duplicates(subset="accessionNumber").sort_values("filingDate").reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path, index=False)
    return result
