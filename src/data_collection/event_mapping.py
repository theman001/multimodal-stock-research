"""KR(DART)/US(EDGAR) ticker → corp_code/CIK 매핑.

Phase 2 이벤트(공시) 수집의 첫 단계 — plan/08_phase2_news_events.md 참고.
일반 뉴스 스크레이핑이 아니라 공식 오픈 API(DART/SEC EDGAR)만 사용한다.

- US: SEC의 `company_tickers.json`은 키가 필요 없지만, SEC 정책상 요청마다
  식별 가능한 User-Agent(연락처 포함)를 보내야 한다 — `SEC_EDGAR_USER_AGENT`.
- KR: DART `corpCode.xml`은 상장/비상장 전체 법인 목록(수만 건)을 zip으로
  내려주며, `DART_API_KEY`(무료 가입, opendart.fss.or.kr)가 필요하다.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
DART_CORP_CODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"

# SEC company_tickers.json은 "현재 시점" 스냅샷만 준다 — 지주회사 전환/합병 등으로
# CIK가 바뀐 종목은 옛 CIK의 이력이 자동으로 안 딸려온다. 60종목 전수 감사(2026-08-12,
# 각 CIK의 filings.files/formerNames로 등록 시점을 대조)로 발견한 케이스만 수동 등록.
# ticker -> 병합해야 할 선행 CIK 목록 (같은 사업의 이전 법인격 — 필터링 대상 아님)
PREDECESSOR_CIKS: dict[str, list[int]] = {
    "XOM": [34088],  # Exxon Mobil Corp 본체 — 2026 지주회사 전환("ExxonMobil Holdings Corp") 전 이력
    "AVGO": [1441634],  # Broadcom Limited(싱가포르) — 2018-04 미국 재편입("Broadcom Inc.") 전 이력
    "GOOG": [1288776],  # Google Inc. — 2015-10 Alphabet 지주회사 전환 전 이력
    "GOOGL": [1288776],
    "FTI": [1135152],  # FMC Technologies Inc — 2017-01 TechnipFMC 합병 전 이력
    "VNOM": [1602065],  # Viper Energy Partners LP/Inc 본체 — 2025-08 신설 지주회사 전 이력
}

# 상장폐지 후 티커 심볼이 무관한 회사에 재배정되어, 현재 스냅샷 매칭이 아예 다른
# 회사를 가리키는 경우 — 자동 매칭 대신 수동 확인한 (cik, title)로 전면 교체한다.
# title도 함께 override해야 한다 — cik만 바꾸고 title을 자동매칭 결과("Everpure,
# Inc.")로 남겨두면 같은 행 안에서 cik와 title이 서로 다른 회사를 가리키는
# 자기모순이 생긴다(round 2 재검토에서 실제로 발견).
TICKER_CIK_OVERRIDES: dict[str, tuple[int, str]] = {
    "P": (1230276, "Pandora Media, Inc."),  # 2019 Sirius XM 인수로 상폐 후 'P' 티커가 무관한 회사에 재배정됨
}


def _sec_user_agent() -> str:
    ua = os.environ.get("SEC_EDGAR_USER_AGENT")
    if not ua:
        raise RuntimeError(
            "SEC_EDGAR_USER_AGENT 환경변수가 설정되지 않음 (.env 확인) — "
            "SEC는 식별 가능한 User-Agent(연락처 포함)를 요구한다"
        )
    return ua


def _dart_api_key() -> str:
    api_key = os.environ.get("DART_API_KEY")
    if not api_key:
        raise RuntimeError("DART_API_KEY 환경변수가 설정되지 않음 (.env 확인)")
    return api_key


def _parse_sec_company_tickers(raw: dict) -> pd.DataFrame:
    """company_tickers.json(순차 정수 키의 dict-of-dict)을 표 형태로 변환한다."""
    df = pd.DataFrame.from_dict(raw, orient="index")
    df["ticker"] = df["ticker"].str.upper()
    return df[["ticker", "cik_str", "title"]].rename(columns={"cik_str": "cik"})


def _match_us_tickers(tickers: list[str], lookup: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """SEC 표기(BRK-B vs BRK.B 등 구두점 차이)를 흡수해가며 매칭한다."""
    by_ticker = lookup.set_index("ticker")
    rows = []
    missing = []
    for t in tickers:
        upper = t.upper()
        candidates = [upper, upper.replace("-", "."), upper.replace(".", "-")]
        found_key = next((c for c in candidates if c in by_ticker.index), None)
        if found_key is None:
            missing.append(t)
            continue
        row = by_ticker.loc[found_key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        cik, title = TICKER_CIK_OVERRIDES.get(t, (int(row["cik"]), row["title"]))
        rows.append({"ticker": t, "cik": cik, "title": title})
    return pd.DataFrame(rows, columns=["ticker", "cik", "title"]), missing


def fetch_us_ticker_cik_map(tickers: list[str], raw_dir: Path) -> pd.DataFrame:
    """SEC company_tickers.json에서 ticker -> CIK 매핑을 가져온다 (캐시 우선).

    매핑되지 않는 종목이 하나라도 있으면 즉시 raise한다 — Phase 1의
    "point-in-time 라벨링 결측 0건" 원칙과 동일하게, 조용히 일부 종목을
    빠뜨리고 넘어가지 않는다.
    """
    events_dir = raw_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_path = events_dir / "company_tickers_raw.json"

    if raw_cache_path.exists():
        raw = json.loads(raw_cache_path.read_text(encoding="utf-8"))
    else:
        resp = requests.get(SEC_TICKERS_URL, headers={"User-Agent": _sec_user_agent()}, timeout=30)
        resp.raise_for_status()
        raw_cache_path.write_text(resp.text, encoding="utf-8")
        raw = resp.json()

    lookup = _parse_sec_company_tickers(raw)
    result, missing = _match_us_tickers(tickers, lookup)
    if missing:
        raise RuntimeError(f"SEC CIK 매핑 실패 종목: {missing}")

    result.to_parquet(events_dir / "ticker_cik_map.parquet", index=False)
    return result


def _parse_dart_corp_code_xml(xml_bytes: bytes) -> pd.DataFrame:
    """corpCode.xml(상장/비상장 전체)을 파싱해 상장종목(stock_code 존재)만 남긴다."""
    root = ElementTree.fromstring(xml_bytes)
    rows = []
    for el in root.findall("list"):
        stock_code = (el.findtext("stock_code") or "").strip()
        if not stock_code:
            continue  # 비상장/코드 없음 — 우리 유니버스는 전부 상장종목
        rows.append(
            {
                "ticker": stock_code,
                "corp_code": el.findtext("corp_code"),
                "corp_name": el.findtext("corp_name"),
            }
        )
    return pd.DataFrame(rows, columns=["ticker", "corp_code", "corp_name"])


def _match_kr_tickers(tickers: list[str], all_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """종목코드를 직접 매칭하고, 실패하면 마지막 자리를 0으로 바꿔 재시도한다.

    DART corp_code는 법인 단위라 우선주 티커(예: 005935 삼성전자우)는 보통주
    (005930 삼성전자)와 corp_code를 공유하며, corpCode.xml에는 보통주 코드만
    등록된 경우가 실제로 있다(005935/003545에서 확인됨). 이 재시도로도 못 찾으면
    추측하지 않고 missing으로 보고한다.
    """
    lookup = all_df.set_index("ticker")
    rows = []
    missing = []
    for t in tickers:
        key = t if t in lookup.index else None
        if key is None:
            common_candidate = t[:-1] + "0"
            if common_candidate in lookup.index:
                key = common_candidate
        if key is None:
            missing.append(t)
            continue
        row = lookup.loc[key]
        rows.append({"ticker": t, "corp_code": row["corp_code"], "corp_name": row["corp_name"]})
    result = pd.DataFrame(rows, columns=["ticker", "corp_code", "corp_name"])
    return result, missing


def fetch_kr_ticker_corpcode_map(tickers: list[str], raw_dir: Path) -> pd.DataFrame:
    """DART corpCode.xml에서 ticker(종목코드) -> corp_code 매핑을 가져온다 (캐시 우선).

    매핑되지 않는 종목이 하나라도 있으면 즉시 raise한다.
    """
    events_dir = raw_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_path = events_dir / "corpCode.xml"

    if not raw_cache_path.exists():
        api_key = _dart_api_key()
        resp = requests.get(DART_CORP_CODE_URL, params={"crtfc_key": api_key}, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            xml_bytes = zf.read(zf.namelist()[0])
        raw_cache_path.write_bytes(xml_bytes)

    all_df = _parse_dart_corp_code_xml(raw_cache_path.read_bytes())
    result, missing = _match_kr_tickers(tickers, all_df)
    if missing:
        raise RuntimeError(f"DART corp_code 매핑 실패 종목: {missing}")

    result.to_parquet(events_dir / "ticker_corp_code_map.parquet", index=False)
    return result
