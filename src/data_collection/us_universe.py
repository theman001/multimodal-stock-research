"""S&P 500/400/600 구성종목 수집 및 유니버스 선정 (해외).

이 모듈은 "어떤 60종목을 유니버스로 쓸지"만 정한다(현재 시점 지수 구성 +
시가총액 상위 기준, point-in-time 커버리지 필터 포함). 각 종목의 point-in-time
size_id 라벨링 자체는 us_point_in_time.py가 Wikipedia 변경이력을 재구성해
담당한다 — 더 이상 size_id가 종목별 상수가 아니다 (과거에는 그랬으나
01_data_collection.md에 예고된 대로 point-in-time 재구성으로 교체됨).
"""
from __future__ import annotations

import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()


def wiki_headers() -> dict:
    """Wikipedia 요청용 User-Agent — 연락처 포함 문자열을 코드에 하드코딩하지
    않고 `.env`의 `WIKI_USER_AGENT`에서 읽는다(round 3 이벤트 축 리뷰에서
    이메일이 소스코드에 그대로 박혀있던 걸 발견해 SEC_EDGAR_USER_AGENT와
    동일한 패턴으로 분리)."""
    ua = os.environ.get("WIKI_USER_AGENT")
    if not ua:
        raise RuntimeError("WIKI_USER_AGENT 환경변수가 설정되지 않음 (.env 확인)")
    return {"User-Agent": ua}


SP_INDEX_PAGES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}

# CLAUDE.md 분류 기준: S&P500=대형(2), S&P400=중형(1), S&P600=소형(0)
SIZE_ID_BY_INDEX = {"sp500": 2, "sp400": 1, "sp600": 0}

UNIVERSE_SIZE_PER_INDEX = 20


def _normalize_ticker(ticker: str) -> str:
    """Wikipedia는 'BRK.B' 표기, yfinance/Yahoo는 'BRK-B' 표기를 씀."""
    return str(ticker).strip().replace(".", "-")


def fetch_index_constituents(index_key: str, raw_dir: Path, force: bool = False) -> list[str]:
    """Wikipedia에서 지수 구성종목 티커 목록을 가져온다 (raw HTML 캐시)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / f"wikipedia_{index_key}.html"

    if cache_path.exists() and not force:
        html = cache_path.read_text(encoding="utf-8")
    else:
        resp = requests.get(SP_INDEX_PAGES[index_key], headers=wiki_headers(), timeout=20)
        resp.raise_for_status()
        html = resp.text
        cache_path.write_text(html, encoding="utf-8")

    table = pd.read_html(StringIO(html))[0]
    return [_normalize_ticker(t) for t in table["Symbol"].tolist()]


def fetch_market_caps(
    tickers: list[str],
    raw_dir: Path,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.0,
    request_delay_seconds: float = 0.1,
) -> pd.DataFrame:
    """티커별 현재 시가총액을 yfinance로 조회한다 (캐시 + 재시도 + 실패 스킵).

    이미 raw_dir/market_cap_cache.parquet에 있는 티커는 재요청하지 않는다.
    실패한 티커는 skip하고 로그만 남긴다 (전체 파이프라인 중단 방지, CLAUDE.md 리스크 대응).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / "market_cap_cache.parquet"

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
    else:
        cached = pd.DataFrame(columns=["ticker", "market_cap"])

    cached_tickers = set(cached["ticker"])
    to_fetch = [t for t in dict.fromkeys(tickers) if t not in cached_tickers]

    rows = []
    failed = []
    for ticker in to_fetch:
        market_cap = None
        for attempt in range(max_retries):
            try:
                info = yf.Ticker(ticker).get_info()
                market_cap = info.get("marketCap")
                break
            except Exception:
                time.sleep(retry_backoff_seconds * (2**attempt))
        if market_cap is None:
            failed.append(ticker)
        else:
            rows.append({"ticker": ticker, "market_cap": market_cap})
        time.sleep(request_delay_seconds)

    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached.to_parquet(cache_path, index=False)

    if failed:
        preview = failed[:20]
        suffix = "..." if len(failed) > 20 else ""
        print(f"[us_universe] market cap 조회 실패 {len(failed)}건 (스킵): {preview}{suffix}")

    return cached[cached["ticker"].isin(tickers)].reset_index(drop=True)


MIN_COVERAGE_DAYS = 200  # point-in-time 라벨이 검증되는 날짜가 이보다 적으면 후보에서 제외
# (MA60 워밍업 + 최소한의 학습 표본 확보 목적 — 순수 시가총액 기준으로만 뽑으면
#  현재는 크지만 지수 편입 이력이 거의 없는 종목이 뽑혀 피처 엔지니어링에서 전부
#  소실될 수 있다)


def select_us_universe(
    raw_dir: Path,
    universe_size: int = UNIVERSE_SIZE_PER_INDEX,
    index_intervals: dict | None = None,
    coverage_start: pd.Timestamp | None = None,
    coverage_end: pd.Timestamp | None = None,
    min_coverage_days: int = MIN_COVERAGE_DAYS,
) -> pd.DataFrame:
    """S&P 500/400/600 각각에서, point-in-time 커버리지가 충분한 후보 중
    시가총액 상위 universe_size개씩 선정.

    index_intervals가 주어지면(us_point_in_time.build_all_index_intervals 결과)
    커버리지 부족 후보를 걸러낸 뒤 순위를 매긴다. 주어지지 않으면(하위호환)
    커버리지 필터 없이 순수 시가총액 순으로 선정한다.

    반환 컬럼: ticker, market_id(=0 고정), size_id, source_index, market_cap
    """
    from src.data_collection.us_point_in_time import compute_coverage_days

    frames = []
    for index_key in ("sp500", "sp400", "sp600"):
        constituents = fetch_index_constituents(index_key, raw_dir)
        caps = fetch_market_caps(constituents, raw_dir)
        caps = caps.dropna(subset=["market_cap"])

        if index_intervals is not None:
            caps = caps.copy()
            caps["coverage_days"] = caps["ticker"].map(
                lambda t: compute_coverage_days(t, index_intervals, coverage_start, coverage_end)
            )
            caps = caps[caps["coverage_days"] >= min_coverage_days]

        top = caps.sort_values("market_cap", ascending=False).head(universe_size).copy()
        top["source_index"] = index_key
        top["size_id"] = SIZE_ID_BY_INDEX[index_key]
        frames.append(top)

    universe = pd.concat(frames, ignore_index=True)
    universe["market_id"] = 0
    return universe[["ticker", "market_id", "size_id", "source_index", "market_cap"]]
