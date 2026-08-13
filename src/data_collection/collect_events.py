"""Phase 2 이벤트(공시) 수집 진입점 — KR(DART)/US(EDGAR).

Phase 1이 이미 point-in-time 검증까지 끝낸 120종목 유니버스
(`processed/ohlcv_meta_{us,kr}.parquet`)를 그대로 재사용한다 — 이벤트
축에서 새로 유니버스를 선정하지 않는다. round 3 리뷰에서 이 오케스트레이션
스크립트 자체가 없어서(수집이 1회성 스크립트로만 진행돼 재현 불가능한 상태)
발견된 공백을 메우기 위해 작성했다.

각 종목의 실제 원문 수집(raw 캐싱)은 `event_mapping.py`/`events_us.py`/
`events_kr.py`의 `fetch_*` 함수가 이미 캐시 우선으로 처리하므로, 여기서는
그 함수들을 유니버스 전체에 대해 호출하고 결과를 하나로 합쳐
`processed/events_{us,kr}.parquet`에 저장하는 조립 역할만 한다.

산출물: ${DATA_ROOT}/processed/events_us.parquet, events_kr.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import get_data_root
from src.data_collection.event_mapping import fetch_kr_ticker_corpcode_map, fetch_us_ticker_cik_map
from src.data_collection.events_kr import fetch_kr_disclosures
from src.data_collection.events_us import fetch_us_filings


def _load_universe_tickers(data_root: Path, market_file: str) -> list[str]:
    df = pd.read_parquet(data_root / "processed" / market_file, columns=["ticker"])
    return sorted(df["ticker"].unique())


def collect_events_us(data_root: Path | None = None) -> pd.DataFrame:
    data_root = data_root or get_data_root()
    raw_dir = data_root / "raw" / "us"
    tickers = _load_universe_tickers(data_root, "ohlcv_meta_us.parquet")

    cik_map = fetch_us_ticker_cik_map(tickers, raw_dir)  # 매핑 실패 종목이 있으면 여기서 즉시 raise

    rows = []
    failed_tickers = []
    for _, u in cik_map.iterrows():
        try:
            rows.append(fetch_us_filings(cik=u["cik"], ticker=u["ticker"], raw_dir=raw_dir))
        except Exception as e:
            failed_tickers.append((u["ticker"], str(e)))

    if failed_tickers:
        print(f"[collect_events_us] 수집 실패 {len(failed_tickers)}종목 (스킵, 다음 실행 시 캐시 없어 재시도됨): {failed_tickers}")
    if not rows:
        raise RuntimeError("US 이벤트 수집 결과가 비어 있음 — 전체 실패")

    combined = pd.concat(rows, ignore_index=True)

    processed_dir = data_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "events_us.parquet"
    combined.to_parquet(output_path, index=False)

    print(f"[collect_events_us] {combined['ticker'].nunique()}종목, {len(combined)}건 저장 -> {output_path}")
    return combined


def collect_events_kr(data_root: Path | None = None) -> pd.DataFrame:
    data_root = data_root or get_data_root()
    raw_dir = data_root / "raw" / "kr"
    tickers = _load_universe_tickers(data_root, "ohlcv_meta_kr.parquet")

    corp_map = fetch_kr_ticker_corpcode_map(tickers, raw_dir)  # 매핑 실패 종목이 있으면 여기서 즉시 raise

    rows = []
    failed_tickers = []
    for _, u in corp_map.iterrows():
        try:
            rows.append(fetch_kr_disclosures(corp_code=u["corp_code"], ticker=u["ticker"], raw_dir=raw_dir))
        except Exception as e:
            failed_tickers.append((u["ticker"], str(e)))

    if failed_tickers:
        print(f"[collect_events_kr] 수집 실패 {len(failed_tickers)}종목 (스킵, 다음 실행 시 캐시 없어 재시도됨): {failed_tickers}")
    if not rows:
        raise RuntimeError("KR 이벤트 수집 결과가 비어 있음 — 전체 실패")

    combined = pd.concat(rows, ignore_index=True)

    processed_dir = data_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "events_kr.parquet"
    combined.to_parquet(output_path, index=False)

    print(f"[collect_events_kr] {combined['ticker'].nunique()}종목, {len(combined)}건 저장 -> {output_path}")
    return combined


if __name__ == "__main__":
    collect_events_us()
    collect_events_kr()
