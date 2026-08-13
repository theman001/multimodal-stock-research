"""Phase 1 데이터 수집 진입점 — 해외(US).

Wikipedia 변경이력 기반 point-in-time 구성종목 재구성을 적용한다
(us_point_in_time.py) — 01_data_collection.md에 명시된 survivorship bias
완화 후속 조치. 유니버스 60종목 선정 자체는 여전히 "현재 시점 상위 시가총액"
기준이지만(선정 로직은 그대로), 선정된 각 종목의 size_id는 실제 그 날짜에
어느 지수(S&P 500/400/600)에 속해 있었는지로 라벨링한다. 어느 지수에도
소속이 확인되지 않는 구간(특히 S&P 600은 변경이력이 2019-12부터만 존재)의
행은 라벨을 붙이지 않고 제외한다 — 추정 라벨 금지(leakage-guard 원칙).

산출물: ${DATA_ROOT}/processed/ohlcv_meta_us.parquet
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.config import get_data_root
from src.data_collection.ohlcv import fetch_us_ohlcv
from src.data_collection.us_point_in_time import build_all_index_intervals, label_point_in_time_us
from src.data_collection.us_universe import select_us_universe

START_DATE = "2015-01-01"

OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "market_id",
    "size_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "market_cap",
]


def collect_us(data_root: Path | None = None) -> pd.DataFrame:
    data_root = data_root or get_data_root()
    raw_dir = data_root / "raw" / "us"
    end_date = date.today().isoformat()

    index_intervals = build_all_index_intervals(raw_dir)
    universe = select_us_universe(
        raw_dir,
        index_intervals=index_intervals,
        coverage_start=pd.Timestamp(START_DATE),
        coverage_end=pd.Timestamp(end_date),
    )

    rows = []
    missing_tickers = []
    dropped_no_label_tickers = []
    for _, u in universe.iterrows():
        ticker = u["ticker"]
        ohlcv = fetch_us_ohlcv(ticker, START_DATE, end_date, raw_dir)
        if ohlcv is None or ohlcv.empty:
            missing_tickers.append(ticker)
            continue

        labeled = label_point_in_time_us(ohlcv, ticker, index_intervals)
        if labeled.empty:
            dropped_no_label_tickers.append(ticker)
            continue

        labeled = labeled.copy()
        labeled["ticker"] = ticker
        labeled["market_id"] = u["market_id"]
        labeled["market_cap"] = u["market_cap"]
        rows.append(labeled)

    if missing_tickers:
        print(f"[collect_us] OHLCV 수집 실패 {len(missing_tickers)}종목 (스킵): {missing_tickers}")
    if dropped_no_label_tickers:
        print(f"[collect_us] point-in-time 라벨 없음 (스킵): {dropped_no_label_tickers}")

    if not rows:
        raise RuntimeError("US OHLCV 수집 결과가 비어 있음 — 전체 실패")

    combined = pd.concat(rows, ignore_index=True)[OUTPUT_COLUMNS]

    processed_dir = data_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "ohlcv_meta_us.parquet"
    combined.to_parquet(output_path, index=False)

    print(f"[collect_us] {combined['ticker'].nunique()}종목, {len(combined)}행 저장 -> {output_path}")
    if missing_tickers:
        print(f"[collect_us] 유니버스 대비 수집 실패 {len(missing_tickers)}종목: {missing_tickers}")

    return combined


if __name__ == "__main__":
    collect_us()
