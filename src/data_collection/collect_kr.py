"""Phase 1 국내(KR) 데이터 수집 진입점 (KRX Open API 기반).

1. 정기변경 기간(2014년말~현재) point-in-time 순위 테이블 생성/로드
2. 최신 정기변경 기준 대형/중형/소형 각 20종목(총 60종목) 유니버스 선정
3. 전체 기간(2015-01-01~현재) 일별 전종목 스냅샷을 캐시에 채운 뒤,
   60종목 각각의 OHLCV를 캐시에서 추출 + point-in-time size_id 라벨 부여

산출물: ${DATA_ROOT}/processed/ohlcv_meta_kr.parquet
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.config import get_data_root
from src.data_collection.kr_universe import (
    build_point_in_time_rankings,
    detect_and_adjust_splits,
    fetch_daily_snapshot,
    generate_rebalance_periods,
    label_point_in_time,
    select_kr_universe,
)

START_DATE = pd.Timestamp("2015-01-01")
REBALANCE_START_YEAR = 2014  # 2015-01-01 시점에 유효한 분류를 보장하려면 그 이전 정기변경(2014-12)부터 필요

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


def build_or_load_rankings(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = data_root / "processed"
    rankings_path = processed_dir / "kr_point_in_time_rankings.parquet"
    periods_path = processed_dir / "kr_rebalance_periods.parquet"
    raw_dir = data_root / "raw" / "kr"

    if rankings_path.exists() and periods_path.exists():
        return pd.read_parquet(rankings_path), pd.read_parquet(periods_path)

    end_date = pd.Timestamp(date.today())
    periods = generate_rebalance_periods(REBALANCE_START_YEAR, end_date, raw_dir)
    rankings = build_point_in_time_rankings(periods, raw_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    rankings.to_parquet(rankings_path, index=False)
    periods.to_parquet(periods_path, index=False)
    return rankings, periods


def fetch_all_daily_snapshots(start_date: pd.Timestamp, end_date: pd.Timestamp, raw_dir: Path) -> None:
    """OHLCV 추출에 필요한 전체 기간의 일별 스냅샷을 캐시에 채운다 (이미 캐시된 날짜는 스킵)."""
    d = start_date
    total = (end_date - start_date).days + 1
    count = 0
    while d <= end_date:
        fetch_daily_snapshot(d, raw_dir)
        d += pd.Timedelta(days=1)
        count += 1
        if count % 500 == 0:
            print(f"[collect_kr] 일별 스냅샷 캐시 진행: {count}/{total}")


def extract_universe_ohlcv(
    tickers: set[str], start_date: pd.Timestamp, end_date: pd.Timestamp, raw_dir: Path
) -> pd.DataFrame:
    """캐시된 일별 전종목 스냅샷을 한 번만 순회하며 유니버스 전체 종목의 OHLCV를 동시에 뽑아낸다.

    종목별로 전체 기간을 따로 훑으면 (종목 수 × 일 수)만큼 캐시를 다시 읽어야 해서
    비효율적이므로, 하루치 스냅샷을 한 번 읽을 때 유니버스에 속한 종목들을 한 번에
    걸러낸다.
    """
    rows = []
    d = start_date
    while d <= end_date:
        snapshot = fetch_daily_snapshot(d, raw_dir)
        if not snapshot.empty:
            matched = snapshot[snapshot["ticker"].isin(tickers)]
            if not matched.empty:
                matched = matched.copy()
                matched["date"] = d
                rows.append(matched)
        d += pd.Timedelta(days=1)
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume", "market_cap"])
    return pd.concat(rows, ignore_index=True)


def collect_kr(data_root: Path | None = None) -> pd.DataFrame:
    data_root = data_root or get_data_root()
    raw_dir = data_root / "raw" / "kr"
    end_date = pd.Timestamp(date.today())

    rankings, periods = build_or_load_rankings(data_root)

    fetch_range_start = min(periods["window_start"].min(), START_DATE)
    fetch_all_daily_snapshots(fetch_range_start, end_date, raw_dir)

    universe = select_kr_universe(rankings)
    tickers = set(universe["ticker"])
    market_id_by_ticker = dict(zip(universe["ticker"], universe["market_id"]))

    all_ohlcv = extract_universe_ohlcv(tickers, START_DATE, end_date, raw_dir)
    all_ohlcv = detect_and_adjust_splits(all_ohlcv)

    rows = []
    missing_tickers = []
    dropped_no_label_tickers = []
    for ticker in sorted(tickers):
        ohlcv = all_ohlcv[all_ohlcv["ticker"] == ticker]
        if ohlcv.empty:
            missing_tickers.append(ticker)
            continue

        labeled = label_point_in_time(ohlcv, ticker, rankings)
        if labeled.empty:
            dropped_no_label_tickers.append(ticker)
            continue

        labeled = labeled.copy()
        labeled["ticker"] = ticker
        labeled["market_id"] = market_id_by_ticker[ticker]
        rows.append(labeled)

    if missing_tickers:
        print(f"[collect_kr] OHLCV 수집 실패 {len(missing_tickers)}종목 (스킵): {missing_tickers}")
    if dropped_no_label_tickers:
        print(f"[collect_kr] point-in-time 라벨 없음 (스킵): {dropped_no_label_tickers}")

    if not rows:
        raise RuntimeError("KR OHLCV 수집 결과가 비어 있음 — 전체 실패")

    combined = pd.concat(rows, ignore_index=True)[OUTPUT_COLUMNS]

    processed_dir = data_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "ohlcv_meta_kr.parquet"
    combined.to_parquet(output_path, index=False)

    print(f"[collect_kr] {combined['ticker'].nunique()}종목, {len(combined)}행 저장 -> {output_path}")
    return combined


if __name__ == "__main__":
    collect_kr()
