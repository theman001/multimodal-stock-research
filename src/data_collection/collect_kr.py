"""Phase 1 국내(KR) 데이터 수집 진입점 (KRX Open API 기반).

1. 정기변경 기간(2014년말~현재) point-in-time 순위 테이블 생성/로드
2. 최신 정기변경 기준 대형/중형/소형 각 20종목(총 60종목) 유니버스 선정
3. 전체 기간(2015-01-01~현재) 일별 전종목 스냅샷을 캐시에 채운 뒤,
   60종목 각각의 OHLCV를 캐시에서 추출 + point-in-time size_id 라벨 부여

산출물: ${DATA_ROOT}/processed/ohlcv_meta_kr.parquet
"""
from __future__ import annotations

import os
import tempfile
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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def fetch_all_daily_snapshots(
    start_date: pd.Timestamp, end_date: pd.Timestamp, raw_dir: Path, incremental: bool = True
) -> None:
    """OHLCV 추출에 필요한 전체 기간의 일별 스냅샷을 캐시에 채운다.

    `fetch_daily_snapshot()` 자체는 하루 단위로 이미 캐시 우선(캐시 있으면
    네트워크 호출 없이 반환)이라 새 날짜는 정상적으로 캐시에 쌓이지만, 이
    함수가 호출될 때마다 `start_date`(2014년 말)부터 `end_date`(오늘)까지
    ~4,000일 전체를 매번 다시 순회하며 이미 캐시된 옛날 parquet 파일들까지
    전부 다시 여는 게 문제였다 — 매일 도는 cron에서 누적되는 불필요한 I/O.
    `incremental=True`(기본값)면 지난 호출이 `raw_dir/stock_daily_trade/
    _last_processed_date.txt`에 기록해둔 마지막 처리 날짜 다음날부터만
    순회한다. `incremental=False`는 항상 `start_date`부터 전체를 순회한다
    (구 동작, 재현/디버깅용).
    """
    state_path = raw_dir / "stock_daily_trade" / "_last_processed_date.txt"
    d = start_date
    if incremental and state_path.exists():
        last_processed = pd.Timestamp(state_path.read_text().strip())
        resume_from = last_processed + pd.Timedelta(days=1)
        if resume_from > d:
            d = resume_from

    total = (end_date - d).days + 1 if d <= end_date else 0
    count = 0
    while d <= end_date:
        fetch_daily_snapshot(d, raw_dir)
        if incremental:
            _atomic_write_text(state_path, d.strftime("%Y-%m-%d"))
        d += pd.Timedelta(days=1)
        count += 1
        if count % 500 == 0:
            print(f"[collect_kr] 일별 스냅샷 캐시 진행: {count}/{total}")


def extract_universe_ohlcv(
    tickers: set[str], start_date: pd.Timestamp, end_date: pd.Timestamp, raw_dir: Path, incremental: bool = True
) -> pd.DataFrame:
    """캐시된 일별 전종목 스냅샷을 순회하며 유니버스 전체 종목의 OHLCV를 동시에 뽑아낸다.

    종목별로 전체 기간을 따로 훑으면 (종목 수 × 일 수)만큼 캐시를 다시 읽어야 해서
    비효율적이므로, 하루치 스냅샷을 한 번 읽을 때 유니버스에 속한 종목들을 한 번에
    걸러낸다.

    `incremental=True`(기본값)면 지난 호출의 추출 결과를
    (`raw_dir/kr_universe_ohlcv_cache.parquet`)를 재사용하고, 캐시의 마지막
    날짜 다음날부터만 스냅샷에서 새로 추출해 append한다 — 안 그러면 매일 도는
    cron이 ~4,000일치 스냅샷 parquet 파일을 매번 전부 다시 열어야 한다(게다가
    `fetch_all_daily_snapshots()`도 같은 범위를 독립적으로 순회하므로 중복
    낭비가 두 배가 된다). **유니버스(`tickers`)가 지난 호출과 달라지면**
    (정기변경 등으로 60종목 구성이 바뀌면) 캐시된 rows에는 새로 편입된
    종목의 과거 이력이 없으므로, 캐시를 무시하고 `start_date`부터 전체를
    재추출한다(정기변경은 연 2회뿐이라 이 경로의 비용은 감수할 만함).

    캐시에는 "지금까지 한 번이라도 요청됐던 가장 늦은 end_date"까지의 데이터가
    남아있을 수 있다 — 이번 호출의 `end_date`가 그보다 이르면(예: 과거 재현/
    디버깅 목적으로 오늘보다 이전 날짜를 넘기는 경우), 캐시를 그대로 반환하면
    `end_date`를 초과하는 미래 행이 섞여 나가는 계약 위반이 된다(구 while
    루프는 항상 `d <= end_date`로 직접 상한을 걸었으므로 이 문제 자체가 없었다
    — 캐싱 도입으로 새로 생긴 부작용). 반환 직전에 항상 `date <= end_date`로
    한 번 더 걸러낸다. 캐시 파일 자체는 잘라내지 않는다(미래에 더 큰 end_date로
    다시 호출될 때 그대로 재사용해야 하므로).
    """
    cache_path = raw_dir / "kr_universe_ohlcv_cache.parquet"
    ticker_set_path = raw_dir / "kr_universe_ohlcv_cache_tickers.txt"

    cached = None
    if incremental and cache_path.exists() and ticker_set_path.exists():
        cached_tickers = {t for t in ticker_set_path.read_text().splitlines() if t}
        if cached_tickers == tickers:
            cached = pd.read_parquet(cache_path)

    extract_start = start_date
    if cached is not None and not cached.empty:
        extract_start = pd.Timestamp(cached["date"].max()) + pd.Timedelta(days=1)
        if extract_start > end_date:
            return cached[cached["date"] <= end_date].reset_index(drop=True)

    empty_columns = ["date", "ticker", "open", "high", "low", "close", "volume", "market_cap"]
    rows = []
    d = extract_start
    while d <= end_date:
        snapshot = fetch_daily_snapshot(d, raw_dir)
        if not snapshot.empty:
            matched = snapshot[snapshot["ticker"].isin(tickers)]
            if not matched.empty:
                matched = matched.copy()
                matched["date"] = d
                rows.append(matched)
        d += pd.Timedelta(days=1)
    new_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=empty_columns)

    if cached is not None and not cached.empty:
        result = pd.concat([cached, new_df], ignore_index=True)
    else:
        result = new_df

    if incremental:
        _atomic_write_parquet(cache_path, result)
        _atomic_write_text(ticker_set_path, "\n".join(sorted(tickers)))

    return result


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
