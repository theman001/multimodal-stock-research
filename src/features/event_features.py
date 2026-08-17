"""Phase 2 모듈 4 — 이벤트 피처 집계 + features.parquet 병합.

`events_{us,kr}.parquet`(발행 메타)와 `events_sentiment_{us,kr}.parquet`(감성 점수,
Google Drive에서 내려받아 `${DATA_ROOT}/processed/`에 둔 파일)를 결합해 종목·거래일
단위 이벤트 피처 4종(CLAUDE.md "Phase 2 구체 사양" > "피처 결합")을 계산하고, 기존
`features.parquet`(Phase 1, 13개 기술지표 + target)에 병합한다.

타임스탬프 정렬(leakage-safe effective_date 배정)은 반드시
`src.data_collection.event_timing`의 `assign_us_effective_date`/
`assign_kr_effective_date`를 그대로 재사용한다 — 이 파일에서 새로 만들지 않는다.

leakage 방지 핵심 불변식: 종목 t의 어떤 피처든 "그 종목의 t번째(오름차순) 거래일
row_date"를 기준으로, `effective_date <= row_date`인 이벤트만 사용한다. 이는
`compute_features()`의 `rolling(...)`이 현재 행을 포함한 과거만 쓰는 것과 동일한
원칙이다(당일 마감 전 반영된 이벤트는 이미 "그 날의 정보"로 취급).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_data_root
from src.data_collection.event_timing import assign_kr_effective_date, assign_us_effective_date
from src.features.indicators import EVENT_FEATURE_COLUMNS

# days_since_last_event: "오늘 이벤트 발생"도 0이 될 수 있어 CLAUDE.md의 "이벤트
# 없는 날=0" 규칙을 그대로 쓰면 "이력 자체가 아직 없음"과 구분이 안 된다. 유효한
# days_since 값은 항상 0 이상이므로, 0과 절대 충돌하지 않는 -1을 이력-없음
# sentinel로 쓴다(구현 중 조정 가능 항목 — CLAUDE.md에 명시된 예외 재량).
NO_PRIOR_EVENT_SENTINEL = -1

_MARKET_CONFIG = {
    "us": {
        "meta_file": "events_us.parquet",
        "sentiment_file": "events_sentiment_us.parquet",
        "join_key": "accessionNumber",
        "raw_date_col": "acceptanceDateTime",
        "market_id": 0,
        "assign_fn": assign_us_effective_date,
    },
    "kr": {
        "meta_file": "events_kr.parquet",
        "sentiment_file": "events_sentiment_kr.parquet",
        "join_key": "rcept_no",
        "raw_date_col": "rcept_dt",
        "market_id": 1,
        "assign_fn": staticmethod(assign_kr_effective_date),
    },
}


def _trading_days(ohlcv_meta: pd.DataFrame, market_id: int) -> pd.DatetimeIndex:
    """실제 관측된 거래일(pd.bdate_range 같은 간이 달력이 아니라 ohlcv_meta의
    실제 date)을 정렬된 DatetimeIndex로 반환한다. US/KR 공휴일이 다르므로
    market_id별로 따로 구성해야 한다."""
    days = ohlcv_meta.loc[ohlcv_meta["market_id"] == market_id, "date"].unique()
    return pd.DatetimeIndex(sorted(days))


def _load_events_with_scores(data_root: Path, market: str) -> pd.DataFrame:
    """이벤트 메타 + 감성 점수를 (ticker, join_key) 복합키로 결합한다.

    accessionNumber/rcept_no 단독으로는 유니크하지 않다(GOOG/GOOGL, 삼성전자
    보통주/우선주처럼 같은 filing을 공유하는 종목 쌍이 있음) — 반드시
    (ticker, join_key) 복합키로 조인해야 한다.
    """
    cfg = _MARKET_CONFIG[market]
    processed = data_root / "processed"
    meta = pd.read_parquet(processed / cfg["meta_file"])
    scores = pd.read_parquet(processed / cfg["sentiment_file"])

    key = ["ticker", cfg["join_key"]]
    merged = meta.merge(
        scores[key + ["score"]], on=key, how="left", validate="one_to_one", indicator=True
    )
    unmatched = int((merged["_merge"] == "left_only").sum())
    if unmatched:
        raise RuntimeError(
            f"[event_features] {market}: 감성 점수와 매칭되지 않은 이벤트 {unmatched}건 발견 — "
            "phase2_sentiment_scoring_report.md 기준 전체 매칭이 정상이므로 데이터 확인 필요"
        )
    return merged.drop(columns="_merge")


def _assign_effective_dates(events: pd.DataFrame, market: str, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """각 이벤트에 leakage-safe effective_date를 배정한다.

    캘린더가 아직 도달하지 못한(수집 시점 근처) 트레일링 이벤트는
    `assign_*_effective_date`가 ValueError를 낸다 — 이런 이벤트는 어차피
    features.parquet의 어떤 행보다도 뒤에 위치해 최종 병합 결과에 영향을 줄 수
    없으므로 드롭하고 개수만 로그로 남긴다.
    """
    cfg = _MARKET_CONFIG[market]
    assign_fn = cfg["assign_fn"]
    raw_col = cfg["raw_date_col"]

    effective_dates: list[pd.Timestamp | None] = []
    dropped = 0
    for value in events[raw_col]:
        try:
            effective_dates.append(assign_fn(value, trading_days))
        except ValueError:
            effective_dates.append(None)
            dropped += 1
    events = events.copy()
    events["effective_date"] = pd.to_datetime(pd.Series(effective_dates, index=events.index))
    if dropped:
        print(
            f"[event_features] {market}: 거래일 캘린더 범위를 벗어난 트레일링 이벤트 "
            f"{dropped}건 드롭(수집 시점 근처, 어떤 feature row에도 영향 없음)"
        )
    return events.dropna(subset=["effective_date"])


def _compute_features_for_ticker(events_ticker: pd.DataFrame, row_dates: np.ndarray) -> pd.DataFrame:
    """단일 종목의 이벤트(effective_date 배정 완료) + 그 종목의 거래일 그리드로
    4개 이벤트 피처를 계산한다.

    row_dates: features.parquet에 실제 존재하는 이 종목의 날짜(오름차순, 중복 없음).
    "5일" 윈도우는 기존 momentum5/ma5_ratio 등과 동일하게 "달력일 5일"이 아니라
    "그 종목의 최근 5개 거래일 row"로 정의한다(코드베이스 컨벤션과 일치).
    """
    events_ticker = events_ticker.sort_values("effective_date")
    eff = events_ticker["effective_date"].to_numpy()
    score = events_ticker["score"].to_numpy(dtype=float)  # KR None-score는 NaN으로 옴
    has_score = ~np.isnan(score)
    score_filled = np.where(has_score, score, 0.0)

    n = len(row_dates)

    if len(eff) == 0:
        # 이 종목에 이벤트 자체가 전혀 없음 — searchsorted/인덱싱 없이 바로 중립값.
        return pd.DataFrame(
            {
                "date": row_dates,
                "event_count_5d": np.zeros(n, dtype=int),
                "event_sentiment_latest": np.zeros(n, dtype=float),
                "event_sentiment_mean5d": np.zeros(n, dtype=float),
                "days_since_last_event": np.full(n, NO_PRIOR_EVENT_SENTINEL, dtype=int),
            }
        )

    cum_score_sum = np.concatenate([[0.0], np.cumsum(score_filled)])
    cum_has_score = np.concatenate([[0], np.cumsum(has_score.astype(int))])

    # "가장 최근 유효 점수"는 score가 있는 이벤트만의 부분수열에서 찾는다.
    score_idx = np.where(has_score)[0]
    score_eff = eff[score_idx]
    score_val = score[score_idx]

    idx = np.arange(n)
    window_start = row_dates[np.maximum(0, idx - 4)]

    lefts = np.searchsorted(eff, window_start, side="left")
    rights = np.searchsorted(eff, row_dates, side="right")  # eff <= row_date (전체 누적)

    event_count_5d = rights - lefts

    window_score_sum = cum_score_sum[rights] - cum_score_sum[lefts]
    window_has_score_cnt = cum_has_score[rights] - cum_has_score[lefts]
    sentiment_mean5d = np.where(
        window_has_score_cnt > 0, window_score_sum / np.maximum(window_has_score_cnt, 1), 0.0
    )

    if len(score_val) == 0:
        sentiment_latest = np.zeros(n, dtype=float)
    else:
        s_rights = np.searchsorted(score_eff, row_dates, side="right")
        sentiment_latest = np.where(s_rights > 0, score_val[np.maximum(s_rights - 1, 0)], 0.0)

    last_event_exists = rights > 0
    last_event_date = eff[np.maximum(rights - 1, 0)]
    days_since = np.where(
        last_event_exists,
        (row_dates - last_event_date) / np.timedelta64(1, "D"),
        NO_PRIOR_EVENT_SENTINEL,
    ).astype(int)

    return pd.DataFrame(
        {
            "date": row_dates,
            "event_count_5d": event_count_5d.astype(int),
            "event_sentiment_latest": sentiment_latest.astype(float),
            "event_sentiment_mean5d": sentiment_mean5d.astype(float),
            "days_since_last_event": days_since,
        }
    )


def build_event_features(data_root: Path | None = None) -> pd.DataFrame:
    """이벤트 피처를 계산해 `features.parquet`에 병합하고 덮어쓴다.

    산출물: ${DATA_ROOT}/processed/features.parquet (기존 13개 기술지표 + target에
    event_count_5d/event_sentiment_latest/event_sentiment_mean5d/days_since_last_event
    4개 컬럼 추가, 총 17개 피처 + target).
    """
    data_root = data_root or get_data_root()
    processed = data_root / "processed"

    ohlcv_meta = pd.read_parquet(processed / "ohlcv_meta.parquet")
    features = pd.read_parquet(processed / "features.parquet")

    all_events = []
    for market in ("us", "kr"):
        cfg = _MARKET_CONFIG[market]
        trading_days = _trading_days(ohlcv_meta, cfg["market_id"])
        events = _load_events_with_scores(data_root, market)
        events = _assign_effective_dates(events, market, trading_days)
        all_events.append(events[["ticker", "effective_date", "score"]])
    events_all = pd.concat(all_events, ignore_index=True)

    per_ticker_frames = []
    for ticker, group in features.groupby("ticker", sort=False):
        row_dates = np.sort(group["date"].unique())
        events_ticker = events_all.loc[events_all["ticker"] == ticker]
        ticker_features = _compute_features_for_ticker(events_ticker, row_dates)
        ticker_features["ticker"] = ticker
        per_ticker_frames.append(ticker_features)
    event_features = pd.concat(per_ticker_frames, ignore_index=True)

    merged = features.merge(event_features, on=["ticker", "date"], how="left", validate="one_to_one")

    missing = int(merged[EVENT_FEATURE_COLUMNS].isna().any(axis=1).sum())
    if missing:
        raise RuntimeError(
            f"[event_features] 이벤트 피처 병합 후 결측 {missing}행 — 병합 키(ticker,date) 불일치 의심"
        )

    output_path = processed / "features.parquet"
    merged.to_parquet(output_path, index=False)
    print(
        f"[event_features] {merged['ticker'].nunique()}종목, {len(merged)}행, "
        f"이벤트 피처 {len(EVENT_FEATURE_COLUMNS)}개 추가 -> {output_path}"
    )
    return merged


if __name__ == "__main__":
    build_event_features()
