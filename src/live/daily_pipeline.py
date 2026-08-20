"""Phase 4 공유 — 오늘자 관측에 필요한 데이터를 준비한다.

## 배경: features.parquet은 구조적으로 "오늘"을 절대 담지 못한다

`build_features()`(Phase 1)는 `target`(내일 종가로 계산)이 non-null인 행만
남긴다 — `compute_features()`가 `target[future_close.isna()] = pd.NA`로
명시적으로 그렇게 만든다(`src/features/indicators.py`, 미래 참조는 target
계산에만 허용된다는 leakage-guard 원칙 그대로). 문제는, 라이브에서 "오늘"은
정의상 항상 "가장 최근 거래일"이라 "내일 종가"가 아직 존재하지 않는다 —
그래서 `daily_pipeline`을 매일 아무리 성실하게 돌려도, `build_features()`를
그대로 다시 호출하는 한 오늘자 행은 `features.parquet`에 영원히 들어가지
못한다(Phase 4 4단계 구현 중 실제로 발견한 구조적 버그 — 이전 단계에서 만든
`observation.py`는 "features.parquet에 오늘자 행이 있다"고 그냥 전제하고
있었다).

해결 방향을 정하기 전에 `target`을 살려서 `features.parquet`에 오늘자 행을
직접 append하는 방법도 검토했으나, 기존 target 컬럼이 (dropna 후
`astype(int)`라) 순수 int64인데 신규 행의 target을 NA로 넣으면 pandas가
전체 컬럼을 object/float64로 승격시켜(실측 확인) 배치 파일의 dtype을
오염시킨다는 걸 확인해 기각했다. 대신 `features.parquet`은 Phase 1/2가
소유한 배치 산출물로 그대로 두고 절대 쓰지 않는다 — 라이브 쪽에서만
`compute_features()`/이벤트 피처 계산을 재사용해 오늘자 행을 메모리상으로
계산해 배치 윈도우에 이어붙인다(디스크에는 안 남음, target 컬럼 자체가
없음 — 라이브 소비자는 애초에 안 씀).

## 배경: ohlcv_meta_us.parquet -> ohlcv_meta.parquet 병합 함수가 없었다

`collect_us()`/`collect_kr()`는 시장별 파일(`ohlcv_meta_{us,kr}.parquet`)만
쓴다. 이 둘을 합쳐 `build_features()`/`event_features.py`/`panel.py`가
공통으로 읽는 `ohlcv_meta.parquet`을 만드는 작업은 Phase 1 완료 시점에
한 번 수작업으로 했을 뿐 재사용 가능한 함수로 남아있지 않았다 — 매일
자동으로 갱신해야 하는 라이브 파이프라인엔 이게 없으면 안 되므로 여기서
추가한다.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_data_root
from src.features.event_features import (
    _assign_effective_dates,
    _compute_features_for_ticker as _compute_event_features_for_ticker,
    _load_events_with_scores,
    _trading_days,
)
from src.features.indicators import BASE_FEATURE_COLUMNS, EVENT_FEATURE_COLUMNS, compute_features

_MARKET_KEY = {0: "us", 1: "kr"}


def merge_ohlcv_meta(data_root: Path | None = None) -> pd.DataFrame:
    """`ohlcv_meta_us.parquet` + `ohlcv_meta_kr.parquet`을 합쳐 통합
    `ohlcv_meta.parquet`을 다시 쓴다. 한쪽 파일이 아직 없으면(예: KR 트랙이
    한 번도 안 돌았거나 이 저장소를 처음 clone한 경우) 있는 쪽만으로 만든다
    — 두 트랙이 독립적으로 개발된다는 설계(plan/10 §0-5)상 한쪽이 없다고
    막으면 안 된다.

    출력 파일에 직접 `to_parquet()`하지 않고 tmp-then-rename으로 쓴다
    (`safety.py::save_state()`/`progress_log.json`과 동일한 패턴). plan/10
    §B-6이 KR/US 두 트랙의 decide_*.py를 같은 07:00~08:00 KST 창에 cron으로
    돌리도록 설계했는데, 이 함수는 양쪽 트랙이 똑같이 호출하는 **공유** 산출물
    이라 — 직접 쓰기였다면 한쪽이 쓰는 도중(parquet은 파일 끝에 메타데이터
    footer를 쓰는 포맷이라 쓰기 중간 상태는 유효하지 않음) 다른 쪽이 같은
    파일을 읽어(`build_live_features_window()`) 손상된 파일을 만나 그날의
    decide 사이클 전체가 크래시할 수 있었다(KR 트랙과의 간섭 여부를 점검하다
    발견 — 두 트랙이 붙기 전에는 동시 쓰기 자체가 없어 드러나지 않았음)."""
    data_root = data_root or get_data_root()
    processed = data_root / "processed"

    frames = []
    for name in ("ohlcv_meta_us.parquet", "ohlcv_meta_kr.parquet"):
        path = processed / name
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise RuntimeError(f"{processed}에 ohlcv_meta_us.parquet/ohlcv_meta_kr.parquet이 둘 다 없음")

    combined = pd.concat(frames, ignore_index=True)
    output_path = processed / "ohlcv_meta.parquet"
    fd, tmp_path = tempfile.mkstemp(dir=processed, suffix=".tmp")
    os.close(fd)
    try:
        combined.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, output_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return combined


def _compute_live_rows(
    data_root: Path,
    ohlcv_meta: pd.DataFrame,
    target_date: pd.Timestamp,
    tickers: list[str],
    market_id: int,
    include_event_features: bool,
) -> pd.DataFrame:
    base_rows = []
    for ticker in tickers:
        ohlcv_ticker = ohlcv_meta.loc[ohlcv_meta["ticker"] == ticker]
        if ohlcv_ticker.empty:
            continue
        computed = compute_features(ohlcv_ticker)
        row = computed.loc[computed["date"] == target_date]
        if row.empty:
            # 그날 그 종목 OHLCV 자체가 없음(휴장/데이터 공백) — 조용히 스킵.
            # build_grid()가 어차피 0-채움+valid_mask=0으로 마스킹한다(leakage-guard
            # 불변식 그대로 유지, panel.py와 동일한 "row 없으면 마스킹" 원칙).
            continue
        r = row.iloc[0]
        if r[BASE_FEATURE_COLUMNS].isna().any():
            # 초기 롤링 윈도우(MA60 등) 미충족 — 이 티커가 아직 60거래일 이력이
            # 안 쌓임(신규 상장/신규 유니버스 편입 직후). 마찬가지로 스킵 —
            # 부분적으로 NaN인 행을 그대로 내보내면 안 된다.
            continue
        base_rows.append(
            {
                "date": target_date,
                "ticker": ticker,
                "market_id": r["market_id"],
                "size_id": r["size_id"],
                **{c: r[c] for c in BASE_FEATURE_COLUMNS},
            }
        )
    if not base_rows:
        return pd.DataFrame()
    base_df = pd.DataFrame(base_rows)

    if not include_event_features:
        return base_df

    market_key = _MARKET_KEY[market_id]
    trading_days = _trading_days(ohlcv_meta, market_id)
    events = _load_events_with_scores(data_root, market_key)
    events = _assign_effective_dates(events, market_key, trading_days)

    target_date_np = target_date.to_datetime64()
    event_rows = []
    for ticker in base_df["ticker"]:
        # "5거래일" 윈도우는 달력일이 아니라 그 종목의 "최근 5개 거래일 row"로
        # 정의된다(_compute_features_for_ticker의 window_start = row_dates[idx-4]).
        # target_date 하나만 담긴 1행짜리 row_dates를 넘기면 idx-4가 0으로
        # clamp돼 window_start==target_date 자기 자신이 되어 버려, 최근 1~4
        # 거래일 사이에 있었던 이벤트가 event_count_5d/event_sentiment_mean5d에서
        # 통째로 빠지는 버그가 있었다(배치 경로는 그 종목의 전체 날짜 grid를
        # row_dates로 넘겨 정상 동작 — 라이브만 수치가 어긋났음, review-loop 5차
        # 재검토로 발견). 이 종목의 실제 최근 거래일(target_date까지)을 넘겨
        # 배치와 동일한 윈도우 정의를 재현한다.
        ticker_days = np.sort(
            ohlcv_meta.loc[
                (ohlcv_meta["ticker"] == ticker) & (ohlcv_meta["date"] <= target_date), "date"
            ].unique()
        )
        row_dates = ticker_days[-5:] if len(ticker_days) >= 5 else ticker_days
        events_ticker = events.loc[events["ticker"] == ticker]
        ef = _compute_event_features_for_ticker(events_ticker, row_dates)
        ef = ef.loc[ef["date"] == target_date_np]
        ef["ticker"] = ticker
        event_rows.append(ef)
    event_df = pd.concat(event_rows, ignore_index=True)

    merged = base_df.merge(event_df, on=["ticker", "date"], how="left", validate="one_to_one")
    if merged[EVENT_FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("[daily_pipeline] 라이브 이벤트 피처 병합 후 결측 발견 — 병합 키(ticker,date) 불일치 의심")
    return merged


def build_live_features_window(
    data_root: Path,
    target_date: pd.Timestamp,
    lookback_days: int,
    market_id: int,
    include_event_features: bool,
) -> pd.DataFrame:
    """`features.parquet`의 [target_date - lookback_days, target_date] 구간을
    읽되, `target_date` 행이 빠진(구조적으로 항상 빠짐 — 모듈 docstring 참고)
    `market_id` 시장 티커에 한해 오늘자 기술지표(+옵션으로 이벤트 피처)를
    즉석 계산해 붙인 결과를 반환한다. `features.parquet` 자체는 절대 쓰지
    않는다(디스크에 남기지 않음 — Phase 1/2 배치 산출물 순수성 유지). `target`
    컬럼은 반환값에서 제외한다(라이브 소비자는 쓰지 않으므로).

    호출 전에 그날 `ohlcv_meta.parquet`/`events_us.parquet`/
    `events_sentiment_us.parquet`이 이미 최신이어야 한다 — 수집 자체는 이
    함수의 책임이 아니다(decide_us.py가 먼저 실행).
    """
    data_root = data_root or get_data_root()
    processed = data_root / "processed"
    target_date = pd.Timestamp(target_date).normalize()
    window_start = target_date - pd.Timedelta(days=lookback_days)

    features = pd.read_parquet(processed / "features.parquet")
    window = features[(features["date"] >= window_start) & (features["date"] <= target_date)].copy()
    if "target" in window.columns:
        window = window.drop(columns="target")

    ohlcv_meta = pd.read_parquet(processed / "ohlcv_meta.parquet")
    market_tickers = sorted(ohlcv_meta.loc[ohlcv_meta["market_id"] == market_id, "ticker"].unique())
    present_today = set(window.loc[window["date"] == target_date, "ticker"])
    missing_tickers = [t for t in market_tickers if t not in present_today]
    if not missing_tickers:
        return window

    live_rows = _compute_live_rows(data_root, ohlcv_meta, target_date, missing_tickers, market_id, include_event_features)
    if live_rows.empty:
        return window
    return pd.concat([window, live_rows], ignore_index=True)
