# Phase 2 모듈 4 — 이벤트 피처 집계 + features.parquet 병합 완료 리포트

생성 시각: 2026-08-17

## 개요

`events_{us,kr}.parquet`(발행 메타, 모듈 1)와 `events_sentiment_{us,kr}.parquet`(감성 점수, 모듈 3)를 결합해 종목·거래일 단위 이벤트 피처 4종을 계산하고, Phase 1 `features.parquet`(13개 기술지표 + target)에 병합했다.

- 구현: `src/features/event_features.py` (`build_event_features()`)
- 컬럼 정의: `src/features/indicators.py`(`BASE_FEATURE_COLUMNS` 13개 + `EVENT_FEATURE_COLUMNS` 4개 = `FEATURE_COLUMNS` 17개)
- 테스트: `tests/test_event_features.py` (6건, look-ahead 불변식 포함) — 기존 122건 + 신규 6건 = 128건 전체 통과

## 조인 키

US는 `(ticker, accessionNumber)`, KR은 `(ticker, rcept_no)` 복합키로 조인했다. `accessionNumber`/`rcept_no` 단독으로는 유니크하지 않다 — 같은 filing을 공유하는 종목 쌍(GOOG/GOOGL, 삼성전자 보통주/우선주 등)이 실제 데이터에 존재함을 사전에 확인하고 복합키를 채택했다.

## 타임스탬프 정렬

`src/data_collection/event_timing.py`의 `assign_us_effective_date`/`assign_kr_effective_date`를 그대로 재사용했다(신규 로직 없음). `trading_days` 캘린더는 `pd.bdate_range` 같은 간이 달력이 아니라 `ohlcv_meta.parquet`의 실제 관측 거래일(market_id별)로 구성했다.

캘린더가 아직 도달하지 못한 트레일링 이벤트(수집 시점 근처)는 US 5건, KR 26건 드롭됐다 — 이 이벤트들의 effective_date는 `features.parquet`의 마지막 행보다도 뒤에 위치해 어떤 행에도 영향을 줄 수 없으므로 안전하게 제외했다.

## 피처 정의 및 leakage 방지

종목 t의 어떤 피처든 "그 종목의 t번째(오름차순) 거래일 row_date" 기준으로 `effective_date <= row_date`인 이벤트만 사용한다 — `compute_features()`의 `rolling(...)`이 현재 행을 포함한 과거만 쓰는 것과 동일한 원칙이다.

- `event_count_5d` / `event_sentiment_mean5d`: "해당 종목의 최근 5개 거래일 row" 윈도우(기존 `momentum5`/`ma5_ratio`와 동일한 "5" 정의) 내 이벤트 집계. `event_sentiment_mean5d`는 0-채움된 일별 그리드에 rolling mean을 적용하지 않고 **윈도우 내 실제 이벤트만의 평균**으로 계산한다 — 그렇지 않으면 이벤트 희소 구간에서 평균이 0쪽으로 희석된다.
- `event_sentiment_latest`: 가장 최근 유효 점수를 가진 이벤트의 score.
- `days_since_last_event`: 마지막 이벤트로부터의 캘린더일수. "오늘 이벤트 발생"(0)과 "첫 이벤트 이전(이력 없음)"이 둘 다 0이 되는 충돌을 피하기 위해, 후자는 CLAUDE.md의 "이벤트 없는 날=0" 규칙의 명시적 예외로 sentinel(-1)을 사용한다(이번 데이터셋에서는 전 종목이 첫 feature row 이전에 이미 이벤트 이력이 있어 실제로 발현되지 않았지만, 메커니즘은 유닛테스트로 별도 검증했다).
- `score=None`(KR 원문 없음, DART status 014, 965건 부류)은 `event_count_5d`/`days_since_last_event`에는 포함하되(공시 자체는 실재하므로) 감성 집계(`latest`/`mean5d`)에서는 제외한다.

## 검증

- **파이프라인 dry-run**: 실제 이벤트 메타데이터에 synthetic 점수를 붙여 스크래치 디렉터리에서 선행 검증(120종목, 252,362행, 결측 0건) — 실제 Colab 실행 결과와 US 5건/KR 26건 트레일링 드롭, 252,362행이 정확히 일치해 파이프라인 로직이 실 데이터에서도 동일하게 동작함을 교차 확인했다.
- **실제 실행**: Google Drive가 마운트된 Colab에서 실행(GPU 불필요, 순수 pandas/numpy 연산). `events_sentiment_{us,kr}.parquet`은 모듈 3 때부터 Drive에 있었고, Phase 1 산출물(`ohlcv_meta.parquet`, `features.parquet`)만 로컬에서 Drive로 추가 업로드했다.
- **결측/타입**: 이벤트 피처 4개 컬럼 결측 0건, dtype 정상(`event_count_5d`/`days_since_last_event`는 int64, `event_sentiment_latest`/`event_sentiment_mean5d`는 float64), 120종목 전부 존재, `target` 결측 0건.
- **테스트**: `pytest tests/ -v` 128개 전체 통과.
- **leakage-guard 체크리스트**: 사전(설계 리뷰) + 사후(`rolling(center=True)`/`shift(-N)`/`.fit(` grep) 모두 통과.

## 최종 통계

| 피처 | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|
| event_count_5d | 0.522 | 1.162 | 0 | 0 | 0 | 1 | 127 |
| event_sentiment_latest | 0.073 | 0.192 | -1.000 | -0.0004 | 0.0006 | 0.140 | 0.997 |
| event_sentiment_mean5d | 0.016 | 0.106 | -1.000 | 0 | 0 | 0 | 0.997 |
| days_since_last_event | 35.1 | 139.6 | 0 | 5 | 14 | 34 | 2740 |

`event_sentiment_latest`/`mean5d`의 평균(0.073/0.016)은 모듈 3 리포트의 원시 점수 평균(US 0.095/KR 0.034)과 같은 방향으로, 자연스러운 범위다. `event_count_5d`/`days_since_last_event`는 점수와 무관한 값이라 사전 synthetic dry-run과 정확히 동일한 분포가 나왔다(교차검증).

시장별로는 KR이 US보다 `event_count_5d>0` 비율이 높다(KR 41.2% vs US 20.6%) — KR 이벤트 건수(22,490)가 US(7,944)보다 약 2.8배 많다는 모듈 1 결과와 일치한다. `event_sentiment_latest` 평균도 US(0.107)가 KR(0.043)보다 높은데, 이는 모듈 3에서 확인된 8-K의 긍정 편향과 일치한다.

## 다음 단계

Phase 2 Definition of Done 중 "이벤트 피처 집계, features.parquet 병합"은 완료. 남은 항목:
- 모듈 5: 이벤트 피처 포함 재학습을 walk-forward CV로 진행, Phase 1 베이스라인과 비교, 리포트 작성 — **사용자 확인 후 착수**
