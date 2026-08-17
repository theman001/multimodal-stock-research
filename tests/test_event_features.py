import numpy as np
import pandas as pd
import pytest

from src.features.event_features import NO_PRIOR_EVENT_SENTINEL, _compute_features_for_ticker


def _events(rows: list[tuple[str, float | None]]) -> pd.DataFrame:
    """rows: [(effective_date str, score or None), ...]"""
    return pd.DataFrame(
        {
            "effective_date": pd.to_datetime([r[0] for r in rows]),
            "score": [r[1] for r in rows],
        }
    )


def test_no_events_gives_neutral_zero_and_sentinel_days_since():
    row_dates = pd.date_range("2024-01-01", periods=10, freq="B").to_numpy()
    out = _compute_features_for_ticker(_events([]), row_dates)
    assert (out["event_count_5d"] == 0).all()
    assert (out["event_sentiment_latest"] == 0.0).all()
    assert (out["event_sentiment_mean5d"] == 0.0).all()
    assert (out["days_since_last_event"] == NO_PRIOR_EVENT_SENTINEL).all()


def test_event_on_row_date_is_included_same_day_not_leaked_from_future():
    """t일에 effective_date==t인 이벤트는 t행에 반영돼야 하고(당일 마감 전 반영된
    정보이므로 rolling(...) 컨벤션과 동일), t-1행에는 아직 보이면 안 된다."""
    row_dates = pd.date_range("2024-01-01", periods=5, freq="B").to_numpy()
    events = _events([(str(pd.Timestamp(row_dates[2])), 0.5)])
    out = _compute_features_for_ticker(events, row_dates)

    assert out.loc[1, "event_count_5d"] == 0  # t-1: 아직 안 보임
    assert out.loc[1, "days_since_last_event"] == NO_PRIOR_EVENT_SENTINEL
    assert out.loc[2, "event_count_5d"] == 1  # t: 당일 반영
    assert out.loc[2, "event_sentiment_latest"] == 0.5
    assert out.loc[2, "days_since_last_event"] == 0


def test_event_after_last_row_date_never_appears_in_any_row():
    """effective_date가 이 종목의 마지막 row_date보다도 미래면 어떤 row에도
    나타나면 안 된다 — look-ahead 방지의 핵심 불변식."""
    row_dates = pd.date_range("2024-01-01", periods=5, freq="B").to_numpy()
    future_date = pd.Timestamp(row_dates[-1]) + pd.Timedelta(days=10)
    events = _events([(str(future_date), 0.9)])
    out = _compute_features_for_ticker(events, row_dates)
    assert (out["event_count_5d"] == 0).all()
    assert (out["event_sentiment_latest"] == 0.0).all()
    assert (out["days_since_last_event"] == NO_PRIOR_EVENT_SENTINEL).all()


def test_event_count_5d_window_is_5_most_recent_rows_inclusive():
    row_dates = pd.date_range("2024-01-01", periods=8, freq="B").to_numpy()
    # 인덱스 0,1,2에 이벤트 발생 (총 3건), 이후 이벤트 없음
    events = _events(
        [
            (str(pd.Timestamp(row_dates[0])), 0.1),
            (str(pd.Timestamp(row_dates[1])), 0.2),
            (str(pd.Timestamp(row_dates[2])), 0.3),
        ]
    )
    out = _compute_features_for_ticker(events, row_dates)
    # row index 4의 5일 윈도우 = row_dates[0..4] -> 이벤트 3건 모두 포함
    assert out.loc[4, "event_count_5d"] == 3
    assert out.loc[4, "event_sentiment_mean5d"] == pytest.approx(np.mean([0.1, 0.2, 0.3]))
    # row index 5의 5일 윈도우 = row_dates[1..5] -> row_dates[0]의 이벤트는 윈도우 밖
    assert out.loc[5, "event_count_5d"] == 2
    assert out.loc[5, "event_sentiment_mean5d"] == pytest.approx(np.mean([0.2, 0.3]))
    # days_since_last_event는 윈도우와 무관하게 마지막 이벤트 기준(row_dates[2])으로 계속 누적
    assert out.loc[5, "days_since_last_event"] == int(
        (pd.Timestamp(row_dates[5]) - pd.Timestamp(row_dates[2])) / pd.Timedelta(days=1)
    )


def test_none_score_event_counts_but_excluded_from_sentiment_aggregates():
    """score=None(KR 원문 없음 케이스)은 event_count_5d/days_since_last_event에는
    포함되지만 감성 집계(latest/mean5d)에서는 제외돼야 한다."""
    row_dates = pd.date_range("2024-01-01", periods=5, freq="B").to_numpy()
    events = _events(
        [
            (str(pd.Timestamp(row_dates[1])), None),
            (str(pd.Timestamp(row_dates[2])), 0.7),
        ]
    )
    out = _compute_features_for_ticker(events, row_dates)
    assert out.loc[2, "event_count_5d"] == 2  # 둘 다 카운트
    assert out.loc[2, "event_sentiment_mean5d"] == 0.7  # None 이벤트는 평균에서 제외
    assert out.loc[2, "event_sentiment_latest"] == 0.7
    # row 1 시점엔 score 있는 이벤트가 아직 없으므로 latest는 중립값 0
    assert out.loc[1, "event_sentiment_latest"] == 0.0
    assert out.loc[1, "event_count_5d"] == 1


def test_feature_at_t_matches_when_recomputed_on_events_truncated_to_t():
    """t 시점의 이벤트 피처는 t 이후 이벤트를 몰라도 동일해야 한다(룩어헤드 없음) —
    t 이후 이벤트를 추가해도 t 시점 값이 바뀌면 안 된다."""
    row_dates = pd.date_range("2024-01-01", periods=10, freq="B").to_numpy()
    events_before_t = _events(
        [
            (str(pd.Timestamp(row_dates[1])), 0.2),
            (str(pd.Timestamp(row_dates[4])), -0.3),
        ]
    )
    events_with_future = pd.concat(
        [events_before_t, _events([(str(pd.Timestamp(row_dates[8])), 0.99)])],
        ignore_index=True,
    )

    out_before = _compute_features_for_ticker(events_before_t, row_dates)
    out_with_future = _compute_features_for_ticker(events_with_future, row_dates)

    t_index = 5  # 마지막 이벤트(row_dates[4]) 이후, 미래 이벤트(row_dates[8]) 이전
    for col in ["event_count_5d", "event_sentiment_latest", "event_sentiment_mean5d", "days_since_last_event"]:
        assert out_before.loc[t_index, col] == out_with_future.loc[t_index, col], (
            f"{col} 값이 미래 이벤트 유무에 따라 달라짐 — 룩어헤드 의심"
        )
