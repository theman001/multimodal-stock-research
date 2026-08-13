"""Train/Test 분할 (전역 날짜 기준, 60일 embargo). plan/03_split_and_leakage_tests.md 기준.

종목별이 아니라 전역 날짜 기준으로 자른다 — 같은 날짜에 어떤 종목은 Train, 어떤
종목은 Test가 되는 것을 방지한다. Train 마지막 날짜와 Test 시작 날짜 사이에는
최장 롤링 윈도우(MA60)를 감안한 60 캘린더데이 embargo를 둔다.
"""
from __future__ import annotations

import pandas as pd

EMBARGO_DAYS = 60
TRAIN_RATIO = 0.8


def split_train_test(
    features: pd.DataFrame, train_ratio: float = TRAIN_RATIO, embargo_days: int = EMBARGO_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """전역 날짜 기준 시간순 80/20 분할 + embargo.

    반환: (train, test, train_cutoff, test_start)
    train_cutoff와 test_start 사이(양 끝 제외한 날짜)의 행은 어느 쪽에도 포함되지
    않고 버려진다 — embargo 구간.
    """
    unique_dates = pd.Series(sorted(features["date"].unique()))
    cutoff_idx = int(len(unique_dates) * train_ratio) - 1
    train_cutoff = unique_dates.iloc[cutoff_idx]
    test_start = train_cutoff + pd.Timedelta(days=embargo_days)

    train = features[features["date"] <= train_cutoff].copy()
    test = features[features["date"] >= test_start].copy()

    return train, test, train_cutoff, test_start
