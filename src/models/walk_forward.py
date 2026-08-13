"""Walk-forward(확장 윈도우) 교차검증.

단일 80/20 분할(src/models/split.py, model_v3)은 딱 한 시기(2024-06~2026-08)에
대해서만 성능을 확인한다 — 그 결과가 진짜 신호인지, 그 특정 구간에서 우연히
잘/못 나온 것인지 구분할 수 없다. 이 모듈은 전체 기간(2015~현재)을 따라가며
학습 구간을 계속 넓혀가되 평가 구간은 서로 겹치지 않게 잘라, 여러 시기에서
반복적으로 같은 결론(방향성 적중률 50~55%대, 조건별 편차)이 재현되는지 본다.

Phase 1 공식 산출물(model_v{N}.json)과는 별개의 진단 도구이며 체크포인트를
새로 만들지 않는다 — 매 폴드 그때그때 학습하고 버린다.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.models.scaling import fit_scaler, transform
from src.models.split import EMBARGO_DAYS
from src.models.train import evaluate, evaluate_by_cell, train_model


@dataclass
class Fold:
    fold: int
    train_cutoff: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_walk_forward_folds(
    features: pd.DataFrame, n_folds: int = 5, embargo_days: int = EMBARGO_DAYS
) -> list[Fold]:
    """전체 날짜를 (n_folds+1)개의 균등한 블록으로 나눈다.

    fold k의 train은 [처음, 블록 k 끝]까지(확장), test는 [블록 k 끝 + embargo,
    블록 k+1 끝]까지 — 폴드 간 test 구간은 서로 겹치지 않는다.
    """
    unique_dates = pd.Series(sorted(features["date"].unique()))
    n_dates = len(unique_dates)
    n_blocks = n_folds + 1
    cut_indices = [int(n_dates * (i + 1) / n_blocks) - 1 for i in range(n_blocks)]

    folds = []
    for k in range(n_folds):
        train_cutoff = unique_dates.iloc[cut_indices[k]]
        test_start = train_cutoff + pd.Timedelta(days=embargo_days)
        test_end = unique_dates.iloc[cut_indices[k + 1]]
        folds.append(Fold(fold=k + 1, train_cutoff=train_cutoff, test_start=test_start, test_end=test_end))
    return folds


def run_walk_forward_cv(
    features: pd.DataFrame, n_folds: int = 5, params: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """각 폴드마다 (해당 폴드의 train에만 fit한 스케일러로) 학습·평가한다.

    params를 넘기면 그 하이퍼파라미터로 학습한다 (하이퍼파라미터 탐색에 재사용하기 위함).
    반환: (fold별 전체 지표 df, fold x 6셀 지표 df)
    """
    folds = generate_walk_forward_folds(features, n_folds=n_folds)

    overall_rows = []
    cell_rows = []
    for f in folds:
        train = features[features["date"] <= f.train_cutoff]
        test = features[(features["date"] >= f.test_start) & (features["date"] <= f.test_end)]
        if train.empty or test.empty:
            continue

        scaler = fit_scaler(train)
        train_scaled = transform(train, scaler)
        test_scaled = transform(test, scaler)

        model = train_model(train_scaled, params=params)

        overall = evaluate(model, test_scaled)
        overall_rows.append({"fold": f.fold, "train_end": f.train_cutoff, "test_start": f.test_start, "test_end": f.test_end, **overall})

        cells = evaluate_by_cell(model, test_scaled)
        cells.insert(0, "fold", f.fold)
        cell_rows.append(cells)

    overall_df = pd.DataFrame(overall_rows)
    cell_df = pd.concat(cell_rows, ignore_index=True) if cell_rows else pd.DataFrame()
    return overall_df, cell_df
