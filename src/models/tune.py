"""하이퍼파라미터 탐색 (walk-forward 교차검증 기준).

단일 80/20 분할로 튜닝하면 그 특정 시기에 과최적화될 위험이 있다 — walk-forward
5폴드 평균 성능을 기준으로 후보를 비교한다. CLAUDE.md 원칙("MVP에서는
과최적화하지 않는다")에 맞춰 넓은 그리드서치가 아니라 적당한 크기의 랜덤서치로
제한한다.
"""
from __future__ import annotations

import random

import pandas as pd

from src.models.train import BASELINE_PARAMS
from src.models.walk_forward import run_walk_forward_cv

PARAM_DISTRIBUTIONS = {
    "max_depth": [3, 4, 5, 6],
    "n_estimators": [200, 300, 400, 500],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0],
}

FIXED_PARAMS = dict(eval_metric="logloss", enable_categorical=True, random_state=42)


def sample_candidates(n_candidates: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    candidates = [dict(BASELINE_PARAMS)]  # 현재 베이스라인도 후보에 포함해서 직접 비교
    for _ in range(n_candidates - 1):
        candidate = {k: rng.choice(v) for k, v in PARAM_DISTRIBUTIONS.items()}
        candidate.update(FIXED_PARAMS)
        candidates.append(candidate)
    return candidates


def random_search(
    features: pd.DataFrame, n_candidates: int = 16, n_folds: int = 5, seed: int = 42
) -> pd.DataFrame:
    """후보마다 walk-forward CV를 돌려 fold 평균 accuracy/roc_auc를 비교한다."""
    candidates = sample_candidates(n_candidates, seed=seed)

    rows = []
    for i, params in enumerate(candidates):
        overall_df, _ = run_walk_forward_cv(features, n_folds=n_folds, params=params)
        rows.append(
            {
                "candidate": i,
                "is_baseline": params == {**BASELINE_PARAMS},
                "mean_accuracy": overall_df["accuracy"].mean(),
                "std_accuracy": overall_df["accuracy"].std(),
                "mean_roc_auc": overall_df["roc_auc"].mean(),
                "min_roc_auc": overall_df["roc_auc"].min(),
                **{f"param_{k}": v for k, v in params.items() if k in PARAM_DISTRIBUTIONS},
            }
        )
    return pd.DataFrame(rows).sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)
