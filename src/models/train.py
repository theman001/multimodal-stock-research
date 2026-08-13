"""Phase 1 모델 학습 모듈. CLAUDE.md / plan/04_model_training.md 기준.

XGBoost 이진분류 베이스라인. market_id/size_id는 원-핫이 아니라 카테고리
피처로 직접 전달한다(enable_categorical=True) — "조건부 학습" 사상 유지.
평가는 전체 지표 + Market_Id × Size_Id 6개 셀 breakdown이 필수다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from src.config import get_data_root
from src.features.indicators import FEATURE_COLUMNS

META_COLUMNS = ["market_id", "size_id"]
MARKET_ID_CATEGORIES = [0, 1]  # CLAUDE.md: 0=해외, 1=국내
SIZE_ID_CATEGORIES = [0, 1, 2]  # CLAUDE.md: 0=소형, 1=중형, 2=대형

# CLAUDE.md 베이스라인 하이퍼파라미터 (튜닝 전 시작점, MVP에서는 과최적화하지 않음)
BASELINE_PARAMS = dict(
    max_depth=4,
    n_estimators=300,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    enable_categorical=True,
    random_state=42,  # subsample/colsample_bytree가 확률적이라 고정하지 않으면 재실행마다 결과가 달라짐
)

# CLAUDE.md 모델/평가 원칙: 50~55% 근방이 현실적 상한 — 크게 벗어나면 과최적화/누수 의심
REALISTIC_ACCURACY_UPPER_BOUND = 0.60


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """market_id/size_id를 고정된 카테고리 집합으로 인코딩한다.

    train/test 각각에서 독립적으로 `.astype("category")`하면, 만약 한쪽에 특정
    값(예: size_id=0)이 우연히 하나도 없을 경우 XGBoost가 내부적으로 쓰는 카테고리
    코드가 train과 test에서 서로 어긋나 조용히 다른 값으로 오인식될 수 있다.
    카테고리 집합을 명시적으로 고정해 이 문제를 원천 차단한다.
    """
    X = df[FEATURE_COLUMNS + META_COLUMNS].copy()
    X["market_id"] = pd.Categorical(X["market_id"], categories=MARKET_ID_CATEGORIES)
    X["size_id"] = pd.Categorical(X["size_id"], categories=SIZE_ID_CATEGORIES)
    return X


def train_model(train: pd.DataFrame, params: dict | None = None) -> XGBClassifier:
    X_train = _prepare_X(train)
    y_train = train["target"]
    model = XGBClassifier(**(params or BASELINE_PARAMS))
    model.fit(X_train, y_train)
    return model


def evaluate(model: XGBClassifier, df: pd.DataFrame) -> dict:
    X = _prepare_X(df)
    y = df["target"]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if y.nunique() > 1 else float("nan"),
    }


def evaluate_by_cell(model: XGBClassifier, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (market_id, size_id), group in df.groupby(["market_id", "size_id"]):
        rows.append({"market_id": market_id, "size_id": size_id, **evaluate(model, group)})
    return pd.DataFrame(rows).sort_values(["market_id", "size_id"]).reset_index(drop=True)


def next_version(checkpoints_dir: Path) -> int:
    versions = []
    for p in checkpoints_dir.glob("model_v*.json"):
        try:
            versions.append(int(p.stem.replace("model_v", "")))
        except ValueError:
            continue
    return max(versions, default=0) + 1


def run(data_root: Path | None = None):
    data_root = data_root or get_data_root()
    processed_dir = data_root / "processed"
    checkpoints_dir = data_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(processed_dir / "train.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")

    model = train_model(train)

    overall_metrics = evaluate(model, test)
    cell_metrics = evaluate_by_cell(model, test)

    suspicious_cells = cell_metrics[cell_metrics["accuracy"] > REALISTIC_ACCURACY_UPPER_BOUND]
    if not suspicious_cells.empty:
        print(
            f"[train] 경고: {REALISTIC_ACCURACY_UPPER_BOUND:.0%}를 초과하는 셀 발견 "
            "— 과최적화/누수 의심, leakage-guard 재점검 필요:"
        )
        print(suspicious_cells.to_string(index=False))

    version = next_version(checkpoints_dir)
    model_path = checkpoints_dir / f"model_v{version}.json"
    meta_path = checkpoints_dir / f"model_v{version}.meta.json"

    model.save_model(str(model_path))

    meta = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": BASELINE_PARAMS,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_columns": FEATURE_COLUMNS,
        "overall_metrics": overall_metrics,
        "cell_metrics": cell_metrics.to_dict(orient="records"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[train] 모델 v{version} 저장 -> {model_path}")
    print(f"[train] 전체 지표: {overall_metrics}")
    print("[train] Market_Id x Size_Id 6셀 breakdown:")
    print(cell_metrics.to_string(index=False))

    return model, overall_metrics, cell_metrics, version


if __name__ == "__main__":
    run()
