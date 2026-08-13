"""스케일러 fit(Train만)/transform. CLAUDE.md 누수 방지 규칙 1항, plan/03 기준."""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.indicators import FEATURE_COLUMNS


def fit_scaler(train: pd.DataFrame) -> StandardScaler:
    """StandardScaler를 Train 데이터에만 fit한다. Test에는 절대 fit하지 않는다."""
    scaler = StandardScaler()
    scaler.fit(train[FEATURE_COLUMNS])
    return scaler


def transform(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """이미 fit된 scaler로 transform만 수행한다 (fit_transform 사용 금지)."""
    df = df.copy()
    df[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])
    return df


def save_scaler(scaler: StandardScaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: Path) -> StandardScaler:
    with open(path, "rb") as f:
        return pickle.load(f)
