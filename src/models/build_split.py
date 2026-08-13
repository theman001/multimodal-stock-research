"""Phase 1 분할 모듈 진입점.

${DATA_ROOT}/processed/features.parquet를 전역 날짜 기준 80/20 + 60일 embargo로
나누고, Train에만 fit한 스케일러로 둘 다 transform한다.

산출물: ${DATA_ROOT}/processed/{train,test}.parquet, ${DATA_ROOT}/checkpoints/scaler.pkl
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import get_data_root
from src.models.scaling import fit_scaler, save_scaler, transform
from src.models.split import split_train_test


def build_split(data_root: Path | None = None):
    data_root = data_root or get_data_root()
    processed_dir = data_root / "processed"
    features = pd.read_parquet(processed_dir / "features.parquet")

    train, test, train_cutoff, test_start = split_train_test(features)

    scaler = fit_scaler(train)
    train_scaled = transform(train, scaler)
    test_scaled = transform(test, scaler)

    train_scaled.to_parquet(processed_dir / "train.parquet", index=False)
    test_scaled.to_parquet(processed_dir / "test.parquet", index=False)

    checkpoints_dir = data_root / "checkpoints"
    save_scaler(scaler, checkpoints_dir / "scaler.pkl")

    print(
        f"[build_split] train: {len(train_scaled)}행 (~{train_cutoff.date()}까지), "
        f"test: {len(test_scaled)}행 ({test_start.date()}~), "
        f"embargo: {train_cutoff.date()} ~ {test_start.date()}"
    )
    return train_scaled, test_scaled, train_cutoff, test_start


if __name__ == "__main__":
    build_split()
