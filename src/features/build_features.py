"""Phase 1 피처 엔지니어링 진입점.

${DATA_ROOT}/processed/ohlcv_meta.parquet(120종목)를 읽어 종목별로 독립적으로
기술지표 + target을 계산하고, market_id/size_id 메타피처는 그대로 통과시킨다.
초기 롤링 윈도우 미충족 구간과 마지막 거래일(target 계산 불가)은 명시적으로
drop하고 비율을 로그로 남긴다.

산출물: ${DATA_ROOT}/processed/features.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_data_root
from src.features.indicators import BASE_FEATURE_COLUMNS as FEATURE_COLUMNS
from src.features.indicators import compute_features

OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "market_id",
    "size_id",
    *FEATURE_COLUMNS,
    "target",
]


def build_features(data_root: Path | None = None) -> pd.DataFrame:
    data_root = data_root or get_data_root()
    processed_dir = data_root / "processed"
    ohlcv_meta = pd.read_parquet(processed_dir / "ohlcv_meta.parquet")

    per_ticker = [
        compute_features(group) for _, group in ohlcv_meta.groupby("ticker", sort=False)
    ]
    features = pd.concat(per_ticker, ignore_index=True)

    # vol_chg5 등 비율 피처는 분모(예: 5일 전 거래량)가 0인 날(거래정지 등)에
    # inf가 나올 수 있다 — 조작 없이 결측과 동일하게 명시적으로 drop 대상에 포함시킨다.
    inf_counts = {
        col: int(np.isinf(features[col]).sum())
        for col in FEATURE_COLUMNS
        if np.isinf(features[col]).any()
    }
    if inf_counts:
        print(f"[build_features] inf 값 발견 (분모 0 등, NaN 처리 후 drop 대상): {inf_counts}")
        features[FEATURE_COLUMNS] = features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    total_rows = len(features)
    required_columns = FEATURE_COLUMNS + ["target"]
    valid_mask = features[required_columns].notna().all(axis=1)
    dropped_rows = total_rows - int(valid_mask.sum())
    drop_ratio = dropped_rows / total_rows if total_rows else 0.0
    print(
        f"[build_features] 결측 행 drop: {dropped_rows}/{total_rows} ({drop_ratio:.2%}) "
        "— 초기 롤링 윈도우(최대 MA60) 미충족 구간 + 종목별 마지막 거래일(target 없음) + inf"
    )

    features = features.loc[valid_mask, OUTPUT_COLUMNS].reset_index(drop=True)
    features["target"] = features["target"].astype(int)

    output_path = processed_dir / "features.parquet"
    features.to_parquet(output_path, index=False)
    print(f"[build_features] {features['ticker'].nunique()}종목, {len(features)}행 저장 -> {output_path}")
    return features


if __name__ == "__main__":
    build_features()
