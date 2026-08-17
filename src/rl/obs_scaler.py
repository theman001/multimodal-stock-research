"""Phase 3 RL 관측 정규화.

CLAUDE.md "Phase 3 구체 사양"의 RL 특화 누수 주의사항: SB3의 `VecNormalize`는
기본적으로 온라인으로 통계를 갱신하는데, 평가 중에도 켜두면 그 시점 이전 결정에
그 시점 "이후" 타임스텝의 통계가 섞여 들어간다. `src/models/scaling.py`의
"Train에만 fit, Test는 transform만" 원칙을 여기서도 그대로 지켜, fit된 스케일러를
학습·평가 양쪽에 동일하게 고정 적용한다(`VecNormalize(training=True)`를 평가에
들고 가지 않는다).

`panel.py`가 만드는 정적 피처 블록 중 BASE_FEATURE_COLUMNS(13)+
EVENT_FEATURE_COLUMNS(0/4)만 표준화 대상이다 — `model_v3_prob`/`market_id`/
`size_id`/`valid_mask`는 이미 유계(bounded)이거나 범주형이라 스케일링 대상에서
제외한다(CLAUDE.md 명시).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.rl.panel import Panel

_UNSCALED_CHANNELS = {"model_v3_prob", "market_id", "size_id", "valid_mask"}


def _scaled_channel_indices(panel: Panel) -> list[int]:
    return [i for i, name in enumerate(panel.feature_names) if name not in _UNSCALED_CHANNELS]


def fit_obs_scaler(panel: Panel, train_end_date) -> StandardScaler:
    """panel의 train 구간(date <= train_end_date)에서, 유효한(valid_mask=1) 항목만으로 fit한다.

    마스킹된 항목은 0으로 채워진 합성값이라 스케일러 fit 통계에 포함시키면 평균/
    분산이 왜곡된다(leakage는 아니지만 데이터 품질 문제 — 실제로 관측되지 않은
    값을 진짜 신호처럼 취급하게 됨).
    """
    cols = _scaled_channel_indices(panel)
    cutoff = pd.Timestamp(train_end_date).to_datetime64()
    train_time_mask = panel.dates <= cutoff

    sub = panel.features[train_time_mask][..., cols]
    valid = panel.valid_mask[train_time_mask].astype(bool)
    rows = sub[valid]

    scaler = StandardScaler()
    scaler.fit(rows)
    return scaler


def transform_obs_features(panel: Panel, scaler: StandardScaler) -> np.ndarray:
    """panel.features 전체(train+eval)에 고정된 scaler로 transform만 수행한다(fit 금지).

    마스킹된 항목은 transform 후에도 0으로 되돌린다 — 그대로 스케일러를
    통과시키면 합성 0이 임의의 0 아닌 값이 되어(마치 "평균보다 낮은 실제
    관측값"처럼 보임), valid_mask라는 명시적 신호와 모순되는 잡음이 추가된다.
    """
    cols = _scaled_channel_indices(panel)
    out = panel.features.copy()
    T, N, _ = out.shape

    flat = out[..., cols].reshape(-1, len(cols))
    flat = scaler.transform(flat)
    out[..., cols] = flat.reshape(T, N, len(cols))

    invalid = ~panel.valid_mask.astype(bool)
    out[invalid] = 0.0
    return out


def save_obs_scaler(scaler: StandardScaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_obs_scaler(path: Path) -> StandardScaler:
    with open(path, "rb") as f:
        return pickle.load(f)
