"""Phase 4 공유 — 저장된 정책 체크포인트로 오늘자 관측을 추론한다.

`model.predict()` 자체는 `src/rl/evaluate.py::rollout_policy_with_positioning_stats`와
동일한 호출 패턴(`deterministic=True` — 라이브 운영에서 정책의 확률적 샘플링에
의한 비재현성을 피하기 위함)을 그대로 재사용한다. 체크포인트 로드 전
`evaluate.py::_load_checkpoint_checked`(177-194행)와 동일한 패턴으로
`include_event_features` 일치를 먼저 검증한다 — 다르면 관측 차원이 어긋나
sklearn/SB3 내부에서 알아보기 힘든 shape 에러로 죽으므로, 여기서 먼저 이름
붙은 에러로 막는다.

`include_event_features` 일치만으로는 부족하다 — `rl_ticker_universe.json`이
(예: 이 저장소에서 다른 목적으로 `build_panel()`/`train_agent.py`/`evaluate.py`가
다시 실행돼) 체크포인트 학습 당시와 다른 티커 구성으로 덮어써지면
`include_event_features`는 그대로인데도 관측 차원이 달라진다 — 그래서
`predict_action()`은 실제 `obs.shape`를 `model.observation_space.shape`와
직접 비교해 더 일반적으로 막는다(재검토로 발견).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO

from src.rl.train_agent import load_checkpoint, load_checkpoint_meta


def load_checkpoint_checked(
    checkpoints_dir: Path, name: str, include_event_features: bool
) -> tuple[PPO, StandardScaler]:
    meta = load_checkpoint_meta(checkpoints_dir, name)
    trained_with = meta.get("include_event_features")
    if trained_with != include_event_features:
        raise ValueError(
            f"{name} 체크포인트는 include_event_features={trained_with}로 학습됐는데 "
            f"지금은 include_event_features={include_event_features}로 관측을 만들었음 — "
            "관측 차원이 어긋나 추론을 진행할 수 없다."
        )
    return load_checkpoint(checkpoints_dir, name)


def predict_action(
    checkpoints_dir: Path, name: str, obs: np.ndarray, include_event_features: bool
) -> np.ndarray:
    """obs(단일 관측벡터, shape=(obs_dim,))에 대해 결정적 행동(shape=(120,),
    {0:SELL,1:HOLD,2:BUY})을 반환한다."""
    if not np.isfinite(obs).all():
        # infer.py는 observation.py가 만든 obs만 받는다고 전제하지 않는다(범용
        # "체크포인트로 추론" 모듈로 문서화돼 있음) — observation.py 쪽에서 이미
        # NaN/Inf를 여러 겹 막아뒀지만(재검토 라운드들), 이 함수 자체도 독립적으로
        # 방어해야 다른 경로로 만들어진 obs가 들어와도 SB3 내부에서 조용히 잘못된
        # 예측을 내는 대신 여기서 이름 붙은 에러로 막을 수 있다(재검토로 발견).
        raise ValueError("obs에 유한하지 않은 값(NaN/Inf)이 있음 — 정책 추론을 진행할 수 없다.")

    model, _scaler = load_checkpoint_checked(checkpoints_dir, name, include_event_features)
    expected_shape = model.observation_space.shape
    if obs.shape != expected_shape:
        raise ValueError(
            f"관측벡터 shape={obs.shape}가 {name} 체크포인트의 "
            f"observation_space shape={expected_shape}와 다름 — "
            "include_event_features는 일치하지만 티커 유니버스 구성"
            "(rl_ticker_universe.json)이 학습 당시와 달라졌을 수 있다."
        )
    action, _state = model.predict(obs, deterministic=True)
    return action
