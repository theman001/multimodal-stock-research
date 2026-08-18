"""Phase 3 — PPO 학습 파이프라인.

CLAUDE.md "Phase 3 구체 사양" > "에피소드/평가" 기준. 폴드마다 처음부터 새로
학습한다(폴드 간 이어학습 없음 — `src.models.walk_forward.run_walk_forward_cv`의
"매 폴드 학습하고 버림" 원칙과 동일). walk-forward 폴드 경계는
`src.models.walk_forward.generate_walk_forward_folds()`를 그대로 재사용하고,
공식 헤드라인 비교용 단일분할은 `src.models.split.split_train_test()`가 쓰는
것과 동일한 cutoff/test_start를 재사용한다(train.parquet/test.parquet과 같은
구간, `backtest_report_v3.md`와 직접 비교 가능하게).

스케일러는 각 폴드/공식분할의 train 구간에만 fit한다(`src.rl.obs_scaler`의
train-only 원칙 그대로 — 폴드 간 재사용하지 않고 매번 새로 fit).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from src.config import get_data_root
from src.models.split import split_train_test
from src.models.walk_forward import generate_walk_forward_folds
from src.rl.obs_scaler import fit_obs_scaler, load_obs_scaler, save_obs_scaler
from src.rl.panel import Panel, build_panel
from src.rl.trading_env import EPISODE_LENGTH_DAYS, TradingEnv

# 관측 차원이 커서(120종목 x D_static, ~2,900) SB3 기본 정책망(2x64)보다 키운다.
POLICY_KWARGS = dict(net_arch=[256, 256])

# 첫 실측 파일럿(learning_rate 기본값 3e-4)에서 approx_kl이 이터레이션마다 계속
# 불어나는(9->93->183) 폭주를 실제로 관측했다. trading_env.py의 REWARD_CLIP만
# 추가했을 때도 여전히 불안정(KL이 종종 50대까지 튐)했지만, learning_rate를
# 3e-5로 10배 낮추자 완전히 안정화됨을 재파일럿으로 확인했다(10개 이터레이션
# 내내 approx_kl 0.067~0.090 유지, explained_variance가 -1.82->0.78로 꾸준히
# 개선). 관측 차원(~2,900) + 120개 동시 행동헤드라는 이 문제의 규모에서는 SB3
# 기본 학습률이 과도했던 것으로 판단, 이 값을 검증된 기본값으로 삼는다.
# target_kl은 그와 별개로 "그래도 한 번의 업데이트가 과도하게 크면 그 자리에서
# 멈춘다"는 2차 안전장치. MultiDiscrete(120)은 SB3가 120개 독립 카테고리
# 분포의 KL을 합산해서 보고하므로, 단일 행동공간에서 흔히 쓰는 값(0.01~0.03)의
# 약 120배 스케일로 잡는다.
PPO_DEFAULT_PARAMS: dict = dict(
    seed=42,
    verbose=0,
    policy_kwargs=POLICY_KWARGS,
    learning_rate=3e-5,
    target_kl=3.0,
)


def date_to_idx(panel: Panel, date) -> int:
    """panel.dates 상에서 date(정확히 일치하는 값)의 인덱스를 찾는다."""
    return int(np.searchsorted(panel.dates, np.datetime64(pd.Timestamp(date))))


def official_split_indices(panel: Panel) -> tuple[int, int, int]:
    """src.models.split.split_train_test와 동일한 cutoff/test_start를 재사용해
    (train_end_idx, test_start_idx, test_end_idx)를 반환한다 — train.parquet/
    test.parquet(Phase 1 공식 분할, backtest_report_v3.md 구간)과 정확히 동일한
    경계."""
    _, _, train_cutoff, test_start = split_train_test(pd.DataFrame({"date": panel.dates}))
    train_end_idx = date_to_idx(panel, train_cutoff)
    test_start_idx = date_to_idx(panel, test_start)
    test_end_idx = len(panel.dates) - 1
    return train_end_idx, test_start_idx, test_end_idx


@dataclass
class FoldIndices:
    fold: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int


def walk_forward_fold_indices(panel: Panel, n_folds: int = 5) -> list[FoldIndices]:
    """generate_walk_forward_folds()가 반환하는 날짜 경계를 panel.dates 상의
    인덱스로 변환한다. 폴드 정의 로직 자체는 새로 만들지 않고 그대로 재사용."""
    folds = generate_walk_forward_folds(pd.DataFrame({"date": panel.dates}), n_folds=n_folds)
    return [
        FoldIndices(
            fold=f.fold,
            train_end_idx=date_to_idx(panel, f.train_cutoff),
            test_start_idx=date_to_idx(panel, f.test_start),
            test_end_idx=date_to_idx(panel, f.test_end),
        )
        for f in folds
    ]


def train_policy(
    panel: Panel,
    train_end_idx: int,
    total_timesteps: int,
    train_start_idx: int = 0,
    episode_length_days: int = EPISODE_LENGTH_DAYS,
    seed: int = 42,
    ppo_params: dict | None = None,
    n_envs: int = 1,
) -> tuple[PPO, StandardScaler]:
    """[train_start_idx, train_end_idx] 구간에만 스케일러를 fit하고, 그 구간
    안에서 무작위 시작 에피소드로 PPO를 학습한다. 반환된 스케일러는 이 정책을
    평가할 때도 그대로(다시 fit하지 않고) 써야 한다 — obs_scaler.py의
    train-only 원칙.

    n_envs>1이면 DummyVecEnv로 여러 환경을 동시에 굴린다. 실측 결과(CLAUDE.md
    "Phase 3 구체 사양" > "연산 자원" 참고) 이 환경의 병목은 env.step() 자체가
    아니라 PPO 정책망이 롤아웃 수집 중 매 스텝 배치크기 1로 순차 추론하는
    부분이었다 — n_envs개 환경을 동시에 돌리면 그 관측들을 한 번에 배치로
    묶어 추론하게 되어 이 병목이 완화된다. SubprocVecEnv(진짜 멀티프로세스)가
    아니라 DummyVecEnv(단일 프로세스 내 순차 스텝)를 쓰는 이유: env.step() 자체는
    이미 충분히 빠르므로(0.14ms) 프로세스 분리로 얻는 이득은 적은 반면, panel
    배열(수십MB)을 서브프로세스마다 복제하는 오버헤드만 늘어난다.
    """
    scaler = fit_obs_scaler(panel, train_end_date=panel.dates[train_end_idx])

    if n_envs == 1:
        env: TradingEnv | VecEnv = TradingEnv(
            panel,
            scaler=scaler,
            date_start_idx=train_start_idx,
            date_end_idx=train_end_idx,
            episode_length_days=episode_length_days,
            random_start=True,
        )
    else:
        def _make_env():
            return TradingEnv(
                panel,
                scaler=scaler,
                date_start_idx=train_start_idx,
                date_end_idx=train_end_idx,
                episode_length_days=episode_length_days,
                random_start=True,
            )

        env = DummyVecEnv([_make_env for _ in range(n_envs)])

    params = {**PPO_DEFAULT_PARAMS, **(ppo_params or {}), "seed": seed}
    model = PPO("MlpPolicy", env, **params)
    model.learn(total_timesteps=total_timesteps)
    return model, scaler


def save_checkpoint(model: PPO, scaler: StandardScaler, checkpoints_dir: Path, name: str, meta: dict) -> None:
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoints_dir / f"{name}.zip"))
    save_obs_scaler(scaler, checkpoints_dir / f"{name}_scaler.pkl")
    meta_path = checkpoints_dir / f"{name}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_checkpoint(checkpoints_dir: Path, name: str) -> tuple[PPO, StandardScaler]:
    model = PPO.load(str(checkpoints_dir / f"{name}.zip"))
    scaler = load_obs_scaler(checkpoints_dir / f"{name}_scaler.pkl")
    return model, scaler


def _checkpoint_exists(checkpoints_dir: Path, name: str) -> bool:
    return (checkpoints_dir / f"{name}.zip").exists()


def run(
    data_root: Path | None = None,
    n_folds: int = 5,
    total_timesteps_per_fold: int = 100_000,
    include_event_features: bool = True,
    n_envs: int = 1,
    ppo_params: dict | None = None,
    resume: bool = True,
) -> None:
    """Walk-forward 5폴드 + 공식 단일분할 정책을 전부 학습해 체크포인트로 저장한다.

    주의: 폴드당 total_timesteps_per_fold가 크면(기본 10만) 전체 실행 시간이
    상당할 수 있다 — 먼저 작은 timestep 예산으로 파일럿을 돌려 벽시계 시간을
    가늠하고, 필요하면 Colab GPU로 옮길 것(CLAUDE.md "Phase 3 구체 사양" >
    "연산 자원" 참고). n_envs>1(DummyVecEnv)로 정책망 추론을 배치화하면 이
    저장소의 4코어 로컬 환경 기준 실측으로 n_envs=8에서 약 2.6배 개선됐다
    (36.01ms/step -> 13.90ms/step) — ppo_params에 n_steps를 n_envs로 나눠
    버퍼 크기를 맞추는 걸 권장(예: n_envs=8이면 n_steps=256으로 버퍼=2048 유지).

    resume=True(기본)면 이미 체크포인트(.zip)가 있는 폴드/공식분할은 다시
    학습하지 않고 건너뛴다 — 전체 실행이 여러 시간 걸릴 수 있는 장시간
    무인 작업이라, 중간에 끊겨도 이미 끝난 폴드는 재사용하고 이어서
    진행할 수 있게 하기 위함(progress_log.json/score_events.py의 재개
    원칙과 동일).
    """
    data_root = data_root or get_data_root()
    checkpoints_dir = data_root / "checkpoints"

    panel = build_panel(data_root=data_root, include_event_features=include_event_features)

    for f in walk_forward_fold_indices(panel, n_folds=n_folds):
        name = f"rl_policy_fold{f.fold}"
        if resume and _checkpoint_exists(checkpoints_dir, name):
            print(f"[train_agent] fold {f.fold} 체크포인트 이미 존재 -> 건너뜀")
            continue

        model, scaler = train_policy(
            panel,
            train_end_idx=f.train_end_idx,
            total_timesteps=total_timesteps_per_fold,
            seed=42,
            n_envs=n_envs,
            ppo_params=ppo_params,
        )
        meta = {
            "fold": f.fold,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "total_timesteps": total_timesteps_per_fold,
            "train_end_idx": f.train_end_idx,
            "test_start_idx": f.test_start_idx,
            "test_end_idx": f.test_end_idx,
            "include_event_features": include_event_features,
            "n_envs": n_envs,
        }
        save_checkpoint(model, scaler, checkpoints_dir, name, meta)
        print(f"[train_agent] fold {f.fold} 학습 완료 -> {name}.zip")

    train_end_idx, test_start_idx, test_end_idx = official_split_indices(panel)
    if resume and _checkpoint_exists(checkpoints_dir, "rl_policy_v1"):
        print("[train_agent] 공식 단일분할 체크포인트 이미 존재 -> 건너뜀")
        return

    model, scaler = train_policy(
        panel,
        train_end_idx=train_end_idx,
        total_timesteps=total_timesteps_per_fold,
        seed=42,
        n_envs=n_envs,
        ppo_params=ppo_params,
    )
    meta = {
        "fold": "official",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "total_timesteps": total_timesteps_per_fold,
        "train_end_idx": train_end_idx,
        "test_start_idx": test_start_idx,
        "test_end_idx": test_end_idx,
        "include_event_features": include_event_features,
        "n_envs": n_envs,
    }
    save_checkpoint(model, scaler, checkpoints_dir, "rl_policy_v1", meta)
    print("[train_agent] 공식 단일분할 정책 학습 완료 -> rl_policy_v1.zip")


if __name__ == "__main__":
    # n_envs=8/n_steps=256(버퍼=2048)은 이 저장소 로컬 4코어 환경에서 실측 검증된
    # 설정(13.90ms/step, CLAUDE.md "Phase 3 구체 사양" > "연산 자원" 참고).
    # verbose=1: 장시간(약 11.6시간 추정) 무인 실행이라 SB3 자체 롤아웃 로그
    # (ep_rew_mean 등)를 nohup 로그 파일로 남겨 진행 상황을 볼 수 있게 한다.
    run(
        total_timesteps_per_fold=500_000,
        n_envs=8,
        ppo_params={"n_steps": 256, "verbose": 1},
    )
