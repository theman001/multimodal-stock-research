import numpy as np
import pandas as pd
import pytest

from src.models.split import split_train_test
from src.models.walk_forward import generate_walk_forward_folds
from src.rl.panel import Panel
from src.rl.train_agent import (
    PPO_DEFAULT_PARAMS,
    _checkpoint_exists,
    date_to_idx,
    load_checkpoint,
    official_split_indices,
    save_checkpoint,
    train_policy,
    walk_forward_fold_indices,
)

FEATURE_NAMES = ["f0", "f1", "valid_mask"]


def _make_panel(n_dates: int, n_tickers: int = 3) -> Panel:
    dates = np.array(
        [pd.Timestamp("2020-01-01") + pd.Timedelta(days=i) for i in range(n_dates)]
    ).astype("datetime64[ns]")
    close = 100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, size=(n_dates, n_tickers)), axis=0)
    valid_mask = np.ones((n_dates, n_tickers), dtype=np.float32)
    features = np.zeros((n_dates, n_tickers, 3), dtype=np.float32)
    features[..., -1] = valid_mask
    tickers = [f"T{i}" for i in range(n_tickers)]
    market_id = [i % 2 for i in range(n_tickers)]
    return Panel(
        dates=dates,
        tickers=tickers,
        feature_names=list(FEATURE_NAMES),
        features=features,
        close=close.astype(np.float32),
        valid_mask=valid_mask,
        market_id=np.array(market_id, dtype=np.int64),
    )


def test_date_to_idx_finds_exact_match():
    panel = _make_panel(10)
    idx = date_to_idx(panel, panel.dates[5])
    assert idx == 5


def test_official_split_indices_matches_split_train_test():
    panel = _make_panel(500)
    dates_df = pd.DataFrame({"date": panel.dates})
    _, _, expected_cutoff, expected_test_start = split_train_test(dates_df)

    train_end_idx, test_start_idx, test_end_idx = official_split_indices(panel)

    assert panel.dates[train_end_idx] == np.datetime64(expected_cutoff)
    assert panel.dates[test_start_idx] == np.datetime64(expected_test_start)
    assert test_end_idx == len(panel.dates) - 1


def test_walk_forward_fold_indices_matches_generate_walk_forward_folds():
    panel = _make_panel(500)
    dates_df = pd.DataFrame({"date": panel.dates})
    expected_folds = generate_walk_forward_folds(dates_df, n_folds=5)

    folds = walk_forward_fold_indices(panel, n_folds=5)

    assert len(folds) == len(expected_folds)
    for actual, expected in zip(folds, expected_folds):
        assert actual.fold == expected.fold
        assert panel.dates[actual.train_end_idx] == np.datetime64(expected.train_cutoff)
        assert panel.dates[actual.test_start_idx] == np.datetime64(expected.test_start)
        assert panel.dates[actual.test_end_idx] == np.datetime64(expected.test_end)


def test_train_policy_runs_and_produces_a_usable_policy():
    """PPO 학습 파이프라인이 실제로 동작하는지(플러밍 확인) — 학습 품질이
    아니라 배관을 검증하는 아주 작은 예산의 스모크 테스트."""
    panel = _make_panel(40, n_tickers=3)
    model, scaler = train_policy(
        panel,
        train_end_idx=30,
        total_timesteps=32,
        episode_length_days=10,
        seed=0,
        ppo_params={"n_steps": 16, "batch_size": 8},
    )

    obs = np.zeros(model.observation_space.shape, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (3,)
    assert model.action_space.contains(action)


def test_train_policy_with_multiple_envs_produces_a_usable_policy():
    """n_envs>1(DummyVecEnv 경로)도 단일 환경과 동일하게 정상 동작해야 한다."""
    panel = _make_panel(40, n_tickers=3)
    model, scaler = train_policy(
        panel,
        train_end_idx=30,
        total_timesteps=32,
        episode_length_days=10,
        seed=0,
        n_envs=4,
        ppo_params={"n_steps": 8, "batch_size": 8},
    )

    obs = np.zeros(model.observation_space.shape, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (3,)
    assert model.action_space.contains(action)


def test_save_and_load_checkpoint_round_trip(tmp_path):
    panel = _make_panel(40, n_tickers=3)
    model, scaler = train_policy(
        panel,
        train_end_idx=30,
        total_timesteps=32,
        episode_length_days=10,
        seed=0,
        ppo_params={"n_steps": 16, "batch_size": 8},
    )
    save_checkpoint(model, scaler, tmp_path, "test_policy", {"fold": "test"})

    loaded_model, loaded_scaler = load_checkpoint(tmp_path, "test_policy")

    obs = np.zeros(model.observation_space.shape, dtype=np.float32)
    action_original, _ = model.predict(obs, deterministic=True)
    action_loaded, _ = loaded_model.predict(obs, deterministic=True)
    np.testing.assert_array_equal(action_original, action_loaded)
    assert np.allclose(loaded_scaler.mean_, scaler.mean_)


def test_checkpoint_exists_detects_presence_and_absence(tmp_path):
    assert not _checkpoint_exists(tmp_path, "rl_policy_fold1")

    panel = _make_panel(40, n_tickers=3)
    model, scaler = train_policy(
        panel,
        train_end_idx=30,
        total_timesteps=32,
        episode_length_days=10,
        seed=0,
        ppo_params={"n_steps": 16, "batch_size": 8},
    )
    save_checkpoint(model, scaler, tmp_path, "rl_policy_fold1", {"fold": 1})

    assert _checkpoint_exists(tmp_path, "rl_policy_fold1")
    assert not _checkpoint_exists(tmp_path, "rl_policy_fold2")


def test_default_learning_rate_is_the_stability_validated_value():
    """learning_rate=3e-4(SB3 기본값)로는 실 데이터 파일럿에서 approx_kl이
    이터레이션마다 폭주(9->93->183)하는 걸 실측으로 확인했고, 3e-5로 낮추자
    10개 이터레이션 내내 0.07~0.09로 안정화됐다 — 이 값이 조용히 원복되지
    않도록 회귀 방지."""
    assert PPO_DEFAULT_PARAMS["learning_rate"] == pytest.approx(3e-5)
    assert PPO_DEFAULT_PARAMS["target_kl"] == pytest.approx(3.0)
