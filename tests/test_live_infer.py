import numpy as np
import pandas as pd
import pytest

from src.live.infer import load_checkpoint_checked, predict_action
from src.rl.panel import Panel
from src.rl.train_agent import save_checkpoint, train_policy

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


def _train_and_save(tmp_path, name="rl_policy_v1", include_event_features=True):
    panel = _make_panel(40, n_tickers=3)
    model, scaler = train_policy(
        panel,
        train_end_idx=30,
        total_timesteps=32,
        episode_length_days=10,
        seed=0,
        ppo_params={"n_steps": 16, "batch_size": 8},
    )
    save_checkpoint(model, scaler, tmp_path, name, {"include_event_features": include_event_features})
    return model


def test_predict_action_returns_valid_action(tmp_path):
    model = _train_and_save(tmp_path, include_event_features=True)
    obs = np.zeros(model.observation_space.shape, dtype=np.float32)

    action = predict_action(tmp_path, "rl_policy_v1", obs, include_event_features=True)

    assert action.shape == (3,)
    assert model.action_space.contains(action)


def test_predict_action_is_deterministic(tmp_path):
    model = _train_and_save(tmp_path, include_event_features=True)
    obs = np.zeros(model.observation_space.shape, dtype=np.float32)

    action1 = predict_action(tmp_path, "rl_policy_v1", obs, include_event_features=True)
    action2 = predict_action(tmp_path, "rl_policy_v1", obs, include_event_features=True)

    np.testing.assert_array_equal(action1, action2)


def test_mismatched_include_event_features_raises_clear_error(tmp_path):
    _train_and_save(tmp_path, include_event_features=True)

    with pytest.raises(ValueError, match="include_event_features"):
        load_checkpoint_checked(tmp_path, "rl_policy_v1", include_event_features=False)


def test_non_finite_obs_raises(tmp_path):
    """infer.py는 observation.py가 만든 obs만 받는다고 전제하지 않는다 —
    NaN/Inf가 섞인 obs가 SB3 내부에서 조용히 잘못된 예측을 내는 대신 여기서
    이름 붙은 에러로 막혀야 한다(재검토로 발견)."""
    model = _train_and_save(tmp_path, include_event_features=True)
    bad_obs = np.zeros(model.observation_space.shape, dtype=np.float32)
    bad_obs[0] = np.nan

    with pytest.raises(ValueError, match="유한하지 않은"):
        predict_action(tmp_path, "rl_policy_v1", bad_obs, include_event_features=True)


def test_obs_shape_mismatch_raises_clear_error_even_with_matching_flag(tmp_path):
    """include_event_features는 맞아도 rl_ticker_universe.json이 학습 당시와
    다른 구성으로 덮어써지면(다른 목적의 build_panel() 재실행 등) 관측 차원이
    달라질 수 있다 — include_event_features 체크만으로는 못 잡으므로 실제
    obs.shape를 직접 비교해 막아야 한다(재검토로 발견)."""
    model = _train_and_save(tmp_path, include_event_features=True)
    wrong_shape_obs = np.zeros(model.observation_space.shape[0] + 3, dtype=np.float32)

    with pytest.raises(ValueError, match="observation_space"):
        predict_action(tmp_path, "rl_policy_v1", wrong_shape_obs, include_event_features=True)
