import numpy as np
import pandas as pd
import pytest

from src.rl.obs_scaler import fit_obs_scaler, load_obs_scaler, save_obs_scaler, transform_obs_features
from src.rl.panel import Panel

FEATURE_NAMES = ["ma5_ratio", "model_v3_prob", "market_id", "size_id", "valid_mask"]


def _make_panel(dates: list[str], tickers: list[str], features: np.ndarray, valid_mask: np.ndarray) -> Panel:
    features = np.asarray(features, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=np.float32)
    return Panel(
        dates=np.array([pd.Timestamp(d).to_datetime64() for d in dates]),
        tickers=list(tickers),
        feature_names=list(FEATURE_NAMES),
        features=features,
        close=np.ones(valid_mask.shape, dtype=np.float32),
        valid_mask=valid_mask,
    )


def test_fit_excludes_masked_rows_from_statistics():
    """ticker B의 2024-01-02는 마스킹(0-채움)이다 — 이게 fit 통계에 섞이면
    ma5_ratio 평균이 10.0에서 어긋난다."""
    features = np.zeros((2, 2, 5), dtype=np.float32)
    valid_mask = np.array([[1, 1], [1, 0]], dtype=np.float32)
    features[..., 0] = [[10.0, 10.0], [10.0, 0.0]]
    features[..., -1] = valid_mask

    panel = _make_panel(["2024-01-01", "2024-01-02"], ["A", "B"], features, valid_mask)
    scaler = fit_obs_scaler(panel, train_end_date="2024-01-02")

    assert scaler.mean_[0] == pytest.approx(10.0)


def test_unscaled_channels_are_never_modified():
    features = np.zeros((3, 1, 5), dtype=np.float32)
    features[..., 0] = [[5.0], [7.0], [9.0]]
    features[..., 1] = [[0.3], [0.6], [0.9]]  # model_v3_prob
    features[..., 2] = [[1.0], [1.0], [1.0]]  # market_id
    features[..., 3] = [[2.0], [2.0], [2.0]]  # size_id
    valid_mask = np.ones((3, 1), dtype=np.float32)
    features[..., -1] = valid_mask

    panel = _make_panel(["2024-01-01", "2024-01-02", "2024-01-03"], ["A"], features, valid_mask)
    scaler = fit_obs_scaler(panel, train_end_date="2024-01-03")
    out = transform_obs_features(panel, scaler)

    np.testing.assert_array_equal(out[..., 1], features[..., 1])
    np.testing.assert_array_equal(out[..., 2], features[..., 2])
    np.testing.assert_array_equal(out[..., 3], features[..., 3])


def test_masked_rows_stay_zero_after_transform():
    features = np.zeros((2, 2, 5), dtype=np.float32)
    valid_mask = np.array([[1, 1], [1, 0]], dtype=np.float32)
    features[..., 0] = [[10.0, 10.0], [10.0, 0.0]]
    features[..., -1] = valid_mask

    panel = _make_panel(["2024-01-01", "2024-01-02"], ["A", "B"], features, valid_mask)
    scaler = fit_obs_scaler(panel, train_end_date="2024-01-02")
    out = transform_obs_features(panel, scaler)

    assert np.all(out[1, 1, :] == 0.0)  # day2(index1)의 ticker B는 마스킹


def test_transform_uses_train_only_statistics_not_eval():
    """train 구간(1/1~1/2)의 ma5_ratio는 항상 10.0, eval 구간(6/1)엔 극단값
    1000.0 — eval 값이 fit에 섞였다면 평균/분산이 eval 쪽으로 끌려간다. 이는
    SB3 VecNormalize를 평가 중에도 온라인 갱신 상태로 두면 생기는 누수와
    동일한 종류의 문제를 사전에 막기 위한 테스트다."""
    features = np.zeros((3, 1, 5), dtype=np.float32)
    features[..., 0] = [[10.0], [10.0], [1000.0]]
    valid_mask = np.ones((3, 1), dtype=np.float32)
    features[..., -1] = valid_mask

    panel = _make_panel(["2024-01-01", "2024-01-02", "2024-06-01"], ["A"], features, valid_mask)
    scaler = fit_obs_scaler(panel, train_end_date="2024-01-02")  # eval(6/1) 제외

    assert scaler.mean_[0] == pytest.approx(10.0)
    assert scaler.var_[0] == pytest.approx(0.0, abs=1e-6)


def test_save_load_round_trip(tmp_path):
    features = np.zeros((2, 1, 5), dtype=np.float32)
    features[..., 0] = [[3.0], [5.0]]
    valid_mask = np.ones((2, 1), dtype=np.float32)
    features[..., -1] = valid_mask
    panel = _make_panel(["2024-01-01", "2024-01-02"], ["A"], features, valid_mask)

    scaler = fit_obs_scaler(panel, train_end_date="2024-01-02")
    path = tmp_path / "scaler.pkl"
    save_obs_scaler(scaler, path)
    loaded = load_obs_scaler(path)

    out_original = transform_obs_features(panel, scaler)
    out_loaded = transform_obs_features(panel, loaded)
    np.testing.assert_array_equal(out_original, out_loaded)
