import numpy as np
import pandas as pd
import pytest

from src.rl.evaluate import (
    _load_checkpoint_checked,
    _window_realized_return_df,
    rollout_policy_daily_returns,
    rollout_policy_with_positioning_stats,
)
from src.rl.panel import Panel
from src.rl.train_agent import save_checkpoint, train_policy
from src.rl.trading_env import ACTION_BUY, ACTION_HOLD

FEATURE_NAMES = ["f0", "f1", "valid_mask"]


class _AlwaysHoldModel:
    """PPO.predict()와 같은 인터페이스를 흉내내는 가짜 정책 — 항상 HOLD만
    낸다. 실제 정책 학습 없이 rollout_policy_daily_returns()를 빠르고
    결정적으로 검증하기 위함(all-HOLD면 거래가 없어 NAV가 정확히 1.0으로
    유지돼야 함 — trading_env.py 개발 때 이미 확인한 불변식과 동일)."""

    def __init__(self, n: int):
        self.n = n

    def predict(self, obs, deterministic: bool = True):
        return np.full(self.n, ACTION_HOLD), None


def _make_panel(n_dates: int, n_tickers: int = 2) -> Panel:
    dates = np.array(
        [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(n_dates)]
    ).astype("datetime64[ns]")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, size=(n_dates, n_tickers)), axis=0)
    valid_mask = np.ones((n_dates, n_tickers), dtype=np.float32)
    features = np.zeros((n_dates, n_tickers, 3), dtype=np.float32)
    features[..., -1] = valid_mask
    return Panel(
        dates=dates,
        tickers=[f"T{i}" for i in range(n_tickers)],
        feature_names=list(FEATURE_NAMES),
        features=features,
        close=close.astype(np.float32),
        valid_mask=valid_mask,
        market_id=np.array([i % 2 for i in range(n_tickers)], dtype=np.int64),
    )


def test_rollout_all_hold_policy_keeps_nav_flat():
    panel = _make_panel(10, n_tickers=2)
    model = _AlwaysHoldModel(n=2)

    daily_returns = rollout_policy_daily_returns(panel, model, scaler=None, date_start_idx=0, date_end_idx=9)

    assert len(daily_returns) == 10
    assert (daily_returns == 0.0).all()  # 거래가 전혀 없으니 NAV가 그대로 유지돼야 함
    assert daily_returns.index[0] == pd.Timestamp(panel.dates[0])
    assert daily_returns.index[-1] == pd.Timestamp(panel.dates[9])


def test_rollout_returns_series_indexed_by_actual_calendar_dates():
    panel = _make_panel(5, n_tickers=1)
    model = _AlwaysHoldModel(n=1)

    daily_returns = rollout_policy_daily_returns(panel, model, scaler=None, date_start_idx=1, date_end_idx=3)

    assert len(daily_returns) == 3
    expected_dates = [pd.Timestamp(panel.dates[i]) for i in (1, 2, 3)]
    assert list(daily_returns.index) == expected_dates


def test_window_realized_return_df_filters_date_range_and_computes_returns():
    features_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["AAA", "AAA", "AAA", "AAA"],
            "market_id": [0, 0, 0, 0],
        }
    )
    ohlcv_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["AAA", "AAA", "AAA", "AAA"],
            "close": [100.0, 110.0, 121.0, 133.1],
        }
    )

    df = _window_realized_return_df(
        features_df, ohlcv_df, pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")
    )

    # 구간 밖(1/1, 1/4)은 결과 행으로 나오지 않되, 구간 안 마지막 날(1/3)의
    # realized_return을 구하려면 구간 밖인 1/4의 종가가 내부적으로 쓰여야 한다
    # — 먼저 자르고 계산하면 1/3이 "다음 행 없음"으로 NaN 처리돼 사라지는 게
    # 실제로 겪은 버그였다(RL 롤아웃은 구간 끝까지 꽉 채워 쓰는데 이 함수만
    # 하루 짧아짐). 반드시 구간의 마지막 날짜(date_end)가 살아있는지 검증한다.
    assert set(df["date"]) == {pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")}
    row_0102 = df[df["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert row_0102["realized_return"] == pytest.approx(121.0 / 110.0 - 1)
    row_0103 = df[df["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    assert row_0103["realized_return"] == pytest.approx(133.1 / 121.0 - 1)


class _AlwaysBuyModel:
    def __init__(self, n: int):
        self.n = n

    def predict(self, obs, deterministic: bool = True):
        return np.full(self.n, ACTION_BUY), None


def test_positioning_stats_exclude_forced_final_liquidation():
    """마지막 스텝은 에피소드 종료 강제청산이라 n_holding=0이 된다 — 이게
    '평상시 얼마나 투자했는지' 통계에 섞이면 실제보다 낮게 나온다. 항상 BUY만
    내는 정책으로, 마지막 스텝 직전까지는 종목을 계속 보유해야 하고
    avg_n_holding이 0에 가깝게 왜곡되면 안 된다."""
    panel = _make_panel(5, n_tickers=3)
    model = _AlwaysBuyModel(n=3)

    _, stats = rollout_policy_with_positioning_stats(panel, model, scaler=None, date_start_idx=0, date_end_idx=4)

    assert stats["avg_n_holding"] > 0  # 마지막 강제청산(0)에 끌려 내려가지 않음
    assert stats["avg_cash_weight"] < 1.0


def test_load_checkpoint_checked_rejects_include_event_features_mismatch(tmp_path):
    """체크포인트가 include_event_features=True로 학습됐는데 평가가 False로
    패널을 만들면 관측 차원이 어긋나 sklearn/SB3 내부에서 알아보기 힘든 shape
    에러가 난다 — 이름 붙은 명확한 에러로 먼저 막아야 한다."""
    panel = _make_panel(40, n_tickers=3)
    model, scaler = train_policy(
        panel,
        train_end_idx=30,
        total_timesteps=32,
        episode_length_days=10,
        seed=0,
        ppo_params={"n_steps": 16, "batch_size": 8},
    )
    save_checkpoint(model, scaler, tmp_path, "rl_policy_fold1", {"fold": 1, "include_event_features": True})

    with pytest.raises(ValueError, match="include_event_features"):
        _load_checkpoint_checked(tmp_path, "rl_policy_fold1", include_event_features=False)

    # 값이 일치하면 정상적으로 로드돼야 함
    loaded_model, loaded_scaler = _load_checkpoint_checked(tmp_path, "rl_policy_fold1", include_event_features=True)
    assert loaded_model is not None
    assert loaded_scaler is not None
