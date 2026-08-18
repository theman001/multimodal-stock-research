import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import kr_round_trip_cost, us_round_trip_cost
from src.rl.panel import Panel
from src.rl.trading_env import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    MAX_MASKED_STREAK_DAYS,
    REWARD_CLIP,
    TradingEnv,
)

FEATURE_NAMES = ["f0", "f1", "valid_mask"]
US_HALF_COST = us_round_trip_cost() / 2.0
KR_HALF_COST = kr_round_trip_cost() / 2.0


def _make_panel(
    n_dates: int,
    close: np.ndarray,  # (n_dates, n_tickers), NaN이면 그 날 가격 없음
    valid_mask: np.ndarray,  # (n_dates, n_tickers)
    market_id: list[int],
) -> Panel:
    n_tickers = close.shape[1]
    close = np.asarray(close, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=np.float32)
    features = np.zeros((n_dates, n_tickers, 3), dtype=np.float32)
    features[..., -1] = valid_mask
    dates = np.array(
        [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(n_dates)]
    ).astype("datetime64[ns]")
    tickers = [f"T{i}" for i in range(n_tickers)]
    return Panel(
        dates=dates,
        tickers=tickers,
        feature_names=list(FEATURE_NAMES),
        features=features,
        close=close,
        valid_mask=valid_mask,
        market_id=np.array(market_id, dtype=np.int64),
    )


def test_reset_returns_correct_shape_and_initial_state():
    panel = _make_panel(
        n_dates=3,
        close=np.full((3, 2), 100.0),
        valid_mask=np.ones((3, 2)),
        market_id=[0, 1],
    )
    env = TradingEnv(panel, date_start_idx=0, date_end_idx=2, random_start=False)
    obs, info = env.reset()

    assert obs.shape == env.observation_space.shape
    assert env.action_space.nvec.tolist() == [3, 3]
    assert env.cash == pytest.approx(1.0)
    assert not env.holding.any()


def test_buy_then_hold_accrues_price_return_without_daily_cost():
    """진입비용은 매수 시점 한 번만 부과되고, 보유 중엔 종가 비율로만 가치가
    바뀐다 — simulate.py처럼 매일 비용이 다시 붙지 않는다는 게 이 환경의 핵심."""
    close = np.array([[100.0, 100.0], [101.0, 100.0], [103.0, 100.0]])
    panel = _make_panel(3, close, np.ones((3, 2)), market_id=[0, 1])  # 티커0=US
    env = TradingEnv(panel, date_start_idx=0, date_end_idx=2, random_start=False, max_position_weight=1.0)
    env.reset()

    env.step(np.array([ACTION_BUY, ACTION_HOLD]))
    expected_pv_after_buy = 1.0 * (1 - US_HALF_COST)
    assert env.position_value[0] == pytest.approx(expected_pv_after_buy)
    assert env.cash == pytest.approx(0.0)
    assert env.holding[0]

    env.step(np.array([ACTION_HOLD, ACTION_HOLD]))
    assert env.position_value[0] == pytest.approx(expected_pv_after_buy * (101.0 / 100.0))
    assert env.cash == pytest.approx(0.0)  # 보유 중엔 비용 없음


def test_buy_then_sell_pays_cost_only_at_the_two_transitions():
    close = np.full((3, 2), 100.0)  # 가격 변동 없음 -> NAV 변화는 순전히 비용 때문
    panel = _make_panel(3, close, np.ones((3, 2)), market_id=[0, 1])
    env = TradingEnv(panel, date_start_idx=0, date_end_idx=2, random_start=False, max_position_weight=1.0)
    env.reset()

    env.step(np.array([ACTION_BUY, ACTION_HOLD]))
    pv_after_buy = 1.0 * (1 - US_HALF_COST)

    env.step(np.array([ACTION_SELL, ACTION_HOLD]))
    expected_cash = pv_after_buy * (1 - US_HALF_COST)
    assert env.cash == pytest.approx(expected_cash)
    assert not env.holding[0]


def test_buy_on_invalid_ticker_is_noop():
    close = np.array([[np.nan, 100.0], [100.0, 100.0], [100.0, 100.0]])
    valid_mask = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    panel = _make_panel(3, close, valid_mask, market_id=[0, 1])
    env = TradingEnv(panel, date_start_idx=0, date_end_idx=2, random_start=False)
    env.reset()

    env.step(np.array([ACTION_BUY, ACTION_HOLD]))

    assert not env.holding[0]
    assert env.cash == pytest.approx(1.0)


def test_capital_allocation_respects_cap_and_does_not_redistribute_leftover():
    close = np.full((3, 2), 100.0)
    panel = _make_panel(3, close, np.ones((3, 2)), market_id=[0, 1])
    env = TradingEnv(panel, date_start_idx=0, date_end_idx=2, random_start=False, max_position_weight=0.3)
    env.reset()

    env.step(np.array([ACTION_BUY, ACTION_BUY]))

    # equal_share = 1.0/2 = 0.5, cap = 0.3*NAV(1.0) = 0.3 -> 둘 다 캡에 걸림
    assert env.cash == pytest.approx(1.0 - 2 * 0.3)
    assert env.position_value[0] == pytest.approx(0.3 * (1 - US_HALF_COST))
    assert env.position_value[1] == pytest.approx(0.3 * (1 - KR_HALF_COST))


def test_forced_liquidation_at_episode_end_overrides_hold_action():
    close = np.full((3, 2), 100.0)
    panel = _make_panel(3, close, np.ones((3, 2)), market_id=[0, 1])
    env = TradingEnv(panel, date_start_idx=0, date_end_idx=2, random_start=False, max_position_weight=1.0)
    env.reset()

    env.step(np.array([ACTION_BUY, ACTION_HOLD]))
    env.step(np.array([ACTION_HOLD, ACTION_HOLD]))
    assert env.holding[0]

    _, _, terminated, truncated, _ = env.step(np.array([ACTION_HOLD, ACTION_HOLD]))  # 마지막 스텝
    assert not env.holding[0]  # 정책은 HOLD를 냈지만 강제 청산됨
    assert truncated
    assert not terminated


def test_masked_streak_beyond_threshold_forces_close():
    """가격이 MAX_MASKED_STREAK_DAYS를 넘게 연속으로 없으면(장기 상장폐지성 공백),
    보유 포지션을 마지막 유효가로 강제 청산한다."""
    n_dates = MAX_MASKED_STREAK_DAYS + 4
    close = np.full((n_dates, 1), np.nan, dtype=np.float64)
    close[0, 0] = 100.0  # 매수 가능한 날은 첫날뿐
    valid_mask = np.zeros((n_dates, 1))
    valid_mask[0, 0] = 1.0

    panel = _make_panel(n_dates, close, valid_mask, market_id=[0])
    env = TradingEnv(
        panel, date_start_idx=0, date_end_idx=n_dates - 1, random_start=False, max_position_weight=1.0
    )
    env.reset()

    env.step(np.array([ACTION_BUY]))
    assert env.holding[0]

    still_holding_at = None
    for _ in range(n_dates - 1):
        env.step(np.array([ACTION_HOLD]))
        if not env.holding[0]:
            still_holding_at = env.masked_streak[0]
            break

    assert still_holding_at is not None  # 강제 청산이 실제로 발생했음
    assert env.cash > 0.0  # 청산 대금이 현금으로 들어옴


def test_reward_reflects_log_nav_change():
    close = np.array([[100.0], [110.0], [110.0]])
    panel = _make_panel(3, close, np.ones((3, 1)), market_id=[0])
    env = TradingEnv(panel, date_start_idx=0, date_end_idx=2, random_start=False, max_position_weight=1.0)
    env.reset()

    _, reward0, _, _, _ = env.step(np.array([ACTION_BUY]))
    nav_after_buy = 1.0 * (1 - US_HALF_COST)
    assert reward0 == pytest.approx(np.log(nav_after_buy / 1.0))

    _, reward1, _, _, _ = env.step(np.array([ACTION_HOLD]))
    nav_after_hold = nav_after_buy * (110.0 / 100.0)
    assert reward1 == pytest.approx(np.log(nav_after_hold / nav_after_buy))


def test_extreme_price_move_clips_reward_but_not_true_nav():
    """실 데이터에서 개별 종목 일간 로그수익률이 -0.84~+0.74까지 나오는 걸
    확인함(042660/CORT, 둘 다 데이터 결함이 아니라 실제 급락/급등) — PPO가
    이런 극단치에 그대로 노출되면 정책 업데이트가 폭주한다(approx_kl 실측
    폭주 확인). reward는 REWARD_CLIP으로 잘리되, NAV 장부(info)는 정확한
    값을 유지해야 한다."""
    close = np.array([[100.0], [20.0], [20.0]])  # 첫날 -80% 폭락
    panel = _make_panel(3, close, np.ones((3, 1)), market_id=[0])
    env = TradingEnv(panel, date_start_idx=0, date_end_idx=2, random_start=False, max_position_weight=1.0)
    env.reset()

    env.step(np.array([ACTION_BUY]))
    _, reward, _, _, info = env.step(np.array([ACTION_HOLD]))  # 폭락 반영되는 스텝

    assert info["raw_reward"] == pytest.approx(np.log(20.0 / 100.0))  # 실제 하락폭 그대로
    assert reward == pytest.approx(-REWARD_CLIP)  # 학습 신호는 클리핑됨
    assert info["nav"] == pytest.approx(env.nav)  # NAV 장부는 클리핑과 무관하게 정확
