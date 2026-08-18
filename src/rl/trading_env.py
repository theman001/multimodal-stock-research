"""Phase 3 — 120종목 포트폴리오 Gymnasium 트레이딩 환경.

CLAUDE.md "Phase 3 구체 사양" 기준. `src/rl/panel.py`가 만드는 정적 패널(시장
데이터만으로 결정되는 부분)에, 이 환경이 스텝마다 계산하는 3개 동적 필드
(holding_flag/unrealized_return/position_weight)를 이어붙여 관측을 만들고,
행동(SELL/HOLD/BUY x 120)을 받아 결정적 자본배분 규칙으로 체결한다.

핵심 불변식(leakage-guard, CLAUDE.md "RL 특화 누수 체크리스트" 참고):
- t시점 관측은 t시점까지의 정보만 사용한다(panel.py가 이미 보장).
- t시점 상태로 결정한 행동은 t종가에 체결되고, 그 손익은 t+1로 넘어가는
  step() 호출에서 정산된다 — simulate.py::add_realized_return이 close.shift(-1)로
  실현수익률을 계산하는 것과 동일한 타이밍 규칙.
- 거래비용 타이밍은 simulate.py(신호가 있는 모든 날 매번 부과)와 다르다 — 이
  환경은 진짜 포지션 상태(holding)가 있으므로 포지션을 열 때/닫을 때만
  round_trip_cost_for_market()의 절반씩 부과한다. 요율 자체는 costs.py 그대로
  재사용(비용 "타이밍"만 다름, "요율"은 안 바뀜).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler

from src.backtest.costs import kr_round_trip_cost, us_round_trip_cost
from src.rl.obs_scaler import transform_obs_features
from src.rl.panel import Panel

ACTION_SELL = 0
ACTION_HOLD = 1
ACTION_BUY = 2

MAX_POSITION_WEIGHT = 0.05  # 신규 진입 시 한 종목이 NAV의 이 비율을 넘지 못함
EPISODE_LENGTH_DAYS = 252  # 학습 에피소드 길이(약 1거래년)
MAX_MASKED_STREAK_DAYS = 10  # simulate.py의 MAX_REALIZED_RETURN_GAP_DAYS와 동일한 값·취지

# 실제 데이터 검증 결과 개별 종목 일간 로그수익률이 -0.84~+0.74까지도 나온다
# (예: 042660 2017-10-30 -83.7%, CORT 2025-03-31 +73.75% — 둘 다 실제 급락/
# 급등이지 액면분할 등 데이터 결함이 아님을 조사로 확인함, 상세는
# data/reports/phase3_reward_clipping_investigation.md 참고). PPO는 보상
# 스케일에 민감해서 이런 진짜 극단치가 그대로 들어가면 정책 업데이트가
# 폭주한다(approx_kl이 반복마다 계속 커지는 걸 실측으로 확인) — 실제 NAV/포지션
# 장부는 클리핑 없이 정확히 추적하고(평가·리포트용), PPO에 돌려주는 학습
# 신호(reward)만 이 값으로 clip한다. 최대 20개 포지션(5% 상한 x 20)에 걸친
# 분산을 감안해도 여유 있게 큰 값으로 잡았다 — Atari DQN의 보상 clip[-1,1]과
# 같은 취지의 표준적인 RL 안정화 기법.
REWARD_CLIP = 0.15

_KR_ROUND_TRIP = kr_round_trip_cost()
_US_ROUND_TRIP = us_round_trip_cost()


def _half_round_trip_cost(market_id: np.ndarray) -> np.ndarray:
    """costs.py::round_trip_cost_for_market과 동일한 시장 판정(market_id==1 -> KR)을
    벡터화한 것 — 요율은 그대로, 진입/청산에 절반씩 나눠 부과하기 위한 절반값."""
    return np.where(market_id == 1, _KR_ROUND_TRIP, _US_ROUND_TRIP) / 2.0


class TradingEnv(gym.Env):
    """단일 정책이 120종목 포트폴리오 전체를 매 스텝 동시에 판단하는 환경.

    panel: src.rl.panel.build_panel()의 결과. scaler를 주면(fit_obs_scaler로
    미리 fit된 것) 생성 시점에 한 번만 transform해서 들고 있는다 — 매 step()마다
    다시 스케일링하지 않는다.
    date_start_idx/date_end_idx: panel.dates 상에서 이 환경이 쓸 수 있는 구간
    (예: 어떤 walk-forward 폴드의 train 구간, 또는 공식 test.parquet 구간).
    random_start=True(학습용): reset()마다 이 구간 내에서 무작위 시작점을 뽑아
    episode_length_days만큼 진행. random_start=False(평가용): 구간 전체를
    한 번, 결정적으로 통과.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: Panel,
        scaler: StandardScaler | None = None,
        date_start_idx: int = 0,
        date_end_idx: int | None = None,
        episode_length_days: int = EPISODE_LENGTH_DAYS,
        max_position_weight: float = MAX_POSITION_WEIGHT,
        initial_nav: float = 1.0,
        random_start: bool = True,
    ):
        super().__init__()
        self.panel = panel
        self.n = len(panel.tickers)
        self.d_static = len(panel.feature_names)  # valid_mask 포함
        self.d_per_ticker = self.d_static + 3  # + holding_flag/unrealized_return/position_weight

        self.date_start_idx = date_start_idx
        self.date_end_idx = date_end_idx if date_end_idx is not None else len(panel.dates) - 1
        if not (0 <= self.date_start_idx <= self.date_end_idx < len(panel.dates)):
            raise ValueError("date_start_idx/date_end_idx가 panel.dates 범위를 벗어남")

        self.episode_length_days = episode_length_days
        self.max_position_weight = max_position_weight
        self.initial_nav = initial_nav
        self.random_start = random_start

        if random_start and (self.date_end_idx - self.date_start_idx + 1) < episode_length_days:
            raise ValueError("지정된 날짜 구간이 episode_length_days보다 짧음")

        self.obs_features = (
            transform_obs_features(panel, scaler) if scaler is not None else panel.features
        )

        self.action_space = spaces.MultiDiscrete([3] * self.n)
        obs_dim = self.n * self.d_per_ticker + 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 에피소드 런타임 상태 — reset()에서 초기화됨
        self.t = self.date_start_idx
        self.episode_start_idx = self.date_start_idx
        self.episode_end_idx = self.date_end_idx
        self.cash = float(initial_nav)
        self.nav = float(initial_nav)
        self.position_value = np.zeros(self.n, dtype=np.float64)
        self.last_close = np.full(self.n, np.nan, dtype=np.float64)
        self.entry_close = np.full(self.n, np.nan, dtype=np.float64)
        self.holding = np.zeros(self.n, dtype=bool)
        self.masked_streak = np.zeros(self.n, dtype=np.int64)

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if self.random_start:
            latest_start = self.date_end_idx - self.episode_length_days + 1
            start = int(self.np_random.integers(self.date_start_idx, latest_start + 1))
            end = start + self.episode_length_days - 1
        else:
            start = self.date_start_idx
            end = self.date_end_idx

        self.episode_start_idx = start
        self.episode_end_idx = end
        self.t = start

        self.cash = float(self.initial_nav)
        self.nav = float(self.initial_nav)
        self.position_value = np.zeros(self.n, dtype=np.float64)
        self.last_close = np.full(self.n, np.nan, dtype=np.float64)
        self.entry_close = np.full(self.n, np.nan, dtype=np.float64)
        self.holding = np.zeros(self.n, dtype=bool)
        self.masked_streak = np.zeros(self.n, dtype=np.int64)

        obs = self._build_observation(self.t)
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action)
        t = self.t
        is_terminal_step = t == self.episode_end_idx

        valid_today = self.panel.valid_mask[t].astype(bool)
        close_today = self.panel.close[t].astype(np.float64)
        has_price_today = ~np.isnan(close_today)

        # 1) 보유 중인 포지션을 오늘 종가로 마킹(가능한 경우만) — 마스킹된 날은
        #    가치를 그대로 들고만 가고 masked_streak를 늘린다.
        mark_to_market = self.holding & has_price_today
        if mark_to_market.any():
            ratio = close_today[mark_to_market] / self.last_close[mark_to_market]
            self.position_value[mark_to_market] *= ratio
            self.last_close[mark_to_market] = close_today[mark_to_market]
            self.masked_streak[mark_to_market] = 0

        still_masked = self.holding & ~has_price_today
        self.masked_streak[still_masked] += 1

        # 2) 장기간(>MAX_MASKED_STREAK_DAYS) 마스킹된 보유 포지션은 강제 청산
        force_close = self.holding & (self.masked_streak > MAX_MASKED_STREAK_DAYS)
        self._close_positions(force_close)

        # 3) SELL 처리(가격이 있어야 체결 가능) — BUY보다 먼저 처리해서 같은
        #    스텝에 판 현금을 그 스텝의 BUY 배분에 쓸 수 있게 한다.
        sell_mask = self.holding & has_price_today & (action == ACTION_SELL)
        self._close_positions(sell_mask)

        # 4) 에피소드 마지막 스텝은 행동과 무관하게 강제 전량 청산.
        #    가격이 없는 극단적 케이스도 포함해서 전부 청산한다 —
        #    position_value는 마지막으로 가격이 있었던 시점의 마킹값을 그대로
        #    들고 있으므로(마스킹된 동안엔 갱신되지 않음), 별도로 가격을
        #    넘겨줄 필요 없이 이 한 번의 호출로 "마지막 유효가 기준 청산"이 된다.
        if is_terminal_step:
            self._close_positions(self.holding)

        # 5) BUY 처리 (마지막 스텝엔 신규 진입 없음)
        if not is_terminal_step:
            buy_mask = (~self.holding) & valid_today & has_price_today & (action == ACTION_BUY)
            buy_idx = np.where(buy_mask)[0]
            if len(buy_idx) > 0:
                nav_now = self.cash + self.position_value.sum()
                equal_share = self.cash / len(buy_idx)
                cap = self.max_position_weight * nav_now
                # equal_share/cap은 스칼라라 그대로 np.minimum하면 스칼라가 나온다 —
                # 반드시 종목 수만큼의 배열로 만들어야 아래 alloc.sum()/인덱스별
                # 대입이 종목별로 올바르게 동작한다(실제로 이 버그로 캡 초과분이
                # 현금에서 제대로 빠지지 않는 걸 테스트로 잡아냄).
                alloc = np.minimum(np.full(len(buy_idx), equal_share), cap)
                half_cost = _half_round_trip_cost(self.panel.market_id[buy_idx])
                cost = half_cost * alloc

                self.cash -= float(alloc.sum())
                self.position_value[buy_idx] = alloc - cost
                self.last_close[buy_idx] = close_today[buy_idx]
                self.entry_close[buy_idx] = close_today[buy_idx]
                self.holding[buy_idx] = True
                self.masked_streak[buy_idx] = 0

        nav_new = max(self.cash + self.position_value.sum(), 1e-8)
        raw_reward = float(np.log(nav_new / self.nav))
        self.nav = nav_new  # 실제 NAV는 클리핑 없이 정확히 추적(평가/리포트는 이 값을 씀)
        reward = float(np.clip(raw_reward, -REWARD_CLIP, REWARD_CLIP))

        self.t += 1
        truncated = self.t > self.episode_end_idx
        terminated = False  # 이 환경엔 파산 등 별도의 "자연 종료" 조건이 없음(v1 스코프 밖)

        obs_t = min(self.t, self.episode_end_idx)
        obs = self._build_observation(obs_t)

        info = {
            "nav": nav_new,
            "cash": self.cash,
            "n_holding": int(self.holding.sum()),
            "raw_reward": raw_reward,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # 내부 헬퍼
    # ------------------------------------------------------------------ #

    def _close_positions(self, mask: np.ndarray) -> None:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        proceeds = self.position_value[idx]
        half_cost = _half_round_trip_cost(self.panel.market_id[idx])
        cost = half_cost * proceeds
        self.cash += float((proceeds - cost).sum())
        self.position_value[idx] = 0.0
        self.holding[idx] = False
        self.entry_close[idx] = np.nan
        self.masked_streak[idx] = 0

    def _build_observation(self, t: int) -> np.ndarray:
        static = self.obs_features[t]  # (N, D_static), 마지막 채널이 valid_mask

        nav = max(self.cash + self.position_value.sum(), 1e-8)
        holding_flag = self.holding.astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            unrealized_return = np.where(
                self.holding, self.last_close / self.entry_close - 1.0, 0.0
            ).astype(np.float32)
        position_weight = np.where(self.holding, self.position_value / nav, 0.0).astype(np.float32)

        dynamic = np.stack([holding_flag, unrealized_return, position_weight], axis=1)  # (N, 3)

        # CLAUDE.md "Phase 3 구체 사양"에 명시된 채널 순서: ... market_id/size_id +
        # holding_flag/unrealized_return/position_weight + valid_mask(마지막).
        # panel.py는 valid_mask를 정적 블록의 마지막 채널로 저장하므로, 동적
        # 3개를 그 앞에 끼워 넣어야 문서화된 순서와 정확히 일치한다.
        per_ticker = np.concatenate([static[:, :-1], dynamic, static[:, -1:]], axis=1)

        n_positions_frac = self.holding.sum() / self.n
        cash_weight = self.cash / nav
        nav_ratio = nav / self.initial_nav - 1.0
        elapsed = t - self.episode_start_idx
        total = max(self.episode_end_idx - self.episode_start_idx, 1)
        step_frac = min(elapsed / total, 1.0)

        portfolio_scalars = np.array(
            [cash_weight, nav_ratio, n_positions_frac, step_frac], dtype=np.float32
        )

        return np.concatenate([per_ticker.reshape(-1), portfolio_scalars]).astype(np.float32)
