"""Phase 4 공유 — 오늘자 단일 시점 관측벡터 생성.

`src/rl/panel.py::build_grid()`(배치 경로, 전체 날짜 그리드)를 재사용하되
"오늘 하루"로 좁힌 경량 경로다. 그대로 오늘 날짜 행만 잘라서 `build_grid()`에
넘기면 `_ticker_market_ids()`(panel.py)가 그날 row가 없는 티커의 market_id를
못 찾아 깨진다(market_id는 티커별 고정값이라 전체 이력에서 조회하도록 설계돼
있음, panel.py 180-191행 참고) — 그래서 `target_date` 하루만이 아니라
`lookback_days`만큼의 최근 윈도우를 `build_grid()`에 넘겨 모든 티커의
market_id/size_id를 안정적으로 resolve하고, 그 결과 패널에서 `target_date`
슬라이스 한 개만 뽑아 쓴다. 윈도우 안에서도 특정 (date,ticker) row가 없으면
`build_grid()`가 이미 하는 대로 0-채움+valid_mask=0으로 마스킹된다(leakage-guard
불변식 그대로 유지).

동적 3필드(holding_flag/unrealized_return/position_weight)는
`TradingEnv._build_observation()`(trading_env.py 262-292행)과 완전히 동일한
채널 순서로 이어붙이지만, 소스가 다르다 — 배치는 env 내부 시뮬레이션 장부에서
읽고, 라이브는 브로커 계좌 조회 결과(`broker_positions`/`broker_cash`)에서
읽는다(plan/10 §A: "포지션 상태의 진실 소스가 시뮬레이션에서 브로커 계좌로
바뀌는 지점").

`step_frac`/`nav_ratio`는 252일 에피소드 학습을 전제로 정의된 채널인데
라이브엔 "에피소드"가 없다(plan/10 §D, 미해결로 명시된 항목) — 여기서는
`step_frac=0.0` 고정, `nav_ratio`는 호출자가 넘기는 `nav_anchor`(배포 시점
기준 NAV) 대비 변화율로 잠정 계산한다. 이 잠정 규칙이 정책 행동에 실제로
얼마나 영향을 주는지는 모의투자 로그로 사후 검증한다(§D).

`HeldPosition.qty`는 음수(숏)를 허용한다(2026-08-20 "롱 베이스, 숏은 전략적
허용" 결정) — 다만 지금은 관측 계층 한정이다. `rl_policy_v1.zip`은 여전히
SELL/HOLD/BUY 3지 선택뿐이라 스스로 숏에 진입하지 않는다(TradingEnv/allocation.py
모두 SELL=롱 포지션 전량 청산으로만 처리, 숏 오픈 경로 없음). 이건 브로커
계좌에 이미 존재할 수 있는 숏 포지션(수동 개입 등)을 관측벡터에 부호까지
정확히 반영해 크래시 없이 흡수하기 위한 대비다 — 정책이 능동적으로 숏을 여는
기능은 액션 공간 자체를 바꿔야 하는 Phase 3 재학습 범위라 이 세션(순수 서빙)
밖에 있다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.rl.obs_scaler import load_obs_scaler, transform_obs_features
from src.rl.panel import MODEL_V3_PROB_COLUMN, build_grid, load_ticker_universe, score_model_v3_probabilities

DEFAULT_LOOKBACK_DAYS = 30  # 모든 티커가 이 윈도우 안에서 최소 1행은 갖는다는 가정(휴장 클러스터 감안 여유)


@dataclass
class HeldPosition:
    """브로커가 보고한 보유 포지션 스냅샷 — BrokerAdapter.get_positions()가
    이 형태(ticker -> HeldPosition)로 반환한다고 가정한다(src/live/broker/*.py,
    다음 단계에서 구현). qty<0은 숏 포지션(모듈 docstring 참고) — 롱/숏 모두
    avg_entry_price는 항상 양수(진입 시점 체결가)."""

    qty: float
    avg_entry_price: float


@dataclass
class TodayObservation:
    """build_today_observation()의 반환값. `obs`가 정책 입력 벡터 본체이고,
    나머지는 `src/live/allocation.py::compute_target_allocations()`가 같은
    스텝에서 재사용할 부가 정보다(다시 계산하지 않도록)."""

    obs: np.ndarray  # (N * D_per_ticker + 4,) — TradingEnv.observation_space와 동일 shape
    tickers: list[str]
    valid_mask: np.ndarray  # (N,) 오늘 row 존재 여부
    has_price: np.ndarray  # (N,) 오늘 종가 존재 여부(valid_mask와 별도 — panel.py build_grid는 두 소스를 독립적으로 마스킹)
    close: np.ndarray  # (N,) 오늘 종가(없으면 NaN — 마스킹된 보유 포지션의 실제 평가액은 close가 아니라 position_value를 쓸 것)
    position_value: np.ndarray  # (N,) 보유 중인 티커의 시가평가액(마스킹 시 마지막 유효가 carry-forward 반영, 미보유는 0)
    market_id: np.ndarray  # (N,) 티커별 고정 시장 코드
    cash: float
    nav: float


def build_today_observation(
    data_root: Path,
    checkpoints_dir: Path,
    checkpoint_name: str,
    target_date: pd.Timestamp | str,
    broker_positions: dict[str, HeldPosition],
    broker_cash: float,
    nav_anchor: float,
    include_event_features: bool,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> TodayObservation:
    """target_date 하루치 관측벡터를 만든다. 그날 features.parquet/ohlcv_meta.parquet
    행이 갱신돼 있어야 한다(daily_pipeline.py가 그 전에 실행돼야 함, 다음 단계).

    nav_anchor: 배포 시점(또는 마지막으로 리베이스한 시점) 기준 NAV. nav_ratio
    계산의 잠정 앵커로만 쓰인다(모듈 docstring 참고).
    """
    if lookback_days <= 0:
        # lookback_days>0이 이 함수 전체 설계의 전제다(모듈 docstring 1문단) —
        # 0 이하를 넘기면 그냥 "오늘 하루만" 넘기는 것과 같아져서, 이 함수가
        # 원래 피하려 했던 market_id resolve 실패 버그가 그대로 재현된다
        # (재검토로 발견, 현재 호출자가 없어 아직 실사용된 적은 없음).
        raise ValueError(f"lookback_days는 1 이상이어야 함: {lookback_days!r}")
    if not np.isfinite(broker_cash):
        # broker_cash는 브로커 API 응답에서 그대로 온 외부 입력이다 — NaN/Inf가
        # 여기서 걸러지지 않으면 nav 전체가 오염되고(nav = cash + ...), 그 nav로
        # 나누는 cash_weight/nav_ratio/position_weight까지 120종목 전체가 NaN으로
        # 물들어 정책 추론 자체가 무너진다(재검토로 발견) — 개별 포지션 하나가
        # 아니라 관측벡터 전체가 오염되는 훨씬 넓은 반경의 실패라 여기서 먼저 막는다.
        raise ValueError(f"broker_cash가 유한하지 않음: {broker_cash!r}")
    if not (np.isfinite(nav_anchor) and nav_anchor > 0):
        # nav_anchor는 broker_cash처럼 브로커 API 응답은 아니지만(호출자가 상태
        # 파일 등에서 읽어오는 값), 손상된 상태 파일이면 똑같이 NaN/Inf/음수가
        # 들어올 수 있다. nav_ratio = nav/max(nav_anchor,1e-8)-1.0 채널 하나만
        # 오염되는 것처럼 보이지만, 신경망은 입력을 층마다 섞으므로 NaN 채널
        # 하나가 전체 행동 예측을 오염시킬 수 있다(재검토로 발견) — broker_cash와
        # 동일한 이유로 여기서 막는다.
        raise ValueError(f"nav_anchor가 유효하지 않음(유한한 양수여야 함): {nav_anchor!r}")
    for ticker, pos in broker_positions.items():
        if not np.isfinite(pos.qty):
            raise ValueError(f"{ticker}의 브로커 포지션 수량이 유한하지 않음: {pos!r}")
        # avg_entry_price는 qty==0(잔여 청산 항목 등)이면 아래 루프에서 애초에
        # 안 쓰인다(continue로 건너뜀) — qty!=0일 때만 검증한다. 그렇지 않으면
        # 브로커가 완전 청산된 항목에 의미 없는 avg_entry_price(NaN 등)를
        # 돌려주는 흔한 경우까지 걸려 불필요하게 막힌다(재검토로 발견, 바로
        # 위 unknown_tickers의 qty==0 예외와 동일한 성격).
        if pos.qty != 0 and not np.isfinite(pos.avg_entry_price):
            raise ValueError(f"{ticker}의 브로커 포지션 평단가가 유한하지 않음: {pos!r}")
        # qty<0(숏 포지션)은 에러가 아니다 — "롱 베이스, 숏은 전략적으로 허용"
        # 방향으로 확정(2026-08-20 사용자 결정). 지금은 관측 계층만: RL 정책
        # (rl_policy_v1.zip)은 여전히 SELL/HOLD/BUY 3지 선택뿐이라 스스로 숏을
        # 열지 않는다 — 이건 브로커 계좌에 이미 존재하는 숏 포지션(수동 개입 등)을
        # 관측벡터에 부호까지 정확히 반영하기 위한 대비다. 정책이 능동적으로
        # 숏에 진입하는 기능은 액션 공간 자체를 바꿔야 하는 Phase 3 재학습
        # 범위라 이번 세션(순수 서빙) 밖에 있음.

    target_date = pd.Timestamp(target_date)
    if target_date.tzinfo is not None:
        # features.parquet의 date는 tz-naive(ohlcv.py가 tz_localize(None)으로 저장) —
        # 호출자가 tz-aware Timestamp를 넘기면(예: KST cron이 datetime.now(tz=...)를
        # 그대로 넘기는 경우) 비교가 조용히 어긋나거나 UTC 변환으로 날짜가 하루
        # 밀릴 수 있어 여기서 먼저 벗겨낸다.
        target_date = target_date.tz_localize(None)
    target_date = target_date.normalize()
    window_start = target_date - pd.Timedelta(days=lookback_days)

    features_df = pd.read_parquet(data_root / "processed" / "features.parquet")
    window_df = features_df[
        (features_df["date"] >= window_start) & (features_df["date"] <= target_date)
    ].copy()
    if window_df.empty or target_date not in set(window_df["date"]):
        raise ValueError(
            f"{target_date.date()} 기준 features.parquet에 오늘자 행이 없음 — "
            "daily_pipeline.py의 증분 수집이 먼저 실행됐는지 확인할 것."
        )
    window_df[MODEL_V3_PROB_COLUMN] = score_model_v3_probabilities(window_df, data_root)

    ohlcv_df = pd.read_parquet(data_root / "processed" / "ohlcv_meta.parquet")[["date", "ticker", "close"]]
    ohlcv_window = ohlcv_df[(ohlcv_df["date"] >= window_start) & (ohlcv_df["date"] <= target_date)]

    tickers = load_ticker_universe(data_root)
    # qty==0인 포지션은 NAV에 영향이 없으므로 유니버스 밖이어도 무해하다 —
    # 브로커가 완전 청산 직후에도 한동안 잔여 qty=0 항목을 돌려주는 경우가
    # 흔한데, 그런 항목까지 걸리면 매일 불필요하게 막힌다(재검토로 발견).
    unknown_tickers = {t for t, pos in broker_positions.items() if pos.qty != 0} - set(tickers)
    if unknown_tickers:
        # broker_positions를 그대로 순회하지 않고 tickers 순서로 조회하므로(아래
        # 루프), 유니버스 밖 티커는 지금 조용히 무시되고 그 포지션의 실제 가치는
        # NAV 계산에서 통째로 빠진다 — 수동 거래/스핀오프 등으로 실제 발생 가능한
        # 상황이라 조용히 넘어가지 않고 여기서 막는다(구현 중 리뷰로 발견).
        raise ValueError(
            f"브로커 포지션에 RL 유니버스 밖 티커가 있음: {sorted(unknown_tickers)} — "
            "관측벡터에 반영할 수 없어 NAV가 조용히 틀어진다."
        )
    panel = build_grid(window_df, ohlcv_window, tickers, include_event_features=include_event_features)

    target_np = target_date.to_datetime64()
    if target_np not in panel.dates:
        raise ValueError(f"{target_date.date()}가 패널 날짜 그리드에 없음")
    t = int(np.searchsorted(panel.dates, target_np))

    scaler = load_obs_scaler(checkpoints_dir / f"{checkpoint_name}_scaler.pkl")
    scaled = transform_obs_features(panel, scaler)
    static_today = scaled[t]  # (N, D_static), 마지막 채널이 valid_mask

    n = len(tickers)
    close_today = panel.close[t].astype(np.float64)
    has_price = ~np.isnan(close_today)
    valid_today = panel.valid_mask[t].astype(bool)

    # 보유 중인데 오늘 가격이 마스킹된 티커는 TradingEnv의 mark-to-market과 동일하게
    # "마지막으로 알려진 가격"을 그대로 들고 간다(trading_env.py 182-183행 still_masked
    # 처리와 동일 취지) — 0으로 떨어뜨리면 그 포지션의 실제 가치가 NAV/position_weight
    # 계산에서 통째로 사라져 "보유 중인데 비중 0%"라는 내부 모순 상태가 되고 NAV도
    # 그만큼 과소평가된다(구현 중 리뷰로 발견한 버그).
    close_history = panel.close[: t + 1].astype(np.float64)  # (t+1, N)
    with np.errstate(all="ignore"):
        last_known_close = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            col = close_history[:, i]
            valid_vals = col[~np.isnan(col)]
            if len(valid_vals) > 0:
                last_known_close[i] = valid_vals[-1]

    holding_flag = np.zeros(n, dtype=np.float32)
    unrealized_return = np.zeros(n, dtype=np.float32)
    position_value = np.zeros(n, dtype=np.float64)

    for i, ticker in enumerate(tickers):
        pos = broker_positions.get(ticker)
        if pos is None or pos.qty == 0:
            continue
        holding_flag[i] = 1.0

        effective_price = close_today[i] if has_price[i] else last_known_close[i]
        if np.isnan(effective_price):
            # lookback_days 안에 이 티커 가격이 전혀 없는 극단적 데이터 공백 —
            # 마지막 수단으로 평단가를 써서(평가손익 0) 최소한 포지션 원가만큼은
            # NAV에 반영되게 한다(0으로 두어 포지션 전체가 사라지는 것보다 안전).
            effective_price = pos.avg_entry_price

        if pos.avg_entry_price > 0:
            price_return = effective_price / pos.avg_entry_price - 1.0
            # 롱(qty>0)은 가격이 오르면 이익이라 price_return을 그대로 쓰면 되지만,
            # 숏(qty<0)은 정반대로 가격이 내려야 이익이다 — 부호를 그대로 두면
            # 숏 포지션의 손실을 이익으로 잘못 표시하게 된다(qty의 부호를 곱해
            # 방향을 맞춘다: 숏이면 price_return의 부호를 뒤집음).
            unrealized_return[i] = np.float32(np.sign(pos.qty) * price_return)
        # avg_entry_price<=0은 비정상 데이터 — division by zero/NaN이 정책 입력에
        # 그대로 흘러들어가지 않도록 unrealized_return을 0으로 둔다(가드).
        position_value[i] = pos.qty * effective_price

    nav = max(broker_cash + float(position_value.sum()), 1e-8)
    position_weight = np.where(holding_flag.astype(bool), position_value / nav, 0.0).astype(np.float32)

    dynamic = np.stack([holding_flag, unrealized_return, position_weight], axis=1)  # (N, 3)

    # trading_env.py의 채널 순서 그대로: 정적 블록(valid_mask 제외) + 동적 3개 + valid_mask(마지막)
    per_ticker = np.concatenate([static_today[:, :-1], dynamic, static_today[:, -1:]], axis=1)

    n_positions_frac = float(holding_flag.sum()) / n
    cash_weight = broker_cash / nav
    nav_ratio = nav / max(nav_anchor, 1e-8) - 1.0
    step_frac = 0.0  # 라이브엔 에피소드 개념이 없음(모듈 docstring 참고)

    portfolio_scalars = np.array(
        [cash_weight, nav_ratio, n_positions_frac, step_frac], dtype=np.float32
    )

    obs = np.concatenate([per_ticker.reshape(-1), portfolio_scalars]).astype(np.float32)

    if not np.isfinite(obs).all():
        # 지금까지 입력(broker_cash/nav_anchor/포지션)은 전부 검증했지만, 이건
        # 그 검증들을 우회하는 다른 경로(재사용 중인 transform_obs_features의
        # 극단치, 향후 코드 변경 등)로 새어드는 NaN/Inf까지 잡는 마지막
        # 방어선이다 — infer.py도 독립적으로 obs를 검증하지만(재검토 발견),
        # 여기서 먼저 잡아야 오염이 어디서 시작됐는지(observation.py 자체)를
        # 바로 알 수 있다(재검토로 발견).
        raise ValueError(
            "생성된 관측벡터에 유한하지 않은 값(NaN/Inf)이 있음 — 입력 검증을 "
            "모두 통과했는데도 발생했다면 model_v3/obs_scaler 쪽 이상치를 의심할 것."
        )

    return TodayObservation(
        obs=obs,
        tickers=tickers,
        valid_mask=valid_today,
        has_price=has_price,
        close=close_today.astype(np.float32),
        position_value=position_value.astype(np.float32),
        market_id=panel.market_id,
        cash=broker_cash,
        nav=nav,
    )
