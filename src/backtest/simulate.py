"""전략/랜덤워크/Buy&Hold 시뮬레이션. plan/05_backtesting.md 기준.

단순화 가정(명시): 예측=1(상승)인 날에만 매수해 익일 매도하고, 그 왕복에
수수료/슬리피지를 부과한다. 예측=0인 날은 현금 보유(수익률 0, 비용 없음).
연속으로 상승 예측이 이어져도 매일 재진입하는 것으로 취급해 보유하는 모든 날에
왕복비용을 부과한다 — 가장 보수적인 근사이며, 실제 포지션 유지 로직으로
정교화하는 것은 이 MVP 범위 밖이다.

셀 단위 포트폴리오 수익률은 그 날 신호가 존재하는 모든 종목의 순수익률을
동일가중 평균해 하루치 수익률로 잡고, 이를 누적곱해 누적수익률 곡선을 만든다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.costs import round_trip_cost_for_market


MAX_REALIZED_RETURN_GAP_DAYS = 10  # 이보다 넓게 떨어진 다음 행은 "익일"로 취급하지 않는다


def add_realized_return(df: pd.DataFrame) -> pd.DataFrame:
    """종목별 익일 실현수익률(Close_{t+1}/Close_t - 1)을 붙인다.

    거래정지 등으로 feature 단계에서 며칠~몇 주 치 행이 빠지면(vol_chg5가
    0/0=NaN이 되는 등 정상적인 drop), dataframe상 "바로 다음 행"이 실제로는
    몇 주 뒤 거래재개일일 수 있다. 이를 그대로 "익일 수익률"로 계산하면 몇 주치
    누적수익률에 하루치 매매비용만 적용하는 오류가 생긴다. 다음 행과의 날짜
    간격이 MAX_REALIZED_RETURN_GAP_DAYS를 넘으면 결측 처리한다 — 근거 없는
    "익일" 가정보다 결측이 낫다는 leakage-guard 원칙과 동일한 취지.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["realized_return"] = df.groupby("ticker")["close"].transform(lambda s: s.shift(-1) / s - 1)
    next_date_gap = df.groupby("ticker")["date"].transform(lambda s: (s.shift(-1) - s).dt.days)
    df.loc[next_date_gap > MAX_REALIZED_RETURN_GAP_DAYS, "realized_return"] = float("nan")
    return df


def add_net_return(df: pd.DataFrame, predicted_up: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    round_trip_cost = df["market_id"].map(round_trip_cost_for_market)
    df["net_return"] = np.where(predicted_up == 1, df["realized_return"] - round_trip_cost, 0.0)
    return df


def daily_portfolio_returns(df: pd.DataFrame, return_col: str = "net_return") -> pd.Series:
    """날짜별로 그 날 신호가 존재하는 종목들의 수익률을 동일가중 평균한다."""
    return df.groupby("date")[return_col].mean().sort_index()


def cumulative_return(daily_returns: pd.Series) -> pd.Series:
    return (1 + daily_returns).cumprod() - 1


def random_walk_total_returns(df: pd.DataFrame, n_sims: int = 1000, seed: int = 42) -> np.ndarray:
    """매일 50% 확률로 매수/현금을 무작위 선택하는 시뮬레이션을 n_sims회 반복,
    각 회차의 (동일가중 일별 포트폴리오) 누적수익률 최종값 분포를 반환한다."""
    rng = np.random.default_rng(seed)
    round_trip_cost = df["market_id"].map(round_trip_cost_for_market).to_numpy()
    realized = df["realized_return"].to_numpy()

    unique_dates, inverse = np.unique(df["date"].to_numpy(), return_inverse=True)
    n_dates = len(unique_dates)

    totals = np.empty(n_sims)
    for i in range(n_sims):
        choices = rng.random(len(df)) < 0.5
        net = np.where(choices, realized - round_trip_cost, 0.0)
        daily_sum = np.bincount(inverse, weights=net, minlength=n_dates)
        daily_count = np.bincount(inverse, minlength=n_dates)
        daily_mean = np.divide(daily_sum, daily_count, out=np.zeros(n_dates), where=daily_count > 0)
        totals[i] = np.prod(1 + daily_mean) - 1
    return totals


def buy_and_hold_return(df: pd.DataFrame) -> float:
    """참고 벤치마크: 종목별 단순 보유수익률(비용 미반영)의 동일가중 평균."""

    def _ret(g: pd.DataFrame) -> float:
        g = g.sort_values("date")
        return g["close"].iloc[-1] / g["close"].iloc[0] - 1

    per_ticker = df.groupby("ticker").apply(_ret, include_groups=False)
    return float(per_ticker.mean())
