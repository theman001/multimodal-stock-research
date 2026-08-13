"""통계적 유의성 검정: 방향성 이항검정(Bonferroni 보정) + 수익률 부트스트랩 CI.

plan/05_backtesting.md 기준: 6개 셀을 동시에 검정하므로 다중비교 보정이 필수.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

N_CELLS = 6
BONFERRONI_ALPHA = 0.05 / N_CELLS  # ≈ 0.0083


def binomial_test_direction(n_correct: int, n_total: int) -> dict:
    """방향성 적중률이 50%보다 유의하게 높은가 (단측 이항검정)."""
    if n_total == 0:
        return {
            "n_correct": 0,
            "n_total": 0,
            "accuracy": float("nan"),
            "p_value": float("nan"),
            "significant_bonferroni": False,
        }
    result = stats.binomtest(n_correct, n_total, p=0.5, alternative="greater")
    p_value = result.pvalue
    return {
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": n_correct / n_total,
        "p_value": p_value,
        "significant_bonferroni": p_value < BONFERRONI_ALPHA,
    }


def bootstrap_return_ci(daily_returns: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    """일별 포트폴리오 수익률을 리샘플링해 누적수익률의 95% 신뢰구간을 구한다."""
    if len(daily_returns) == 0:
        return {"point_estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    rng = np.random.default_rng(seed)
    n = len(daily_returns)
    totals = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(daily_returns, size=n, replace=True)
        totals[i] = np.prod(1 + sample) - 1
    return {
        "point_estimate": float(np.prod(1 + daily_returns) - 1),
        "ci_low": float(np.percentile(totals, 2.5)),
        "ci_high": float(np.percentile(totals, 97.5)),
    }
