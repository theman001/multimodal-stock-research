import numpy as np
import pytest

from src.backtest.significance import paired_bootstrap_return_diff_ci


def test_point_estimate_matches_direct_cumulative_return_difference():
    rng = np.random.default_rng(0)
    returns_a = rng.normal(0.001, 0.01, size=100)
    returns_b = rng.normal(0.0005, 0.01, size=100)

    result = paired_bootstrap_return_diff_ci(returns_a, returns_b)

    expected = (np.prod(1 + returns_a) - 1) - (np.prod(1 + returns_b) - 1)
    assert result["point_estimate"] == pytest.approx(expected)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        paired_bootstrap_return_diff_ci(np.zeros(10), np.zeros(9))


def test_empty_input_returns_nan():
    result = paired_bootstrap_return_diff_ci(np.array([]), np.array([]))
    assert np.isnan(result["point_estimate"])
    assert np.isnan(result["ci_low"])
    assert np.isnan(result["ci_high"])


def test_constant_daily_edge_gives_a_tight_confidence_interval():
    """A가 매일 정확히 B보다 0.01만큼 높은 수익률이면(변동성 없는 우위),
    리샘플링해도 매번 같은 차이가 나와야 하므로 CI가 거의 한 점에 몰려야 한다."""
    returns_b = np.full(50, 0.0)
    returns_a = np.full(50, 0.01)

    result = paired_bootstrap_return_diff_ci(returns_a, returns_b)

    assert result["point_estimate"] > 0
    assert result["ci_high"] - result["ci_low"] < 1e-6


def test_consistent_edge_with_shared_noise_yields_confidently_positive_ci():
    """A와 B가 공통 노이즈를 공유하고(같은 시장 국면에 노출) A가 매일 일정
    마진만큼 앞서면 — 짝짓기가 그 공통 노이즈를 상쇄해줘야 하는 전형적 상황 —
    신뢰구간이 0을 포함하지 않고 뚜렷하게 양수 쪽에 있어야 한다(통계적으로
    "A가 유의하게 낫다"고 결론 내릴 수 있는 상태)."""
    rng = np.random.default_rng(1)
    common = rng.normal(0, 0.02, size=200)
    returns_a = common + 0.002
    returns_b = common

    result = paired_bootstrap_return_diff_ci(returns_a, returns_b, n_boot=2000)

    assert result["point_estimate"] > 0
    assert result["ci_low"] > 0
