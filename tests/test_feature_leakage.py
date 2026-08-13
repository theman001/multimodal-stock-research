import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import FEATURE_COLUMNS, compute_features

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_SRC_DIR = REPO_ROOT / "src" / "features"


def _make_synthetic_ohlcv(n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    close = np.maximum(100 + np.cumsum(rng.normal(0, 1, size=n_days)), 1.0)
    high = close + rng.uniform(0, 1, size=n_days)
    low = close - rng.uniform(0, 1, size=n_days)
    open_ = close + rng.uniform(-0.5, 0.5, size=n_days)
    volume = rng.integers(1000, 10000, size=n_days)
    return pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def test_feature_at_t_matches_when_recomputed_on_data_truncated_to_t():
    """t 시점의 피처는 t까지만 자른 데이터로 재계산해도 동일해야 한다 (룩어헤드 없음)."""
    full = _make_synthetic_ohlcv(120)
    full_features = compute_features(full)

    t_index = 90  # 최대 롤링 윈도우(MA60) 충족 이후 시점
    truncated = full.iloc[: t_index + 1]
    truncated_features = compute_features(truncated)

    full_row = full_features.iloc[t_index]
    truncated_row = truncated_features.iloc[t_index]

    for col in FEATURE_COLUMNS:
        assert full_row[col] == pytest.approx(truncated_row[col], nan_ok=True), (
            f"{col} 값이 미래 데이터 유무에 따라 달라짐 — 룩어헤드 의심"
        )


def test_feature_at_early_t_with_insufficient_window_is_nan_in_both():
    """롤링 윈도우가 아직 안 찬 초기 구간은 전체/절단 계산 모두 NaN이어야 한다."""
    full = _make_synthetic_ohlcv(120)
    full_features = compute_features(full)

    t_index = 10  # MA60은커녕 MA20도 아직 안 찬 시점
    truncated = full.iloc[: t_index + 1]
    truncated_features = compute_features(truncated)

    full_row = full_features.iloc[t_index]
    truncated_row = truncated_features.iloc[t_index]

    assert pd.isna(full_row["ma60_ratio"])
    assert pd.isna(truncated_row["ma60_ratio"])


def test_last_row_per_ticker_has_null_target():
    """target은 shift(-1)로 계산되므로 마지막 거래일은 target이 없어야 한다."""
    df = _make_synthetic_ohlcv(30)
    features = compute_features(df)
    assert pd.isna(features["target"].iloc[-1])
    assert features["target"].iloc[:-1].notna().all()


def test_no_negative_shift_outside_target_calculation():
    """shift(-N)은 target 계산에만 허용된다 — 피처 계산 코드에 있으면 누수 의심.

    변수를 여러 줄로 나눠 쓸 수 있으므로 같은 줄이 아니라 주변 문맥(앞뒤 몇 줄)에
    "target"이 있는지로 판단한다.
    """
    context_window = 4
    for path in FEATURES_SRC_DIR.glob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            if re.search(r"shift\(\s*-", line):
                context = "\n".join(lines[max(0, lineno - 1 - context_window) : lineno - 1 + context_window])
                assert "target" in context, (
                    f"{path.name}:{lineno} 에서 미래 참조(shift(-N))가 target 계산 문맥 밖에서 사용됨: {line.strip()}"
                )


def test_no_center_true_in_rolling_windows():
    """rolling(..., center=True)는 미래 데이터를 포함하므로 절대 금지.

    docstring/주석에서 이 규칙을 설명하는 문구까지 오탐지하지 않도록, 실제
    `rolling(...)` 호출 구문 안에서만 `center=True`를 찾는다.
    """
    pattern = re.compile(r"rolling\([^)]*center\s*=\s*True")
    for path in FEATURES_SRC_DIR.glob("*.py"):
        match = pattern.search(path.read_text(encoding="utf-8"))
        assert match is None, f"{path.name}에서 rolling(center=True) 사용됨: {match}"
