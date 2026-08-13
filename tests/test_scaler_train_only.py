import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.indicators import FEATURE_COLUMNS
from src.models.scaling import fit_scaler, transform

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_SRC_DIR = REPO_ROOT / "src" / "models"


def _make_synthetic_features(n: int, seed: int, offset: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(offset, 1.0, size=n) for col in FEATURE_COLUMNS}
    return pd.DataFrame(data)


def test_scaler_fit_on_train_only_differs_from_fit_on_everything():
    """Train만으로 fit한 스케일러와 Train+Test 전체로 fit한 스케일러의 파라미터는
    달라야 한다 — 같다면 Test 데이터 통계가 fit에 섞여들어갔다는 뜻(누수)."""
    train = _make_synthetic_features(150, seed=1, offset=0.0)
    test = _make_synthetic_features(50, seed=2, offset=5.0)  # 분포를 의도적으로 이동

    scaler_train_only = fit_scaler(train)
    scaler_fit_on_everything = fit_scaler(pd.concat([train, test], ignore_index=True))

    assert not np.allclose(scaler_train_only.mean_, scaler_fit_on_everything.mean_)


def test_scaler_fit_on_train_reproduces_train_mean_zero_after_transform():
    train = _make_synthetic_features(200, seed=3, offset=2.0)
    scaler = fit_scaler(train)
    transformed = transform(train, scaler)
    assert np.allclose(transformed[FEATURE_COLUMNS].mean(), 0.0, atol=1e-8)


def test_test_transform_uses_train_statistics_not_its_own():
    """Test를 transform한 결과의 평균은 (Test 분포가 Train과 다르면) 0이 아니어야
    한다 — 0이면 Test 자체 통계로 다시 정규화됐다는 뜻(fit_transform 오용 의심)."""
    train = _make_synthetic_features(150, seed=1, offset=0.0)
    test = _make_synthetic_features(50, seed=2, offset=5.0)

    scaler = fit_scaler(train)
    test_transformed = transform(test, scaler)

    assert not np.allclose(test_transformed[FEATURE_COLUMNS].mean(), 0.0, atol=0.5)


def test_no_fit_transform_used_in_models_source():
    """Test/Val 쪽에서 fit_transform을 잘못 호출하는 패턴을 원천 차단한다.

    주석/docstring에서 이 금지 규칙 자체를 설명하는 문구까지 오탐지하지 않도록,
    실제 `.fit_transform(` 메서드 호출 구문만 찾는다.
    """
    for path in MODELS_SRC_DIR.glob("*.py"):
        assert ".fit_transform(" not in path.read_text(encoding="utf-8"), (
            f"{path.name}에서 .fit_transform( 호출 사용됨 — fit은 Train에만, transform은 별도 호출해야 함"
        )


def test_fit_is_only_called_with_train_argument_in_build_split():
    """build_split.py에서 fit_scaler가 test 변수로 호출되지 않는지 확인."""
    content = (MODELS_SRC_DIR / "build_split.py").read_text(encoding="utf-8")
    calls = re.findall(r"fit_scaler\(([^)]*)\)", content)
    assert calls, "fit_scaler 호출을 찾지 못함"
    for call_arg in calls:
        assert "test" not in call_arg, f"fit_scaler가 test 관련 인자로 호출됨: {call_arg}"
