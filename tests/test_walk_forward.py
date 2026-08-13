import pandas as pd

from src.models.walk_forward import generate_walk_forward_folds


def _make_synthetic_features(n_days: int = 1200) -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"date": dates, "ticker": "AAA"})


def test_generates_requested_number_of_folds():
    features = _make_synthetic_features()
    folds = generate_walk_forward_folds(features, n_folds=5)
    assert len(folds) == 5


def test_each_fold_respects_embargo():
    features = _make_synthetic_features()
    folds = generate_walk_forward_folds(features, n_folds=5, embargo_days=60)
    for f in folds:
        assert (f.test_start - f.train_cutoff).days >= 60


def test_train_window_expands_across_folds():
    """뒤 폴드로 갈수록 학습 구간(train_cutoff)이 항상 더 늦은 날짜여야 한다."""
    features = _make_synthetic_features()
    folds = generate_walk_forward_folds(features, n_folds=5)
    cutoffs = [f.train_cutoff for f in folds]
    assert cutoffs == sorted(cutoffs)
    assert len(set(cutoffs)) == len(cutoffs)


def test_test_windows_do_not_overlap_across_folds():
    features = _make_synthetic_features()
    folds = generate_walk_forward_folds(features, n_folds=5)
    for a, b in zip(folds, folds[1:]):
        assert a.test_end < b.test_start


def test_test_window_is_strictly_after_its_own_train_cutoff():
    features = _make_synthetic_features()
    folds = generate_walk_forward_folds(features, n_folds=5)
    for f in folds:
        assert f.test_start > f.train_cutoff
        assert f.test_end >= f.test_start
