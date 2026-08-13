import pandas as pd

from src.models.split import EMBARGO_DAYS, split_train_test


def _make_synthetic_features(n_days: int = 500) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"date": dates, "ticker": "AAA", "close": range(n_days)})


def test_embargo_gap_is_at_least_60_calendar_days():
    features = _make_synthetic_features()
    _, _, train_cutoff, test_start = split_train_test(features)
    gap_days = (test_start - train_cutoff).days
    assert gap_days >= EMBARGO_DAYS


def test_train_and_test_dates_do_not_overlap():
    features = _make_synthetic_features()
    train, test, _, _ = split_train_test(features)
    assert train["date"].max() < test["date"].min()


def test_split_ratio_is_approximately_80_20_of_unique_dates():
    features = _make_synthetic_features(n_days=1000)
    train, test, _, _ = split_train_test(features)
    train_dates = train["date"].nunique()
    total_dates = features["date"].nunique()
    assert 0.75 <= train_dates / total_dates <= 0.85


def test_multiple_tickers_share_the_same_global_cutoff():
    """종목별이 아니라 전역 날짜 기준으로 잘라야 한다 — 같은 날짜에 어떤 종목은
    Train, 어떤 종목은 Test가 되면 안 된다."""
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    features = pd.concat(
        [
            pd.DataFrame({"date": dates, "ticker": "AAA"}),
            pd.DataFrame({"date": dates, "ticker": "BBB"}),
        ],
        ignore_index=True,
    )
    train, test, train_cutoff, test_start = split_train_test(features)

    for ticker in ("AAA", "BBB"):
        ticker_train_max = train[train["ticker"] == ticker]["date"].max()
        ticker_test_min = test[test["ticker"] == ticker]["date"].min()
        assert ticker_train_max == train_cutoff
        assert ticker_test_min == test_start
