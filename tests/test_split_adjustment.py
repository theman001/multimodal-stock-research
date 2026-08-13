import pandas as pd

from src.data_collection.kr_universe import detect_and_adjust_splits


def _make_df(ticker, dates, closes, market_caps, volumes=None):
    n = len(dates)
    volumes = volumes or [1000] * n
    return pd.DataFrame(
        {
            "ticker": [ticker] * n,
            "date": pd.to_datetime(dates),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
            "market_cap": market_caps,
        }
    )


def test_forward_split_is_detected_and_backward_adjusted():
    """2:1 분할(가격 반토막, 시가총액은 유지)이면 분할 이전 가격을 절반으로 조정해야 한다."""
    df = _make_df(
        "AAA",
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
        closes=[100, 101, 102, 51, 52],
        market_caps=[1000, 1010, 1020, 1020, 1040],  # 분할일(1020->1020) 시가총액 변화 없음
        volumes=[500, 500, 500, 1000, 1000],
    )
    adjusted = detect_and_adjust_splits(df)

    assert adjusted["close"].tolist() == [50.0, 50.5, 51.0, 51, 52]
    assert adjusted["volume"].tolist() == [1000.0, 1000.0, 1000.0, 1000, 1000]


def test_genuine_large_move_is_not_treated_as_split():
    """가격도 크게 움직이고 시가총액도 같이 크게 움직이면(진짜 시장 변동) 조정하면 안 된다."""
    df = _make_df(
        "BBB",
        ["2020-01-01", "2020-01-02", "2020-01-03"],
        closes=[100, 101, 40],
        market_caps=[1000, 1010, 400],  # 가격과 시가총액이 같이 -60% -> 진짜 폭락
    )
    adjusted = detect_and_adjust_splits(df)

    assert adjusted["close"].tolist() == [100, 101, 40]


def test_multiple_splits_apply_cumulatively():
    """한 종목에 분할이 두 번 있으면 각각의 비율이 누적으로 과거 가격에 적용돼야 한다."""
    df = _make_df(
        "CCC",
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
        closes=[400, 200, 100, 100],  # 1/2 -> 1/2 두 번의 분할 (실질 1/4)
        market_caps=[1000, 1000, 1000, 1010],
    )
    adjusted = detect_and_adjust_splits(df)

    assert adjusted["close"].tolist() == [100.0, 100.0, 100, 100]


def test_market_cap_column_is_never_adjusted():
    df = _make_df(
        "AAA",
        ["2020-01-01", "2020-01-02"],
        closes=[100, 50],
        market_caps=[1000, 1000],
    )
    adjusted = detect_and_adjust_splits(df)

    assert adjusted["market_cap"].tolist() == [1000, 1000]


def test_multiple_tickers_are_independent():
    """한 종목의 분할이 다른 종목 가격에 영향을 주면 안 된다."""
    df = pd.concat(
        [
            _make_df("AAA", ["2020-01-01", "2020-01-02"], closes=[100, 50], market_caps=[1000, 1000]),
            _make_df("BBB", ["2020-01-01", "2020-01-02"], closes=[200, 205], market_caps=[2000, 2050]),
        ],
        ignore_index=True,
    )
    adjusted = detect_and_adjust_splits(df)

    bbb = adjusted[adjusted["ticker"] == "BBB"]
    assert bbb["close"].tolist() == [200, 205]
