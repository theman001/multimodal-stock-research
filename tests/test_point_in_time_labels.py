import pandas as pd

from src.data_collection.kr_universe import label_point_in_time


def test_size_id_changes_across_rebalance_periods():
    """정기변경 기간 경계를 넘어가면 같은 종목도 size_id가 바뀔 수 있어야 한다."""
    ohlcv = pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-01-05", "2015-07-01", "2016-01-05"]),
            "close": [100, 110, 120],
        }
    )
    rankings = pd.DataFrame(
        {
            "ticker": ["005930", "005930", "005930"],
            "effective_date": pd.to_datetime(["2014-12-12", "2015-06-13", "2015-12-12"]),
            "size_id": [1, 2, 0],
            "market_cap": [1.0, 2.0, 3.0],
            "rank": [150, 50, 400],
        }
    )

    labeled = label_point_in_time(ohlcv, "005930", rankings)

    assert labeled.set_index("date")["size_id"].to_dict() == {
        pd.Timestamp("2015-01-05"): 1,
        pd.Timestamp("2015-07-01"): 2,
        pd.Timestamp("2016-01-05"): 0,
    }


def test_rows_before_first_effective_date_are_dropped_not_mislabeled():
    """정기변경 데이터가 없는(상장 전 등) 시점의 행은 라벨 없이 버려야 한다 — 추정 라벨 금지."""
    ohlcv = pd.DataFrame(
        {
            "date": pd.to_datetime(["2014-01-01", "2015-01-05"]),
            "close": [100, 110],
        }
    )
    rankings = pd.DataFrame(
        {
            "ticker": ["005930"],
            "effective_date": pd.to_datetime(["2014-12-12"]),
            "size_id": [2],
            "market_cap": [1.0],
            "rank": [1],
        }
    )

    labeled = label_point_in_time(ohlcv, "005930", rankings)

    assert len(labeled) == 1
    assert labeled.iloc[0]["date"] == pd.Timestamp("2015-01-05")


def test_unknown_ticker_returns_empty_instead_of_guessing():
    ohlcv = pd.DataFrame({"date": pd.to_datetime(["2015-01-05"]), "close": [100]})
    rankings = pd.DataFrame(columns=["ticker", "effective_date", "size_id", "market_cap", "rank"])

    labeled = label_point_in_time(ohlcv, "999999", rankings)

    assert labeled.empty


def test_size_id_stays_constant_within_a_single_period():
    ohlcv = pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-06-15", "2015-08-01", "2015-12-01"]),
            "close": [100, 105, 108],
        }
    )
    rankings = pd.DataFrame(
        {
            "ticker": ["005930"],
            "effective_date": pd.to_datetime(["2015-06-13"]),
            "size_id": [2],
            "market_cap": [1.0],
            "rank": [1],
        }
    )

    labeled = label_point_in_time(ohlcv, "005930", rankings)

    assert (labeled["size_id"] == 2).all()
