import numpy as np
import pandas as pd
import pytest

from src.backtest.simulate import add_net_return, add_realized_return, cumulative_return, daily_portfolio_returns


def _make_df(ticker, dates, closes, market_id=0):
    return pd.DataFrame(
        {
            "ticker": [ticker] * len(dates),
            "date": pd.to_datetime(dates),
            "close": closes,
            "market_id": [market_id] * len(dates),
        }
    )


def test_realized_return_for_consecutive_days_is_normal_pct_change():
    df = _make_df("AAA", ["2020-01-01", "2020-01-02", "2020-01-03"], [100, 110, 90])
    df = add_realized_return(df)
    assert df["realized_return"].iloc[0] == pytest.approx(0.10)
    assert df["realized_return"].iloc[1] == pytest.approx((90 - 110) / 110)


def test_realized_return_is_nan_when_gap_exceeds_threshold():
    """거래정지 등으로 다음 행이 실제로는 몇 주 뒤라면 '익일 수익률'로 취급하면
    안 된다 — 결측 처리해야 한다."""
    df = _make_df("AAA", ["2020-01-01", "2020-02-15"], [100, 150])  # 45일 간격
    df = add_realized_return(df)
    assert pd.isna(df["realized_return"].iloc[0])


def test_realized_return_within_threshold_is_still_computed():
    df = _make_df("AAA", ["2020-01-01", "2020-01-08"], [100, 110])  # 7일 간격, 임계값(10일) 이내
    df = add_realized_return(df)
    assert not pd.isna(df["realized_return"].iloc[0])


def test_add_net_return_applies_cost_only_when_predicted_up():
    df = _make_df("AAA", ["2020-01-01", "2020-01-02"], [100, 110], market_id=0)
    df = add_realized_return(df)
    predicted_up = np.array([1, 0])
    df = add_net_return(df, predicted_up)
    # US 종목이므로 왕복비용 = 5bp = 0.0005
    assert df["net_return"].iloc[0] == pytest.approx(0.10 - 0.0005)


def test_daily_portfolio_and_cumulative_return_compound_correctly():
    daily = pd.Series([0.10, -0.10, 0.05], index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    cum = cumulative_return(daily)
    expected_final = (1.10 * 0.90 * 1.05) - 1
    assert cum.iloc[-1] == pytest.approx(expected_final)
