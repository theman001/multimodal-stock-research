"""fetch_us_ohlcv()의 증분 수집 회귀 테스트 — 실제 yfinance 네트워크 호출 금지(mock)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_collection.ohlcv import fetch_us_ohlcv


def _yf_history_df(dates: list[str]) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100},
        index=pd.DatetimeIndex(idx, name="Date"),
    )


def _mock_ticker(history_df: pd.DataFrame) -> MagicMock:
    ticker_obj = MagicMock()
    ticker_obj.history.return_value = history_df
    return ticker_obj


def test_no_cache_fetches_full_range(tmp_path):
    with patch("src.data_collection.ohlcv.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.return_value = _mock_ticker(_yf_history_df(["2020-01-01", "2020-01-02"]))
        result = fetch_us_ohlcv("AAPL", "2020-01-01", "2020-01-03", tmp_path)

    assert mock_ticker_cls.call_args.args == ("AAPL",)
    _, kwargs = mock_ticker_cls.return_value.history.call_args
    assert kwargs["start"] == "2020-01-01"
    assert kwargs["end"] == "2020-01-03"
    assert len(result) == 2
    assert (tmp_path / "AAPL.parquet").exists()


def test_incremental_fetches_only_after_cached_max_date(tmp_path):
    cache_path = tmp_path / "AAPL.parquet"
    cached = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [100, 100],
        }
    )
    cached.to_parquet(cache_path, index=False)

    with patch("src.data_collection.ohlcv.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.return_value = _mock_ticker(_yf_history_df(["2020-01-03"]))
        result = fetch_us_ohlcv("AAPL", "2020-01-01", "2020-01-04", tmp_path)

    _, kwargs = mock_ticker_cls.return_value.history.call_args
    assert kwargs["start"] == "2020-01-03"  # 캐시 마지막 날짜(01-02) 다음날부터만 요청
    assert kwargs["end"] == "2020-01-04"
    assert sorted(result["date"].dt.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-01-02", "2020-01-03"]


def test_incremental_skips_network_when_cache_already_up_to_date(tmp_path):
    cache_path = tmp_path / "AAPL.parquet"
    cached = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [100, 100],
        }
    )
    cached.to_parquet(cache_path, index=False)

    with patch("src.data_collection.ohlcv.yf.Ticker") as mock_ticker_cls:
        result = fetch_us_ohlcv("AAPL", "2020-01-01", "2020-01-03", tmp_path)

    mock_ticker_cls.assert_not_called()
    assert len(result) == 2


def test_incremental_false_returns_cache_without_network_call(tmp_path):
    cache_path = tmp_path / "AAPL.parquet"
    cached = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [100],
        }
    )
    cached.to_parquet(cache_path, index=False)

    with patch("src.data_collection.ohlcv.yf.Ticker") as mock_ticker_cls:
        result = fetch_us_ohlcv("AAPL", "2020-01-01", "2020-06-01", tmp_path, incremental=False)

    mock_ticker_cls.assert_not_called()
    assert len(result) == 1


def test_network_failure_falls_back_to_cache(tmp_path):
    cache_path = tmp_path / "AAPL.parquet"
    cached = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [100],
        }
    )
    cached.to_parquet(cache_path, index=False)

    with patch("src.data_collection.ohlcv.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.return_value = _mock_ticker(pd.DataFrame())  # empty response -> None
        result = fetch_us_ohlcv("AAPL", "2020-01-01", "2020-06-01", tmp_path, max_retries=1)

    assert len(result) == 1  # 실패해도 기존 캐시는 유실되지 않음


def test_no_cache_and_network_failure_returns_none(tmp_path):
    with patch("src.data_collection.ohlcv.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.return_value = _mock_ticker(pd.DataFrame())
        result = fetch_us_ohlcv("AAPL", "2020-01-01", "2020-01-03", tmp_path, max_retries=1)

    assert result is None
