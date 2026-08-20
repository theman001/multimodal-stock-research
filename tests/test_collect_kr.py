"""fetch_all_daily_snapshots()/extract_universe_ohlcv()의 증분 처리 회귀 테스트.

두 함수 모두 예전엔 호출될 때마다 start_date(2014년 말)~end_date(오늘)
~4,000일 전체를 매번 다시 순회했다(캐시 히트라 네트워크 호출은 없지만
parquet 파일을 매번 다시 여는 낭비 — plan/10 §B-1). 이 테스트는 실제 KRX
네트워크 호출 없이(mock) 두 함수가 이전 호출 이후로 늘어난 날짜만 처리하는지
확인한다.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from src.data_collection.collect_kr import extract_universe_ohlcv, fetch_all_daily_snapshots


def _snapshot(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": tickers,
            "open": [1.0] * len(tickers),
            "high": [1.0] * len(tickers),
            "low": [1.0] * len(tickers),
            "close": [1.0] * len(tickers),
            "volume": [100] * len(tickers),
            "market_cap": [1000.0] * len(tickers),
        }
    )


def test_fetch_all_daily_snapshots_no_state_processes_full_range(tmp_path):
    with patch("src.data_collection.collect_kr.fetch_daily_snapshot") as mock_fetch:
        mock_fetch.return_value = pd.DataFrame()
        fetch_all_daily_snapshots(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"), tmp_path)

    called_dates = [call.args[0] for call in mock_fetch.call_args_list]
    assert called_dates == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    state_path = tmp_path / "stock_daily_trade" / "_last_processed_date.txt"
    assert state_path.read_text().strip() == "2020-01-03"


def test_fetch_all_daily_snapshots_incremental_resumes_from_last_processed_date(tmp_path):
    state_path = tmp_path / "stock_daily_trade" / "_last_processed_date.txt"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("2020-01-02")

    with patch("src.data_collection.collect_kr.fetch_daily_snapshot") as mock_fetch:
        mock_fetch.return_value = pd.DataFrame()
        fetch_all_daily_snapshots(pd.Timestamp("2014-12-01"), pd.Timestamp("2020-01-04"), tmp_path)

    called_dates = [call.args[0] for call in mock_fetch.call_args_list]
    # start_date(2014-12-01)가 아니라 상태 파일의 마지막 처리일(2020-01-02) 다음날부터만 처리
    assert called_dates == [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-04")]
    assert state_path.read_text().strip() == "2020-01-04"


def test_fetch_all_daily_snapshots_incremental_false_ignores_state(tmp_path):
    state_path = tmp_path / "stock_daily_trade" / "_last_processed_date.txt"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("2020-01-02")

    with patch("src.data_collection.collect_kr.fetch_daily_snapshot") as mock_fetch:
        mock_fetch.return_value = pd.DataFrame()
        fetch_all_daily_snapshots(
            pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"), tmp_path, incremental=False
        )

    called_dates = [call.args[0] for call in mock_fetch.call_args_list]
    assert called_dates == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]


def test_extract_universe_ohlcv_no_cache_extracts_full_range(tmp_path):
    with patch("src.data_collection.collect_kr.fetch_daily_snapshot") as mock_fetch:
        mock_fetch.return_value = _snapshot(["005930", "000660"])
        result = extract_universe_ohlcv(
            {"005930"}, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), tmp_path
        )

    assert mock_fetch.call_count == 2
    assert sorted(result["date"].dt.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-01-02"]
    assert set(result["ticker"]) == {"005930"}
    assert (tmp_path / "kr_universe_ohlcv_cache.parquet").exists()
    assert (tmp_path / "kr_universe_ohlcv_cache_tickers.txt").read_text().strip() == "005930"


def test_extract_universe_ohlcv_incremental_reuses_cache_and_appends_new_dates(tmp_path):
    cache_path = tmp_path / "kr_universe_ohlcv_cache.parquet"
    ticker_set_path = tmp_path / "kr_universe_ohlcv_cache_tickers.txt"
    cached = _snapshot(["005930"])
    cached["date"] = pd.Timestamp("2020-01-01")
    cached.to_parquet(cache_path, index=False)
    ticker_set_path.write_text("005930")

    with patch("src.data_collection.collect_kr.fetch_daily_snapshot") as mock_fetch:
        mock_fetch.return_value = _snapshot(["005930"])
        result = extract_universe_ohlcv(
            {"005930"}, pd.Timestamp("2015-01-01"), pd.Timestamp("2020-01-03"), tmp_path
        )

    # start_date(2015-01-01)가 아니라 캐시 마지막 날짜(2020-01-01) 다음날부터만 스냅샷을 다시 연다
    called_dates = [call.args[0] for call in mock_fetch.call_args_list]
    assert called_dates == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    assert sorted(result["date"].dt.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-01-02", "2020-01-03"]


def test_extract_universe_ohlcv_early_return_respects_smaller_end_date(tmp_path):
    """캐시에 이번 호출의 end_date보다 늦은 날짜까지 이미 들어있으면(예: 과거
    재현/디버깅 목적으로 오늘보다 이전 end_date를 넘기는 경우), 캐시를 그대로
    반환하지 않고 end_date 이하로 걸러서 반환해야 한다."""
    cache_path = tmp_path / "kr_universe_ohlcv_cache.parquet"
    ticker_set_path = tmp_path / "kr_universe_ohlcv_cache_tickers.txt"
    cached = pd.concat(
        [
            _snapshot(["005930"]).assign(date=pd.Timestamp("2020-01-01")),
            _snapshot(["005930"]).assign(date=pd.Timestamp("2020-01-05")),
        ],
        ignore_index=True,
    )
    cached.to_parquet(cache_path, index=False)
    ticker_set_path.write_text("005930")

    with patch("src.data_collection.collect_kr.fetch_daily_snapshot") as mock_fetch:
        result = extract_universe_ohlcv(
            {"005930"}, pd.Timestamp("2015-01-01"), pd.Timestamp("2020-01-02"), tmp_path
        )

    mock_fetch.assert_not_called()  # 캐시가 이미 end_date를 넘어서므로 새로 스냅샷을 열 필요 없음
    assert sorted(result["date"].dt.strftime("%Y-%m-%d")) == ["2020-01-01"]  # 01-05는 end_date(01-02) 초과라 제외


def test_extract_universe_ohlcv_ticker_set_changed_invalidates_cache(tmp_path):
    cache_path = tmp_path / "kr_universe_ohlcv_cache.parquet"
    ticker_set_path = tmp_path / "kr_universe_ohlcv_cache_tickers.txt"
    cached = _snapshot(["005930"])
    cached["date"] = pd.Timestamp("2020-01-01")
    cached.to_parquet(cache_path, index=False)
    ticker_set_path.write_text("005930")

    with patch("src.data_collection.collect_kr.fetch_daily_snapshot") as mock_fetch:
        mock_fetch.return_value = _snapshot(["005930", "000660"])
        result = extract_universe_ohlcv(
            {"005930", "000660"}, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), tmp_path
        )

    # 유니버스가 바뀌었으므로(정기변경 등) 캐시를 무시하고 start_date부터 전체 재추출
    called_dates = [call.args[0] for call in mock_fetch.call_args_list]
    assert called_dates == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    assert set(result["ticker"]) == {"005930", "000660"}


def test_extract_universe_ohlcv_incremental_false_ignores_cache(tmp_path):
    cache_path = tmp_path / "kr_universe_ohlcv_cache.parquet"
    ticker_set_path = tmp_path / "kr_universe_ohlcv_cache_tickers.txt"
    cached = _snapshot(["005930"])
    cached["date"] = pd.Timestamp("2020-01-01")
    cached.to_parquet(cache_path, index=False)
    ticker_set_path.write_text("005930")

    with patch("src.data_collection.collect_kr.fetch_daily_snapshot") as mock_fetch:
        mock_fetch.return_value = _snapshot(["005930"])
        extract_universe_ohlcv(
            {"005930"}, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), tmp_path, incremental=False
        )

    called_dates = [call.args[0] for call in mock_fetch.call_args_list]
    assert called_dates == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
