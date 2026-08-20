from unittest.mock import patch

import pandas as pd

from src.data_collection.events_us import (
    START_DATE,
    _filings_block_to_df,
    _filter_target_filings,
    fetch_us_filings,
)


def test_filings_block_to_df_handles_empty_block():
    df = _filings_block_to_df({}, cik=123)
    assert df.empty


def test_filings_block_to_df_attaches_cik():
    block = {"form": ["8-K"], "filingDate": ["2020-01-01"], "accessionNumber": ["0001"]}
    df = _filings_block_to_df(block, cik=320193)
    assert (df["cik"] == 320193).all()


def test_filter_target_filings_keeps_only_8k():
    df = pd.DataFrame(
        {
            "form": ["8-K", "10-K", "8-K"],
            "filingDate": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "accessionNumber": ["0001", "0002", "0003"],
        }
    )
    result = _filter_target_filings(df, ticker="AAPL", start_date=START_DATE)
    assert set(result["form"]) == {"8-K"}
    assert len(result) == 2


def test_filter_target_filings_drops_before_start_date():
    df = pd.DataFrame(
        {
            "form": ["8-K", "8-K"],
            "filingDate": ["2014-12-31", "2015-01-01"],
            "accessionNumber": ["0001", "0002"],
        }
    )
    result = _filter_target_filings(df, ticker="AAPL", start_date="2015-01-01")
    assert len(result) == 1
    assert result.iloc[0]["filingDate"] == "2015-01-01"


def test_filter_target_filings_attaches_ticker():
    df = pd.DataFrame({"form": ["8-K"], "filingDate": ["2020-01-01"], "accessionNumber": ["0001"]})
    result = _filter_target_filings(df, ticker="AAPL", start_date=START_DATE)
    assert (result["ticker"] == "AAPL").all()


def _filings_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_fetch_us_filings_no_cache_fetches_from_start_date(tmp_path):
    with patch("src.data_collection.events_us._fetch_all_filings_for_cik") as mock_fetch:
        mock_fetch.return_value = _filings_df(
            [{"form": "8-K", "filingDate": "2020-01-05", "accessionNumber": "0001"}]
        )
        result = fetch_us_filings(cik=320193, ticker="AAPL", raw_dir=tmp_path, start_date="2015-01-01")

    called_start_date = mock_fetch.call_args.args[1]
    assert called_start_date == "2015-01-01"
    assert len(result) == 1
    assert (tmp_path / "events" / "filings_AAPL.parquet").exists()


def test_fetch_us_filings_incremental_uses_cached_max_filing_date(tmp_path):
    cache_path = tmp_path / "events" / "filings_AAPL.parquet"
    cache_path.parent.mkdir(parents=True)
    cached = _filings_df([{"form": "8-K", "filingDate": "2020-01-05", "accessionNumber": "0001"}])
    cached.to_parquet(cache_path, index=False)

    with patch("src.data_collection.events_us._fetch_all_filings_for_cik") as mock_fetch:
        mock_fetch.return_value = _filings_df(
            [{"form": "8-K", "filingDate": "2020-02-10", "accessionNumber": "0002"}]
        )
        result = fetch_us_filings(cik=320193, ticker="AAPL", raw_dir=tmp_path, start_date="2015-01-01")

    called_start_date = mock_fetch.call_args.args[1]
    assert called_start_date == "2020-01-05"  # start_date가 아니라 캐시 마지막 filingDate부터 재조회
    assert sorted(result["accessionNumber"]) == ["0001", "0002"]


def test_fetch_us_filings_incremental_dedups_by_accession_number(tmp_path):
    cache_path = tmp_path / "events" / "filings_AAPL.parquet"
    cache_path.parent.mkdir(parents=True)
    cached = _filings_df([{"form": "8-K", "filingDate": "2020-01-05", "accessionNumber": "0001"}])
    cached.to_parquet(cache_path, index=False)

    with patch("src.data_collection.events_us._fetch_all_filings_for_cik") as mock_fetch:
        # 재조회 시 캐시에 이미 있는 필링(0001)이 그대로 다시 오는 경우(당일 재조회 시 흔함)
        mock_fetch.return_value = _filings_df(
            [{"form": "8-K", "filingDate": "2020-01-05", "accessionNumber": "0001"}]
        )
        result = fetch_us_filings(cik=320193, ticker="AAPL", raw_dir=tmp_path)

    assert len(result) == 1


def test_fetch_us_filings_incremental_false_returns_cache_without_network_call(tmp_path):
    cache_path = tmp_path / "events" / "filings_AAPL.parquet"
    cache_path.parent.mkdir(parents=True)
    cached = _filings_df([{"form": "8-K", "filingDate": "2020-01-05", "accessionNumber": "0001"}])
    cached.to_parquet(cache_path, index=False)

    with patch("src.data_collection.events_us._fetch_all_filings_for_cik") as mock_fetch:
        result = fetch_us_filings(cik=320193, ticker="AAPL", raw_dir=tmp_path, incremental=False)

    mock_fetch.assert_not_called()
    assert len(result) == 1


def test_fetch_us_filings_force_ignores_cache_and_refetches_from_start_date(tmp_path):
    cache_path = tmp_path / "events" / "filings_AAPL.parquet"
    cache_path.parent.mkdir(parents=True)
    cached = _filings_df([{"form": "8-K", "filingDate": "2020-01-05", "accessionNumber": "0001"}])
    cached.to_parquet(cache_path, index=False)

    with patch("src.data_collection.events_us._fetch_all_filings_for_cik") as mock_fetch:
        mock_fetch.return_value = _filings_df(
            [{"form": "8-K", "filingDate": "2020-01-05", "accessionNumber": "0099"}]
        )
        result = fetch_us_filings(
            cik=320193, ticker="AAPL", raw_dir=tmp_path, start_date="2015-01-01", force=True
        )

    called_start_date = mock_fetch.call_args.args[1]
    assert called_start_date == "2015-01-01"
    assert list(result["accessionNumber"]) == ["0099"]  # force는 캐시와 병합하지 않고 전체 대체
