import pandas as pd

from src.data_collection.events_us import (
    START_DATE,
    _filings_block_to_df,
    _filter_target_filings,
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
