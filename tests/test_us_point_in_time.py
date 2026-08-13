import pandas as pd

from src.data_collection.us_point_in_time import _split_tickers, build_membership_intervals


def test_split_tickers_handles_slash_separated_dual_class():
    assert _split_tickers("UA/UAA") == ["UA", "UAA"]


def test_split_tickers_handles_comma_separated_dual_class():
    assert _split_tickers("CWEN, CWEN-A") == ["CWEN", "CWEN-A"]


def test_split_tickers_handles_single_ticker():
    assert _split_tickers("AAPL") == ["AAPL"]


def test_split_tickers_normalizes_dot_notation_within_group():
    assert _split_tickers("BRK.B/BF.B") == ["BRK-B", "BF-B"]


def test_dual_class_tickers_get_independent_intervals():
    """'UA/UAA' 같은 한 셀짜리 이벤트가 UA, UAA 각각 독립적인 소속 이력으로
    반영돼야 한다 — 합쳐진 문자열이 통째로 가짜 티커가 되면 안 된다."""
    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-06-21", "2025-12-22"]),
            "added": ["UA", None],
            "removed": [None, "UA"],
        }
    )
    intervals, _ = build_membership_intervals(current_constituents=set(), changes=changes)
    ua_rows = intervals[intervals["ticker"] == "UA"]
    assert len(ua_rows) == 1
    assert ua_rows.iloc[0]["start_date"] == pd.Timestamp("2022-06-21")
    assert ua_rows.iloc[0]["end_date"] == pd.Timestamp("2025-12-22")
