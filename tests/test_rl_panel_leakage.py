import numpy as np
import pandas as pd
import pytest

from src.features.indicators import BASE_FEATURE_COLUMNS, EVENT_FEATURE_COLUMNS
from src.rl.panel import (
    MODEL_V3_PROB_COLUMN,
    build_grid,
    load_ticker_universe,
    save_ticker_universe,
)


def _synthetic_features_df(rows: list[dict]) -> pd.DataFrame:
    """rows의 각 dict는 date/ticker/prob와 선택적으로 market_id/size_id를 담는다.
    BASE_FEATURE_COLUMNS/EVENT_FEATURE_COLUMNS는 이 파일의 관심사(그리드 정렬·
    마스킹)와 무관하므로 임의 상수로 채운다."""
    records = []
    for r in rows:
        rec = {c: 1.0 for c in BASE_FEATURE_COLUMNS}
        rec.update({c: 0.5 for c in EVENT_FEATURE_COLUMNS})
        rec["date"] = pd.Timestamp(r["date"])
        rec["ticker"] = r["ticker"]
        rec["market_id"] = r.get("market_id", 0)
        rec["size_id"] = r.get("size_id", 2)
        rec[MODEL_V3_PROB_COLUMN] = r["prob"]
        records.append(rec)
    return pd.DataFrame(records)


def _synthetic_ohlcv_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": pd.Timestamp(r["date"]), "ticker": r["ticker"], "close": r["close"]} for r in rows]
    )


def test_missing_ticker_date_is_zero_filled_and_not_forward_filled():
    """AAA는 1/2에 row가 없음(BBB는 있어서 1/2가 패널 날짜축엔 포함됨) — 1/2의
    AAA는 0-채움+valid_mask=0이어야 하고, 1/1의 값(prob=0.9)이 forward-fill로
    새어들면 안 된다."""
    rows = [
        {"date": "2024-01-01", "ticker": "AAA", "prob": 0.9},
        {"date": "2024-01-02", "ticker": "BBB", "prob": 0.7},
        {"date": "2024-01-03", "ticker": "AAA", "prob": 0.1},
        {"date": "2024-01-01", "ticker": "BBB", "prob": 0.6},
        {"date": "2024-01-03", "ticker": "BBB", "prob": 0.2},
    ]
    features_df = _synthetic_features_df(rows)
    tickers = ["AAA", "BBB"]
    ohlcv_df = _synthetic_ohlcv_df(
        [
            {"date": "2024-01-01", "ticker": "AAA", "close": 100.0},
            {"date": "2024-01-03", "ticker": "AAA", "close": 110.0},
            {"date": "2024-01-01", "ticker": "BBB", "close": 50.0},
            {"date": "2024-01-02", "ticker": "BBB", "close": 51.0},
            {"date": "2024-01-03", "ticker": "BBB", "close": 52.0},
        ]
    )

    panel = build_grid(features_df, ohlcv_df, tickers, include_event_features=True)

    assert len(panel.dates) == 3
    aaa_idx = panel.tickers.index("AAA")
    day2 = 1  # 2024-01-02

    assert panel.valid_mask[day2, aaa_idx] == 0.0
    assert np.all(panel.features[day2, aaa_idx, :-1] == 0.0)
    assert np.isnan(panel.close[day2, aaa_idx])

    prob_idx = panel.feature_names.index(MODEL_V3_PROB_COLUMN)
    assert panel.features[day2, aaa_idx, prob_idx] == 0.0  # 1/1의 0.9가 새어들지 않음


def test_model_v3_prob_aligns_to_correct_ticker_date():
    rows = [
        {"date": "2024-01-01", "ticker": "AAA", "prob": 0.9},
        {"date": "2024-01-01", "ticker": "BBB", "prob": 0.3},
    ]
    features_df = _synthetic_features_df(rows)
    tickers = ["AAA", "BBB"]
    ohlcv_df = _synthetic_ohlcv_df(
        [
            {"date": "2024-01-01", "ticker": "AAA", "close": 100.0},
            {"date": "2024-01-01", "ticker": "BBB", "close": 50.0},
        ]
    )
    panel = build_grid(features_df, ohlcv_df, tickers)
    prob_idx = panel.feature_names.index(MODEL_V3_PROB_COLUMN)

    assert panel.features[0, panel.tickers.index("AAA"), prob_idx] == pytest.approx(0.9)
    assert panel.features[0, panel.tickers.index("BBB"), prob_idx] == pytest.approx(0.3)


def test_size_id_can_change_over_time_for_same_ticker():
    """size_id는 연 2회 재분류가 일어나므로 티커당 한 번만 캐싱하면 안 된다."""
    rows = [
        {"date": "2024-01-01", "ticker": "AAA", "size_id": 0, "prob": 0.5},
        {"date": "2024-06-01", "ticker": "AAA", "size_id": 2, "prob": 0.5},
    ]
    features_df = _synthetic_features_df(rows)
    ohlcv_df = _synthetic_ohlcv_df(
        [
            {"date": "2024-01-01", "ticker": "AAA", "close": 100.0},
            {"date": "2024-06-01", "ticker": "AAA", "close": 100.0},
        ]
    )
    panel = build_grid(features_df, ohlcv_df, ["AAA"])
    size_idx = panel.feature_names.index("size_id")

    assert panel.features[0, 0, size_idx] == 0.0
    assert panel.features[1, 0, size_idx] == 2.0


def test_include_event_features_toggle_changes_dimension():
    rows = [{"date": "2024-01-01", "ticker": "AAA", "prob": 0.5}]
    features_df = _synthetic_features_df(rows)
    ohlcv_df = _synthetic_ohlcv_df([{"date": "2024-01-01", "ticker": "AAA", "close": 100.0}])

    with_events = build_grid(features_df, ohlcv_df, ["AAA"], include_event_features=True)
    without_events = build_grid(features_df, ohlcv_df, ["AAA"], include_event_features=False)

    assert with_events.features.shape[-1] == len(BASE_FEATURE_COLUMNS) + 1 + len(EVENT_FEATURE_COLUMNS) + 3
    assert without_events.features.shape[-1] == len(BASE_FEATURE_COLUMNS) + 1 + 3
    assert with_events.feature_names[-1] == "valid_mask"
    assert without_events.feature_names[-1] == "valid_mask"


def test_duplicate_date_ticker_raises():
    rows = [
        {"date": "2024-01-01", "ticker": "AAA", "prob": 0.5},
        {"date": "2024-01-01", "ticker": "AAA", "prob": 0.6},
    ]
    features_df = _synthetic_features_df(rows)
    ohlcv_df = _synthetic_ohlcv_df([{"date": "2024-01-01", "ticker": "AAA", "close": 100.0}])

    with pytest.raises(ValueError):
        build_grid(features_df, ohlcv_df, ["AAA"])


def test_ticker_universe_round_trip(tmp_path):
    tickers = ["BBB", "AAA", "CCC"]
    save_ticker_universe(tickers, tmp_path)
    loaded = load_ticker_universe(tmp_path)
    assert loaded == tickers
