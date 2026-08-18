import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import src.rl.panel as panel_module
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


def test_market_id_is_correct_even_when_ticker_masked_on_every_sampled_row():
    """KRX 티커(market_id=1)가 특정 날짜에 마스킹되면 features[...,market_id_idx]는
    0-채움돼 US처럼 보인다 — Panel.market_id는 그 그리드가 아니라 원본
    features_df에서 직접 구해야 하므로 이 함정에 걸리면 안 된다."""
    rows = [
        {"date": "2024-01-01", "ticker": "KRX1", "market_id": 1, "prob": 0.5},
        {"date": "2024-01-02", "ticker": "OTHER", "market_id": 0, "prob": 0.5},
        # KRX1은 2024-01-02에 row가 없음 -> 그 날 grid상 market_id 채널은 0-채움됨
    ]
    features_df = _synthetic_features_df(rows)
    tickers = ["KRX1", "OTHER"]
    ohlcv_df = _synthetic_ohlcv_df(
        [
            {"date": "2024-01-01", "ticker": "KRX1", "close": 100.0},
            {"date": "2024-01-02", "ticker": "OTHER", "close": 50.0},
        ]
    )
    panel = build_grid(features_df, ohlcv_df, tickers)

    krx1_idx = panel.tickers.index("KRX1")
    market_id_channel_idx = panel.feature_names.index("market_id")

    # 그리드 자체는(마스킹된 행이라) 0으로 채워져 있어야 정상 — 이게 함정임을 재확인
    assert panel.features[1, krx1_idx, market_id_channel_idx] == 0.0
    # 하지만 Panel.market_id는 마스킹과 무관하게 진짜 값(1=KR)을 가져야 한다
    assert panel.market_id[krx1_idx] == 1


def test_dates_are_proper_datetime64_not_object_array():
    """np.array(sorted(...))가 Timestamp 리스트를 dtype=object로 만들어버리면
    obs_scaler.fit_obs_scaler의 `panel.dates <= cutoff` 비교가 조용히 깨진다
    (실 데이터로 실제로 겪은 버그) — panel.dates는 반드시 datetime64[ns]여야 한다."""
    rows = [{"date": "2024-01-01", "ticker": "AAA", "prob": 0.5}]
    features_df = _synthetic_features_df(rows)
    ohlcv_df = _synthetic_ohlcv_df([{"date": "2024-01-01", "ticker": "AAA", "close": 100.0}])

    panel = build_grid(features_df, ohlcv_df, ["AAA"])

    assert panel.dates.dtype == np.dtype("datetime64[ns]")
    cutoff = pd.Timestamp(panel.dates[0]).to_datetime64()
    assert (panel.dates <= cutoff).sum() == 1  # 비교 연산이 예외 없이 동작해야 함


def test_market_id_inconsistent_across_dates_raises():
    rows = [
        {"date": "2024-01-01", "ticker": "AAA", "market_id": 0, "prob": 0.5},
        {"date": "2024-01-02", "ticker": "AAA", "market_id": 1, "prob": 0.5},
    ]
    features_df = _synthetic_features_df(rows)
    ohlcv_df = _synthetic_ohlcv_df(
        [
            {"date": "2024-01-01", "ticker": "AAA", "close": 100.0},
            {"date": "2024-01-02", "ticker": "AAA", "close": 100.0},
        ]
    )
    with pytest.raises(ValueError):
        build_grid(features_df, ohlcv_df, ["AAA"])


class _RecordingModel:
    """predict_proba에 실제로 들어온 X를 기록하는 가짜 모델 — model_v3.json을
    실제로 로드하지 않고도 "스케일링된 입력이 들어오는가"를 검증하기 위함."""

    def __init__(self):
        self.received_X: pd.DataFrame | None = None

    def load_model(self, path):
        pass

    def predict_proba(self, X):
        self.received_X = X.copy()
        n = len(X)
        return np.column_stack([np.zeros(n), np.zeros(n)])


def test_score_model_v3_probabilities_scales_before_predicting(tmp_path, monkeypatch):
    """model_v3는 스케일링된 입력으로 학습됐다 — raw 값을 그대로 넣으면 트리
    분기 임계값이 완전히 다른 분포를 보게 돼 사실상 무작위 예측이 나온다(실측:
    raw 입력과 스케일링된 입력의 예측 상관계수가 0.044로 사실상 무관 — 실제로
    겪은 버그). 스케일러가 실제로 적용된 뒤 모델에 들어가는지 직접 검증한다."""
    all_cols = list(BASE_FEATURE_COLUMNS) + list(EVENT_FEATURE_COLUMNS)
    features_df = pd.DataFrame({c: [1.0, 3.0, 5.0] for c in all_cols})
    features_df["market_id"] = [0, 1, 0]
    features_df["size_id"] = [2, 1, 0]

    scaler = StandardScaler()
    scaler.fit(pd.DataFrame({c: [0.0, 2.0, 4.0, 6.0] for c in all_cols}))
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    with open(checkpoints_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    recording_model = _RecordingModel()
    monkeypatch.setattr(panel_module, "XGBClassifier", lambda: recording_model)

    panel_module.score_model_v3_probabilities(features_df, tmp_path)

    expected_scaled_all = scaler.transform(features_df[all_cols])
    expected_base = expected_scaled_all[:, : len(BASE_FEATURE_COLUMNS)]
    actual_base = recording_model.received_X[list(BASE_FEATURE_COLUMNS)].to_numpy()

    np.testing.assert_allclose(actual_base, expected_base, atol=1e-8)
    # 회귀 방지 핵심: raw 값 그대로 넘어가면 절대 안 됨
    assert not np.allclose(actual_base, features_df[list(BASE_FEATURE_COLUMNS)].to_numpy())
