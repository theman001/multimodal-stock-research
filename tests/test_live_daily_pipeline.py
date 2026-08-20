"""daily_pipeline.py 유닛테스트.

핵심 검증 대상: build_features()(Phase 1)가 target(내일 종가) non-null 행만
남기는 탓에 "오늘"(가장 최근 거래일)이 features.parquet에 구조적으로 절대
못 들어가는 문제를, build_live_features_window()가 compute_features()를
재사용해 메모리상으로 메우는지 확인한다(파일에는 안 씀)."""
import numpy as np
import pandas as pd
import pytest

from src.features.indicators import BASE_FEATURE_COLUMNS, EVENT_FEATURE_COLUMNS, compute_features
from src.live.daily_pipeline import build_live_features_window, merge_ohlcv_meta

N_WARMUP_DAYS = 65  # MA60/momentum60 충족에 필요한 최소치(60)보다 여유


def _make_ohlcv(tickers: list[str], n_days: int, market_id: int, start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(0)
    rows = []
    for ticker in tickers:
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, size=n_days))
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "market_id": market_id,
                    "size_id": 2,
                    "open": close[i],
                    "high": close[i] * 1.01,
                    "low": close[i] * 0.99,
                    "close": close[i],
                    "volume": 1_000_000,
                    "market_cap": 1e10,
                }
            )
    return pd.DataFrame(rows)


def _write_batch_features(processed_dir, ohlcv_meta: pd.DataFrame, tickers: list[str], drop_last_n: int = 1):
    """compute_features()로 배치 features.parquet을 흉내내되, build_features()와
    동일하게 마지막 drop_last_n개 행(target 없는 최근 거래일)은 뺀다."""
    frames = []
    for ticker in tickers:
        g = ohlcv_meta.loc[ohlcv_meta["ticker"] == ticker]
        computed = compute_features(g)
        computed = computed.dropna(subset=BASE_FEATURE_COLUMNS + ["target"])
        frames.append(computed[["date", "ticker", "market_id", "size_id", *BASE_FEATURE_COLUMNS, "target"]])
    features = pd.concat(frames, ignore_index=True)
    features["target"] = features["target"].astype(int)
    features.to_parquet(processed_dir / "features.parquet", index=False)
    return features


@pytest.fixture
def data_root(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    return tmp_path


def test_merge_ohlcv_meta_combines_us_and_kr(data_root):
    us = _make_ohlcv(["AAPL"], 5, market_id=0)
    kr = _make_ohlcv(["005930"], 5, market_id=1)
    us.to_parquet(data_root / "processed" / "ohlcv_meta_us.parquet", index=False)
    kr.to_parquet(data_root / "processed" / "ohlcv_meta_kr.parquet", index=False)

    combined = merge_ohlcv_meta(data_root)

    assert len(combined) == len(us) + len(kr)
    assert set(combined["ticker"].unique()) == {"AAPL", "005930"}
    on_disk = pd.read_parquet(data_root / "processed" / "ohlcv_meta.parquet")
    assert len(on_disk) == len(combined)


def test_merge_ohlcv_meta_handles_missing_kr_file(data_root):
    """KR/US 트랙이 독립적으로 개발된다는 설계(plan/10 §0-5)상, KR 파일이
    아직 없다고(한 번도 안 돌았거나 새로 clone한 경우) 막으면 안 된다."""
    us = _make_ohlcv(["AAPL"], 5, market_id=0)
    us.to_parquet(data_root / "processed" / "ohlcv_meta_us.parquet", index=False)

    combined = merge_ohlcv_meta(data_root)
    assert set(combined["ticker"].unique()) == {"AAPL"}


def test_merge_ohlcv_meta_raises_if_both_missing(data_root):
    with pytest.raises(RuntimeError):
        merge_ohlcv_meta(data_root)


def test_build_live_features_window_computes_missing_today_row_matching_compute_features(data_root):
    tickers = ["AAPL", "MSFT"]
    ohlcv_meta = _make_ohlcv(tickers, N_WARMUP_DAYS, market_id=0)
    ohlcv_meta.to_parquet(data_root / "processed" / "ohlcv_meta.parquet", index=False)
    _write_batch_features(data_root / "processed", ohlcv_meta, tickers)

    target_date = ohlcv_meta["date"].max()  # 배치 경로에선 target 없어서 빠졌던 바로 그 날
    window = build_live_features_window(
        data_root, target_date=target_date, lookback_days=90, market_id=0, include_event_features=False
    )

    today_rows = window[window["date"] == target_date]
    assert set(today_rows["ticker"]) == set(tickers)
    assert "target" not in window.columns

    # 배치 경로(build_features()가 만들었을 값)와 수치까지 일치해야 한다 —
    # compute_features()를 직접 다시 돌려 대조.
    for ticker in tickers:
        expected = compute_features(ohlcv_meta.loc[ohlcv_meta["ticker"] == ticker])
        expected_row = expected.loc[expected["date"] == target_date].iloc[0]
        actual_row = today_rows.loc[today_rows["ticker"] == ticker].iloc[0]
        for col in BASE_FEATURE_COLUMNS:
            assert actual_row[col] == pytest.approx(expected_row[col]), col


def test_build_live_features_window_returns_unchanged_when_today_already_present(data_root):
    """오늘자 행이 이미 features.parquet에 있으면(드문 경우) 즉석 계산을
    다시 하지 않고 그대로 반환해야 한다."""
    tickers = ["AAPL"]
    ohlcv_meta = _make_ohlcv(tickers, N_WARMUP_DAYS, market_id=0)
    ohlcv_meta.to_parquet(data_root / "processed" / "ohlcv_meta.parquet", index=False)
    # drop_last_n=0: 마지막 행도 포함(target은 인위적으로 0으로 채움 — "이미 있는 경우"를 흉내)
    computed = compute_features(ohlcv_meta)
    computed["target"] = computed["target"].fillna(0).astype(int)
    computed = computed.dropna(subset=BASE_FEATURE_COLUMNS)
    computed[["date", "ticker", "market_id", "size_id", *BASE_FEATURE_COLUMNS, "target"]].to_parquet(
        data_root / "processed" / "features.parquet", index=False
    )

    target_date = ohlcv_meta["date"].max()
    window = build_live_features_window(
        data_root, target_date=target_date, lookback_days=90, market_id=0, include_event_features=False
    )
    assert len(window[window["date"] == target_date]) == 1


def test_build_live_features_window_skips_ticker_with_insufficient_history(data_root):
    """MA60 등 초기 롤링 윈도우를 못 채운(신규 상장/신규 편입) 티커는 오늘자
    행을 못 만든다 — 크래시 대신 조용히 빠져야 한다(build_grid()가 나중에
    0-채움+valid_mask=0으로 마스킹)."""
    established = "AAPL"
    new_ticker = "NEWCO"
    ohlcv_meta = pd.concat(
        [
            _make_ohlcv([established], N_WARMUP_DAYS, market_id=0),
            _make_ohlcv([new_ticker], 10, market_id=0),  # 60일 미만
        ],
        ignore_index=True,
    )
    ohlcv_meta.to_parquet(data_root / "processed" / "ohlcv_meta.parquet", index=False)
    _write_batch_features(data_root / "processed", ohlcv_meta, [established])
    # NEWCO는 build_features 경로에서도 전부 drop되므로 features.parquet에 아예 없음 —
    # 그래도 크래시 없이 established만 채워져야 한다.

    target_date = ohlcv_meta["date"].max()
    window = build_live_features_window(
        data_root, target_date=target_date, lookback_days=90, market_id=0, include_event_features=False
    )

    today_tickers = set(window.loc[window["date"] == target_date, "ticker"])
    assert established in today_tickers
    assert new_ticker not in today_tickers


def test_build_live_features_window_include_event_features(data_root):
    tickers = ["AAPL"]
    ohlcv_meta = _make_ohlcv(tickers, N_WARMUP_DAYS, market_id=0)
    ohlcv_meta.to_parquet(data_root / "processed" / "ohlcv_meta.parquet", index=False)
    _write_batch_features(data_root / "processed", ohlcv_meta, tickers)

    target_date = ohlcv_meta["date"].max()
    accession = "0001"
    events_us = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "cik": [320193],
            "form": ["8-K"],
            "filingDate": [str(pd.Timestamp(target_date).date())],
            "acceptanceDateTime": [f"{pd.Timestamp(target_date).date()}T12:00:00-04:00"],
            "accessionNumber": [accession],
            "primaryDocument": ["doc.htm"],
            "items": ["8.01"],
        }
    )
    events_sentiment_us = pd.DataFrame({"ticker": ["AAPL"], "accessionNumber": [accession], "score": [0.5]})
    events_us.to_parquet(data_root / "processed" / "events_us.parquet", index=False)
    events_sentiment_us.to_parquet(data_root / "processed" / "events_sentiment_us.parquet", index=False)

    window = build_live_features_window(
        data_root, target_date=target_date, lookback_days=90, market_id=0, include_event_features=True
    )

    today_row = window.loc[window["date"] == target_date].iloc[0]
    for col in EVENT_FEATURE_COLUMNS:
        assert col in window.columns
    assert today_row["event_count_5d"] >= 1


def test_build_live_features_window_event_count_5d_includes_events_from_prior_trading_days(data_root):
    """'5거래일' 윈도우는 그 종목의 최근 5개 거래일 row 기준이다(_compute_features_for_ticker
    docstring, 배치 경로와 동일한 정의여야 함) — target_date 당일이 아니라 2거래일
    전에 발생한 이벤트도 event_count_5d에 잡혀야 한다. row_dates를 target_date
    하루만 넘기면 window_start가 target_date 자신으로 무너져 이 이벤트가 빠지는
    버그가 있었다(review-loop 5차 재검토로 발견)."""
    tickers = ["AAPL"]
    ohlcv_meta = _make_ohlcv(tickers, N_WARMUP_DAYS, market_id=0)
    ohlcv_meta.to_parquet(data_root / "processed" / "ohlcv_meta.parquet", index=False)
    _write_batch_features(data_root / "processed", ohlcv_meta, tickers)

    dates = sorted(ohlcv_meta["date"].unique())
    target_date = dates[-1]
    event_date = dates[-3]  # target_date보다 2거래일 전 — 5거래일 윈도우 안이지만 당일은 아님

    accession = "0001"
    events_us = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "cik": [320193],
            "form": ["8-K"],
            "filingDate": [str(pd.Timestamp(event_date).date())],
            "acceptanceDateTime": [f"{pd.Timestamp(event_date).date()}T12:00:00-04:00"],
            "accessionNumber": [accession],
            "primaryDocument": ["doc.htm"],
            "items": ["8.01"],
        }
    )
    events_sentiment_us = pd.DataFrame({"ticker": ["AAPL"], "accessionNumber": [accession], "score": [0.5]})
    events_us.to_parquet(data_root / "processed" / "events_us.parquet", index=False)
    events_sentiment_us.to_parquet(data_root / "processed" / "events_sentiment_us.parquet", index=False)

    window = build_live_features_window(
        data_root, target_date=target_date, lookback_days=90, market_id=0, include_event_features=True
    )

    today_row = window.loc[window["date"] == target_date].iloc[0]
    assert today_row["event_count_5d"] >= 1
    assert today_row["event_sentiment_mean5d"] == pytest.approx(0.5)
