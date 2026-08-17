"""기술지표 계산 (CLAUDE.md 피처 목록, plan/02_feature_engineering.md 기준).

원칙:
- 절대가격이 아닌 비율/변화율만 사용 (국내/해외 가격 스케일 차이가 market_id와
  혼동되는 것을 방지).
- 모든 롤링/지수이동 계산은 과거 데이터만 사용한다 (`center=True` 절대 금지).
  `shift(-N)`(미래 참조)은 이 모듈에서 target 계산에만 사용한다.
- 이 파일의 모든 함수는 단일 종목(하나의 date-정렬된 시계열) 기준으로 동작한다
  — 여러 종목을 섞어서 넣으면 롤링 계산이 오염된다.
"""
from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """표준 RSI (Wilder's smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD, Signal, Histogram (12, 26, 9) — Close로 나눠 비율화한 버전.

    교과서적 MACD(ema_fast - ema_slow)는 원화/달러 절대가격 단위라 국내(수만 원대)와
    해외(수십~수백 달러대) 종목 간 스케일이 완전히 달라진다 — CLAUDE.md의 "모든
    피처는 비율/변화율로, 국내/해외 가격 스케일 차이가 market_id와 혼동되지 않게"
    원칙을 정면으로 어기게 된다. 따라서 (ema_fast - ema_slow)를 close로 나눠
    무차원 비율로 만든 뒤(Percentage Price Oscillator와 동일한 발상) signal/hist도
    그 비율 시계열 기준으로 계산한다.
    """
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow) / close
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_pctb(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger %B = (Close - Lower) / (Upper - Lower), (20, 2σ)."""
    ma = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (close - lower) / (upper - lower)


def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """단일 종목의 date-정렬 OHLCV에 피처 + target 컬럼을 추가해서 반환한다.

    입력에는 최소 date, open, high, low, close, volume 컬럼이 있어야 한다.
    market_id/size_id 등 메타 컬럼이 있으면 그대로 통과시킨다.
    """
    df = ohlcv.sort_values("date").reset_index(drop=True).copy()
    close = df["close"]
    volume = df["volume"]

    ma5 = close.rolling(5, min_periods=5).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    df["ma5_ratio"] = ma5 / close - 1
    df["ma20_ratio"] = ma20 / close - 1
    df["ma60_ratio"] = ma60 / close - 1

    df["rsi14"] = rsi(close, 14)

    macd_line, signal_line, hist = macd(close, 12, 26, 9)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    df["bb_pctb"] = bollinger_pctb(close, 20, 2.0)

    df["vol_chg5"] = volume / volume.shift(5) - 1

    daily_return = close.pct_change()
    df["volatility20"] = daily_return.rolling(20, min_periods=20).std()

    df["momentum5"] = close / close.shift(5) - 1
    df["momentum20"] = close / close.shift(20) - 1
    df["momentum60"] = close / close.shift(60) - 1

    # target 계산에만 미래 값(shift(-1))을 사용한다 — 피처 계산에는 절대 사용 금지.
    future_close = close.shift(-1)
    target = (future_close / close - 1 > 0).astype("Int64")
    target[future_close.isna()] = pd.NA  # NaN > 0은 False로 평가되므로 명시적으로 결측 처리
    df["target"] = target

    return df


BASE_FEATURE_COLUMNS = [
    "ma5_ratio",
    "ma20_ratio",
    "ma60_ratio",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_pctb",
    "vol_chg5",
    "volatility20",
    "momentum5",
    "momentum20",
    "momentum60",
]

# Phase 2 모듈 4 — 이벤트 피처(src/features/event_features.py에서 계산, ticker+date
# 기준으로 features.parquet에 병합됨). compute_features()는 여기 관여하지 않는다 —
# 이 목록은 순전히 다운스트림(모델 학습 파이프라인)이 참조할 컬럼 이름 registry.
EVENT_FEATURE_COLUMNS = [
    "event_count_5d",
    "event_sentiment_latest",
    "event_sentiment_mean5d",
    "days_since_last_event",
]

# 모델 학습 파이프라인이 쓰는 최종 피처 목록 (기술지표 13개 + 이벤트 4개 = 17개).
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + EVENT_FEATURE_COLUMNS
