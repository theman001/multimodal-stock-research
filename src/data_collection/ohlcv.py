"""종목별 OHLCV 수집 (해외: yfinance). 캐시 + 재시도(backoff) + 실패 스킵."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf


def _fetch_yfinance_range(
    ticker: str, start: str, end: str, max_retries: int, retry_backoff_seconds: float
) -> pd.DataFrame | None:
    for attempt in range(max_retries):
        try:
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
            if df.empty:
                raise ValueError("empty response")
            df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            return df
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_backoff_seconds * (2**attempt))
    return None


def fetch_us_ohlcv(
    ticker: str,
    start: str,
    end: str,
    raw_dir: Path,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    incremental: bool = True,
) -> pd.DataFrame | None:
    """단일 티커의 OHLCV를 [start, end)로 조회한다. 실패 시 None을 반환한다.

    raw_dir/{ticker}.parquet에 캐시한다. `incremental=True`(기본값)면 캐시의
    마지막 날짜 이후분만 새로 조회해 append한다 — 캐시가 있다고 통째로
    재사용하면(구 동작) 매일 돌려도 새 날짜가 영영 안 붙는 버그가 있었음.
    `incremental=False`면 캐시가 있을 때 그대로 반환(구 동작, 네트워크 호출
    없이 재현하고 싶은 경우용).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / f"{ticker}.parquet"
    cached = pd.read_parquet(cache_path) if cache_path.exists() else None

    if cached is not None and not incremental:
        return cached

    fetch_start = start
    if cached is not None and not cached.empty:
        next_start = (pd.Timestamp(cached["date"].max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if next_start >= end:
            return cached
        fetch_start = next_start

    new_df = _fetch_yfinance_range(ticker, fetch_start, end, max_retries, retry_backoff_seconds)
    if new_df is None:
        return cached

    if cached is not None and not cached.empty:
        combined = (
            pd.concat([cached, new_df], ignore_index=True)
            .drop_duplicates(subset="date")
            .sort_values("date")
            .reset_index(drop=True)
        )
    else:
        combined = new_df

    combined.to_parquet(cache_path, index=False)
    return combined
