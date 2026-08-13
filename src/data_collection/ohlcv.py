"""종목별 OHLCV 수집 (해외: yfinance). 캐시 + 재시도(backoff) + 실패 스킵."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_us_ohlcv(
    ticker: str,
    start: str,
    end: str,
    raw_dir: Path,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> pd.DataFrame | None:
    """단일 티커의 OHLCV를 [start, end)로 조회한다. 실패 시 None을 반환한다.

    raw_dir/{ticker}.parquet에 캐시하며, 캐시가 있으면 그대로 재사용한다
    (재실행 시 네트워크 요청을 줄이기 위함 — CLAUDE.md 리스크 대응).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / f"{ticker}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    for attempt in range(max_retries):
        try:
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
            if df.empty:
                raise ValueError("empty response")
            df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            df.to_parquet(cache_path, index=False)
            return df
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_backoff_seconds * (2**attempt))

    return None
