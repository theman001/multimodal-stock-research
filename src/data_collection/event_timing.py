"""이벤트 발생 시각 → 거래일 정렬 규칙.

CLAUDE.md "Phase 2 구체 사양"의 데이터 누수 방지 원칙 4를 코드로 구현한다.
핵심 원칙: 시각이 불확실하거나 애매하면 항상 더 보수적인(늦은) 쪽으로 배정한다
— 장마감 후 공시가 당일 종가 예측에 선반영되면 안 된다.

- 한국(DART): `rcept_dt`는 날짜만 제공하고 시:분:초가 없다(API 문서로 확인
  완료) → 접수일자 **다음 거래일(D+1)**부터 무조건 반영.
- 미국(EDGAR): `acceptanceDateTime`은 UTC 초 단위까지 포함 → 정규장 마감
  (16:00 America/New_York, DST 자동 반영) **이전 접수는 당일, 이후 접수는
  다음 거래일**부터 반영. 단, NYSE 조기마감일(독립기념일 전날/추수감사절
  다음날/크리스마스이브, 13:00 마감)은 마감 기준을 13:00으로 낮춘다 — 이걸
  놓치면 조기마감 이후 접수된 공시가 "당일"로 잘못 배정돼 누수가 된다(round 2
  리뷰에서 실제 데이터 3건 발견: TSLA/JPM 2017-07-03, VNOM 2019-07-03).
  공휴일이 주말과 겹쳐 관찰일이 밀리는 경우(예: 2026년 7/4가 토요일이라
  조기마감이 7/2(목)로 밀림, NYSE 공식 캘린더로 확인)까지 반영하기 위해
  날짜를 하드코딩하지 않고 `trading_days` 캘린더로 "그 공휴일 직전 마지막
  거래일"을 직접 계산한다.
"""
from __future__ import annotations

import pandas as pd

US_MARKET_CLOSE_HOUR_ET = 16  # NYSE/NASDAQ 정규장 마감 16:00 America/New_York
US_EARLY_CLOSE_HOUR_ET = 13  # 조기마감일 마감 13:00 America/New_York


def _next_trading_day(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """date보다 엄격히 뒤에 있는 첫 거래일을 반환한다(date 당일은 제외)."""
    later = trading_days[trading_days > date]
    if len(later) == 0:
        raise ValueError(f"거래일 캘린더에 {date} 이후 거래일이 없음 — 캘린더 범위를 확인할 것")
    return later.min()


def _same_or_next_trading_day(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """date가 거래일이면 그대로, 아니면 그 다음 첫 거래일을 반환한다."""
    if date in trading_days:
        return date
    return _next_trading_day(date, trading_days)


def assign_kr_effective_date(rcept_dt: str, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """rcept_dt(YYYYMMDD, 시각 정보 없음)를 다음 거래일에 배정한다 — 보수적 D+1."""
    date = pd.Timestamp(str(rcept_dt))
    return _next_trading_day(date, trading_days)


def _last_trading_day_before(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """date 이전 마지막 거래일을 반환한다.

    `trading_days`가 아직 date까지 도달하지 못했으면(그 시점 데이터가 아직
    없음) None을 반환한다 — 도달 못 했는데 그냥 `trading_days[trading_days <
    date].max()`만 하면 캘린더의 "현재까지의 마지막 날짜"가 우연히 "그 공휴일
    직전 마지막 거래일"인 것처럼 착각된다. round 2 리뷰에서 실제로 발견:
    캘린더가 2026-08-10까지밖에 없는 상태에서 12/25 이전 마지막 거래일을
    구하려다 8/10을 "크리스마스 조기마감일"로 잘못 판정했었다.
    """
    if len(trading_days) == 0 or trading_days.max() < date:
        return None
    earlier = trading_days[trading_days < date]
    return earlier.max() if len(earlier) else None


def _us_early_close_days_for_year(year: int, trading_days: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """NYSE 조기마감(13:00 ET)일 3개를 계산한다 — 독립기념일 전날, 추수감사절
    다음날, 크리스마스이브. 날짜를 고정 목록으로 하드코딩하지 않고 매번
    `trading_days`로 "그 공휴일 직전 마지막 거래일"을 계산하는 이유는, 공휴일이
    주말과 겹쳐 관찰일이 밀리는 해가 있기 때문이다(예: 2026년 7/4가 토요일이라
    조기마감이 7/2(목)로 밀림 — NYSE 공식 캘린더로 확인 완료).
    """
    days = set()

    november = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="D")
    thanksgiving = november[november.dayofweek == 3][3]  # 11월 네 번째 목요일
    days.add(thanksgiving + pd.Timedelta(days=1))  # 다음날(금)은 관찰일 이동 없음

    before_july4 = _last_trading_day_before(pd.Timestamp(year, 7, 4), trading_days)
    if before_july4 is not None:
        days.add(before_july4)

    before_christmas = _last_trading_day_before(pd.Timestamp(year, 12, 25), trading_days)
    if before_christmas is not None:
        days.add(before_christmas)

    return days


def assign_us_effective_date(acceptance_datetime: str, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """acceptanceDateTime(UTC)을 America/New_York 마감 기준으로 당일/익일 배정한다.

    조기마감일은 13:00, 그 외는 16:00을 마감 기준으로 쓴다.
    """
    ts_utc = pd.Timestamp(acceptance_datetime)
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    ts_et = ts_utc.tz_convert("America/New_York")
    filing_date = pd.Timestamp(ts_et.year, ts_et.month, ts_et.day)

    early_close_days = _us_early_close_days_for_year(filing_date.year, trading_days)
    close_hour = US_EARLY_CLOSE_HOUR_ET if filing_date in early_close_days else US_MARKET_CLOSE_HOUR_ET
    market_close = ts_et.replace(hour=close_hour, minute=0, second=0, microsecond=0)

    if ts_et < market_close:
        return _same_or_next_trading_day(filing_date, trading_days)
    return _next_trading_day(filing_date, trading_days)
