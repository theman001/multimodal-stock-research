import pandas as pd
import pytest

from src.data_collection.event_timing import assign_kr_effective_date, assign_us_effective_date

TRADING_DAYS = pd.bdate_range("2015-01-01", "2015-02-15")  # 주말만 제외한 간이 달력(공휴일 무시, 테스트용)


def test_kr_assigns_to_next_trading_day_never_same_day():
    """시각 정보가 없는 KR 공시는 접수일 당일에 반영되면 안 된다 — 항상 D+1."""
    effective = assign_kr_effective_date("20150106", TRADING_DAYS)
    assert effective > pd.Timestamp("2015-01-06")
    assert effective == pd.Timestamp("2015-01-07")


def test_kr_skips_weekend_to_reach_next_trading_day():
    """금요일 접수 공시는 토/일을 건너뛰고 월요일에 반영돼야 한다."""
    effective = assign_kr_effective_date("20150109", TRADING_DAYS)  # 2015-01-09는 금요일
    assert effective == pd.Timestamp("2015-01-12")  # 다음 월요일


def test_us_before_close_is_same_trading_day():
    """16:00 ET 이전 접수는 당일 반영 가능하다."""
    effective = assign_us_effective_date("2015-01-27T18:00:00.000Z", TRADING_DAYS)  # 13:00 EST
    assert effective == pd.Timestamp("2015-01-27")


def test_us_after_close_is_next_trading_day():
    """마감(16:00 ET) 이후 접수는 반드시 다음 거래일로 넘어가야 한다 — 실제 AAPL 사례."""
    effective = assign_us_effective_date("2015-01-27T21:31:00.000Z", TRADING_DAYS)  # 16:31 EST, 마감 직후
    assert effective == pd.Timestamp("2015-01-28")


def test_us_exact_close_boundary_treated_as_after_close():
    """정확히 16:00:00은 '마감 전'이 아니라 보수적으로 마감 후로 취급한다."""
    effective = assign_us_effective_date("2015-01-27T21:00:00.000Z", TRADING_DAYS)  # 정확히 16:00 EST
    assert effective == pd.Timestamp("2015-01-28")


def test_us_handles_summer_dst_correctly():
    """여름(EDT, UTC-4)에도 16:00 ET 마감 기준이 정확히 적용돼야 한다."""
    # 2015-07-15 19:59 UTC = 15:59 EDT(마감 전) -> 당일
    before_close = assign_us_effective_date("2015-07-15T19:59:00.000Z", pd.bdate_range("2015-07-01", "2015-08-15"))
    assert before_close == pd.Timestamp("2015-07-15")
    # 2015-07-15 20:01 UTC = 16:01 EDT(마감 후) -> 익일
    after_close = assign_us_effective_date("2015-07-15T20:01:00.000Z", pd.bdate_range("2015-07-01", "2015-08-15"))
    assert after_close == pd.Timestamp("2015-07-16")


def test_us_weekend_filing_rolls_to_next_monday():
    """드물지만 주말에 접수되는 공시도 다음 거래일(월요일)로 배정돼야 한다."""
    effective = assign_us_effective_date("2015-01-25T15:00:00.000Z", TRADING_DAYS)  # 일요일 오전 EST
    assert effective == pd.Timestamp("2015-01-26")


def test_raises_when_calendar_has_no_future_trading_day():
    short_calendar = pd.bdate_range("2015-01-01", "2015-01-07")
    with pytest.raises(ValueError):
        assign_kr_effective_date("20150107", short_calendar)


def _bdate_range_excluding(start, end, excluded_dates):
    days = pd.bdate_range(start, end)
    return days[~days.isin(pd.to_datetime(excluded_dates))]


def test_us_july3_early_close_pushes_afternoon_filing_to_next_day():
    """실제 사례(round 2 발견) — 2017-07-04(화)가 공휴일이라 7/3(월)이 조기마감(13:00)일인데,
    기존 16:00 기준으로는 TSLA/JPM 공시가 '당일'로 잘못 배정되고 있었다."""
    trading_days = _bdate_range_excluding("2017-06-01", "2017-07-15", ["2017-07-04"])
    tsla = assign_us_effective_date("2017-07-03T19:21:00.000Z", trading_days)  # 15:21 EDT
    jpm = assign_us_effective_date("2017-07-03T17:29:00.000Z", trading_days)  # 13:29 EDT
    assert tsla == pd.Timestamp("2017-07-05")
    assert jpm == pd.Timestamp("2017-07-05")


def test_us_july3_2019_early_close_matches_confirmed_case():
    """실제 사례 — 2019-07-03(수)은 NYSE가 명시적으로 13:00 조기마감을 공지한 날."""
    trading_days = _bdate_range_excluding("2019-06-01", "2019-07-15", ["2019-07-04"])
    vnom = assign_us_effective_date("2019-07-03T17:26:00.000Z", trading_days)  # 13:26 EDT
    assert vnom == pd.Timestamp("2019-07-05")


def test_us_early_close_day_still_allows_same_day_before_1pm():
    """조기마감일이라도 13:00 이전 접수는 여전히 당일 반영 가능해야 한다."""
    trading_days = _bdate_range_excluding("2017-06-01", "2017-07-15", ["2017-07-04"])
    effective = assign_us_effective_date("2017-07-03T16:00:00.000Z", trading_days)  # 12:00 EDT
    assert effective == pd.Timestamp("2017-07-03")


def test_us_early_close_day_exact_1pm_boundary_treated_as_after_close():
    """조기마감일의 정확히 13:00:00도 16:00 케이스와 동일하게 보수적으로 마감 후 취급해야 한다."""
    trading_days = _bdate_range_excluding("2017-06-01", "2017-07-15", ["2017-07-04"])
    effective = assign_us_effective_date("2017-07-03T17:00:00.000Z", trading_days)  # 정확히 13:00 EDT
    assert effective == pd.Timestamp("2017-07-05")


def test_us_regular_day_at_2pm_is_not_treated_as_after_close():
    """조기마감일이 아닌 평일엔 13:00~16:00 사이도 여전히 '당일'이어야 한다 — 회귀 방지."""
    trading_days = pd.bdate_range("2017-06-01", "2017-07-15")
    effective = assign_us_effective_date("2017-06-14T18:00:00.000Z", trading_days)  # 14:00 EDT, 평범한 수요일
    assert effective == pd.Timestamp("2017-06-14")


def test_us_black_friday_is_early_close():
    trading_days = _bdate_range_excluding("2017-11-01", "2017-12-01", ["2017-11-23"])  # 추수감사절(목)
    effective = assign_us_effective_date("2017-11-24T19:30:00.000Z", trading_days)  # 14:30 EST, 블랙프라이데이
    assert effective == pd.Timestamp("2017-11-27")


def test_us_christmas_eve_is_early_close():
    trading_days = _bdate_range_excluding("2015-12-01", "2016-01-05", ["2015-12-25"])
    effective = assign_us_effective_date("2015-12-24T19:30:00.000Z", trading_days)  # 14:30 EST
    assert effective == pd.Timestamp("2015-12-28")  # 12/25(금)는 휴장, 다음 거래일은 12/28(월)


def test_us_early_close_days_excludes_holidays_not_yet_covered_by_calendar():
    """캘린더가 그 해 12월까지 도달하지 못했으면(예: 8월까지만 있음), 캘린더의
    마지막 날짜를 '크리스마스 조기마감일'로 착각하면 안 된다 — round 2 리뷰에서
    실제 데이터(2026년, 캘린더가 8/10까지만 있던 상태)로 발견한 버그."""
    from src.data_collection.event_timing import _us_early_close_days_for_year

    trading_days = _bdate_range_excluding("2026-01-01", "2026-08-10", ["2026-07-03"])  # 12월 데이터 없음
    days = _us_early_close_days_for_year(2026, trading_days)
    assert pd.Timestamp("2026-08-10") not in days
    assert pd.Timestamp("2026-07-02") in days  # 7월은 캘린더 범위 안 -> 정상 계산됨


def test_us_early_close_shifts_when_holiday_falls_on_saturday():
    """2026년 7/4(토)라 관찰 휴장일이 7/3(금)으로 밀리고, 조기마감은 그 전날인
    7/2(목)가 된다 — NYSE 공식 캘린더로 확인된 사례. 하드코딩된 날짜 목록이었다면
    놓쳤을 케이스."""
    trading_days = _bdate_range_excluding("2026-06-01", "2026-07-15", ["2026-07-03"])
    effective = assign_us_effective_date("2026-07-02T19:30:00.000Z", trading_days)  # 15:30 EDT, 조기마감 후
    assert effective == pd.Timestamp("2026-07-06")  # 7/3(금, 관찰휴장)도 건너뛰고 다음 거래일(월)로
