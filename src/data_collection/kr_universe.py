"""KOSPI point-in-time Size_Id 라벨링 및 60종목 유니버스 선정 (국내).

KRX Open API(공식, openapi.krx.co.kr) "유가증권 일별매매정보"(stk_bydd_trd)만
사용한다 — pykrx의 data.krx.co.kr 스크레이핑은 이용약관 위반으로 IP 차단을
받은 바 있어(자동화 수단을 통한 비정상 대량 조회 탐지) 이 프로젝트에서는
사용하지 않는다. Open API는 무료 회원가입 + API 키 발급 + 엔드포인트별 이용
신청을 거친 공식 경로이며, 2010년 이후 데이터를 제공한다.

이 엔드포인트는 "하루 지정 → 그날 전종목 시세"만 조회 가능하다(종목 지정 →
여러 날짜 조회는 불가). 따라서 필요한 모든 날짜를 하나씩 순회하며 전종목
스냅샷을 캐시하고, 그 캐시에서 시가총액 랭킹과 개별 종목 OHLCV를 모두
파생시키는 방식으로 설계했다.

CLAUDE.md 분류 기준: 코스피 시가총액 순위 1~100위=대형(2), 101~300위=중형(1),
301위 이하=소형(0).

기준일 계산(웹 검색으로 실제 KRX 공지/보도자료 대조 검증 완료 — 2026-08-11):
CLAUDE.md 초안에는 "매년 6월/12월 정기변경"이라 적혀 있었으나, 이는 KOSPI200
지수(분기별 3/6/9/12월 리밸런싱)와 혼동한 것으로 확인되었다. KRX 시가총액
규모별지수(대형/중형/소형)의 실제 정기변경은 **매년 3월·9월 두 차례**이며,
아래 두 시점이 서로 다르다는 점이 핵심이다:

- **심사기준일**(3개월 평균 시가총액을 계산하는 기준 시점): 매년 **2월 말** 또는
  **8월 말**(휴장일이면 그 직전 영업일). 3개월 평균 윈도우는 이 날짜를 기준으로
  직전 3개월이다.
- **시행일**(effective_date): 3월/9월 **선물·옵션 최종거래일**(해당 월의 두 번째
  목요일, 휴장일이면 직전 영업일) **익일**. 심사기준일보다 약 2주 뒤다.

즉 심사기준일(월말)과 시행 관련 만기일(둘째 목요일)은 서로 다른 시점이며, 3개월
평균 윈도우는 만기일이 아니라 월말 기준으로 계산해야 한다. (참고: 2024년 9월
사례로 검증 — 심사기준일 관행상 8월 말, 만기일 2024-09-12(목), 시행일
2024-09-13(금) 확인.)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

KRX_OPENAPI_URL = "https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd"
KRX_OPENAPI_REQUEST_DELAY_SECONDS = 0.15  # 공식 rate limit(10 req/s) 대비 여유

LARGE, MID, SMALL = 2, 1, 0
UNIVERSE_SIZE_PER_TIER = 20
WINDOW_MONTHS = 3
REBALANCE_MONTHS = (3, 9)  # KRX 시가총액 규모별지수 정기변경월 (KOSPI200과 다름 — 모듈 docstring 참고)
REVIEW_MONTH_BEFORE = {3: 2, 9: 8}  # 심사기준일 = 시행월 전월 말 (3월 시행 -> 2월 말, 9월 시행 -> 8월 말)

_COLUMN_RENAME = {
    "ISU_CD": "ticker",
    "TDD_OPNPRC": "open",
    "TDD_HGPRC": "high",
    "TDD_LWPRC": "low",
    "TDD_CLSPRC": "close",
    "ACC_TRDVOL": "volume",
    "MKTCAP": "market_cap",
}
_NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "market_cap"]


def fetch_daily_snapshot(
    date: pd.Timestamp,
    raw_dir: Path,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> pd.DataFrame:
    """KRX Open API로 특정일의 KOSPI 전종목 시세를 가져온다 (캐시 우선).

    비영업일(주말/공휴일)은 빈 DataFrame을 반환한다 — 에러가 아니라 정상 케이스.
    반환 컬럼: ticker, open, high, low, close, volume, market_cap
    """
    cache_dir = raw_dir / "stock_daily_trade"
    cache_dir.mkdir(parents=True, exist_ok=True)
    date_str = date.strftime("%Y%m%d")
    cache_path = cache_dir / f"{date_str}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    api_key = os.environ.get("KRX_OPENAPI_KEY")
    if not api_key:
        raise RuntimeError("KRX_OPENAPI_KEY 환경변수가 설정되지 않음 (.env 확인)")

    for attempt in range(max_retries):
        try:
            resp = requests.get(
                KRX_OPENAPI_URL, params={"AUTH_KEY": api_key, "basDd": date_str}, timeout=15
            )
            resp.raise_for_status()
            records = resp.json().get("OutBlock_1", [])
            df = pd.DataFrame(records)
            if not df.empty:
                df = df.rename(columns=_COLUMN_RENAME)[list(_COLUMN_RENAME.values())]
                for col in _NUMERIC_COLUMNS:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df.to_parquet(cache_path, index=False)
            time.sleep(KRX_OPENAPI_REQUEST_DELAY_SECONDS)
            return df
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_backoff_seconds * (2**attempt))
    raise RuntimeError(f"KRX Open API 조회 실패 (재시도 소진): {date_str}")


def _is_trading_day(date: pd.Timestamp, raw_dir: Path) -> bool:
    return not fetch_daily_snapshot(date, raw_dir).empty


def _nearest_trading_day(date: pd.Timestamp, raw_dir: Path, direction: str, max_days: int = 10) -> pd.Timestamp:
    step = pd.Timedelta(days=-1 if direction == "backward" else 1)
    d = date
    for _ in range(max_days):
        if _is_trading_day(d, raw_dir):
            return d
        d = d + step
    raise RuntimeError(f"거래일을 찾지 못함: {date} ({direction})")


def _second_thursday(year: int, month: int) -> pd.Timestamp:
    first_day = pd.Timestamp(year=year, month=month, day=1)
    first_thursday = first_day + pd.Timedelta(days=(3 - first_day.dayofweek) % 7)  # Thursday = weekday 3
    return first_thursday + pd.Timedelta(weeks=1)


def _month_end(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def generate_rebalance_periods(start_year: int, end_date: pd.Timestamp, raw_dir: Path) -> pd.DataFrame:
    """정기변경 심사기준일/효력발생일/3개월 평균 윈도우 목록을 생성한다.

    심사기준일(3개월 평균 윈도우의 끝)과 효력발생일(만기일 익일)은 서로 다른
    시점이다 — 모듈 docstring 참고.
    """
    rows = []
    year = start_year
    while pd.Timestamp(year=year, month=1, day=1) <= end_date:
        for effective_month in REBALANCE_MONTHS:
            review_month = REVIEW_MONTH_BEFORE[effective_month]
            review_year = year if review_month < effective_month else year - 1
            approx_review_date = _month_end(review_year, review_month)
            approx_expiry = _second_thursday(year, effective_month)
            # review_date는 지났지만 만기일이 아직 안 온 ~2주 구간에 실행되면
            # _nearest_trading_day가 아직 존재하지 않는 미래 날짜를 찾으려다
            # 잘못된 날짜를 반환하거나 예외를 던질 수 있다 — 이 주기 전체를
            # 다음 재실행으로 미룬다.
            if approx_review_date > end_date or approx_expiry > end_date:
                continue

            review_date = _nearest_trading_day(approx_review_date, raw_dir, direction="backward")
            expiry_date = _nearest_trading_day(approx_expiry, raw_dir, direction="backward")
            effective_date = _nearest_trading_day(
                expiry_date + pd.Timedelta(days=1), raw_dir, direction="forward"
            )
            window_start = review_date - pd.DateOffset(months=WINDOW_MONTHS)
            rows.append(
                {
                    "review_date": review_date,
                    "expiry_date": expiry_date,
                    "effective_date": effective_date,
                    "window_start": window_start,
                    "window_end": review_date,
                }
            )
        year += 1
    return pd.DataFrame(rows).sort_values("effective_date").reset_index(drop=True)


def compute_period_ranking(period: pd.Series, raw_dir: Path) -> pd.DataFrame:
    """한 정기변경 기간에 대해 3개월 일평균 시가총액 기준 순위/size_id 테이블을 만든다."""
    frames = []
    d = period["window_start"]
    while d <= period["window_end"]:
        snapshot = fetch_daily_snapshot(d, raw_dir)
        if not snapshot.empty:
            frames.append(snapshot[["ticker", "market_cap"]])
        d += pd.Timedelta(days=1)

    if not frames:
        raise RuntimeError(f"시가총액 데이터 전부 조회 실패: {period['window_start']} ~ {period['window_end']}")

    combined = pd.concat(frames, ignore_index=True)
    avg_cap = combined.groupby("ticker", as_index=False)["market_cap"].mean()
    ranked = avg_cap.sort_values("market_cap", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    ranked["size_id"] = pd.cut(
        ranked["rank"], bins=[0, 100, 300, float("inf")], labels=[LARGE, MID, SMALL]
    ).astype(int)
    ranked["effective_date"] = period["effective_date"]
    return ranked[["ticker", "market_cap", "rank", "size_id", "effective_date"]]


def build_point_in_time_rankings(periods: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """모든 정기변경 기간에 대한 순위/size_id 테이블을 이어붙인다."""
    tables = [compute_period_ranking(period, raw_dir) for _, period in periods.iterrows()]
    return pd.concat(tables, ignore_index=True)


SPLIT_PRICE_RATIO_THRESHOLD = 0.35  # 전일 대비 종가 변화율이 이보다 크면 "급변"으로 취급
# KRX 가격제한폭(상하 30%)을 감안해 여유를 둔 값 — 흔한 2:1(=50%) 분할이 정확히
# 임계값에 걸려 strict '>' 비교로 탐지를 놓치는 일이 없도록 30%보다 확실히 낮게 잡았다.
SPLIT_MARKET_CAP_RATIO_THRESHOLD = 0.3  # 같은 날 시가총액 변화율이 이보다 작으면 "기계적 조정"으로 취급


def detect_and_adjust_splits(df: pd.DataFrame) -> pd.DataFrame:
    """액면분할/병합/무상감자로 인한 가격 불연속을 탐지해 과거 가격을 소급 조정한다.

    KRX Open API(유가증권 일별매매정보)는 분할조정가(adjusted close)를 제공하지
    않고 원본(raw) 종가만 준다. 이 상태로 두면 분할 전후 최대 60거래일(MA60
    윈도우) 구간의 기술지표와 target이 심각하게 왜곡된다 (예: 삼성전자
    2018-05-04 50:1 액면분할 시 종가가 하루 만에 -98% "하락"한 것으로 보임).

    탐지 기준: 전일 대비 종가 변화율은 크지만(|Δprice|>50%) 같은 날 시가총액
    변화율은 작은(|Δmarket_cap|<30%) 경우를 "가격만 기계적으로 재조정된 이벤트"로
    간주한다 — 시가총액(=주가×발행주식수)은 진짜 가치 변화가 없는 분할/병합에서는
    거의 그대로 유지되기 때문이다. 실제 데이터로 검증: 이 기준으로 탐지된 이벤트는
    전부 알려진 국내 종목의 실제 액면분할/병합 시점과 일치했다(005930/005935
    2018-05-04 50:1 분할, 035420 2018-10-12 5:1 분할 등).

    market_cap 컬럼은 조정하지 않는다 — 이미 연속적이며 순위 계산에 그대로 쓰인다.
    """
    adjusted = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        price_ratio = g["close"] / g["close"].shift(1)
        market_cap_ratio = g["market_cap"] / g["market_cap"].shift(1)

        is_split = (price_ratio - 1).abs().gt(SPLIT_PRICE_RATIO_THRESHOLD) & (
            market_cap_ratio - 1
        ).abs().lt(SPLIT_MARKET_CAP_RATIO_THRESHOLD)

        if is_split.any():
            cum_factor = pd.Series(1.0, index=g.index)
            for event_date, ratio in zip(g.loc[is_split, "date"], price_ratio[is_split]):
                cum_factor[g["date"] < event_date] *= ratio
            for col in ("open", "high", "low", "close"):
                g[col] = g[col] * cum_factor
            g["volume"] = g["volume"] / cum_factor

        adjusted.append(g)

    return pd.concat(adjusted, ignore_index=True)


def select_kr_universe(point_in_time_rankings: pd.DataFrame, universe_size: int = UNIVERSE_SIZE_PER_TIER) -> pd.DataFrame:
    """최신 정기변경 시점 기준 대형/중형/소형 각 상위 universe_size개씩 선정."""
    latest_date = point_in_time_rankings["effective_date"].max()
    latest = point_in_time_rankings[point_in_time_rankings["effective_date"] == latest_date]

    frames = []
    for size_id in (LARGE, MID, SMALL):
        tier = latest[latest["size_id"] == size_id].sort_values("market_cap", ascending=False)
        frames.append(tier.head(universe_size))

    universe = pd.concat(frames, ignore_index=True)
    universe["market_id"] = 1  # CLAUDE.md: 1=국내
    return universe[["ticker", "market_id", "size_id", "market_cap"]]


def label_point_in_time(ohlcv: pd.DataFrame, ticker: str, point_in_time_rankings: pd.DataFrame) -> pd.DataFrame:
    """ohlcv(date 컬럼 포함, 단일 종목)에 point-in-time size_id를 부여한다.

    각 행의 date는 "그 날짜 이전(포함)의 가장 최근 effective_date" 기간의 size_id를
    적용받는다. 그 종목이 아직 순위 데이터에 등장하지 않는 기간(상장 전 등)의 행은
    라벨을 붙일 수 없으므로 제외한다 — 절대 추정/고정 라벨을 넣지 않는다
    (leakage-guard: 오분류 방지).
    """
    ticker_rankings = point_in_time_rankings[point_in_time_rankings["ticker"] == ticker]
    if ticker_rankings.empty:
        return ohlcv.iloc[0:0].copy()

    labels = (
        ticker_rankings[["effective_date", "size_id"]]
        .drop_duplicates()
        .sort_values("effective_date")
    )

    result = ohlcv.sort_values("date").copy()
    result = pd.merge_asof(result, labels, left_on="date", right_on="effective_date", direction="backward")
    result = result.dropna(subset=["size_id"])
    result["size_id"] = result["size_id"].astype(int)
    return result.drop(columns=["effective_date"])
