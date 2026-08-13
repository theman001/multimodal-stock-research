"""S&P 500/400/600 point-in-time 구성종목 재구성 (해외 survivorship bias 완화).

Wikipedia의 "List of S&P {500,400,600} companies" 문서에는 현재 구성종목 표 외에
"Selected changes to the components"(변경 이력) 표가 있다. 이 이력을 과거→현재로
재생하면 임의 과거 시점의 구성종목을 재구성할 수 있다 — 01_data_collection.md에
문서화된 정확한 point-in-time 재구성 방법.

이력 표가 커버하는 기간(로그 시작일)은 지수마다 다르다(2026-08-11 기준 확인):
  - S&P 500: 1976년부터 (2015~ 전체 커버)
  - S&P 400: 2012년부터 (2015~ 전체 커버)
  - S&P 600: 2019-12부터 (2015-01~2019-12 구간은 이력 데이터가 없어 그 종목의
    S&P 600 소속 여부를 검증할 수 없음)

이력이 없는 구간은 "현재 구성종목이었다"고 추정하지 않고 라벨을 아예 비운다
(leakage-guard 원칙: 근거 없는 라벨보다 결측이 낫다). 그 결과 S&P 600 이력에만
의존하는 종목은 2019-12 이전 행이 최종 산출물에서 제외된다 — known limitation으로
문서화됨.
"""
from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from src.data_collection.us_universe import SP_INDEX_PAGES, _normalize_ticker, wiki_headers

_CITATION_SUFFIX = re.compile(r"\[\d+\]\s*$")
_MULTI_TICKER_SPLIT = re.compile(r"[,/]")


def _clean_date_str(raw: str) -> str:
    return _CITATION_SUFFIX.sub("", str(raw)).strip()


def _split_tickers(raw: str) -> list[str]:
    """일부 행은 복수클래스 종목이 한 셀에 같이 적혀있다 (예: 'UA/UAA',
    'CWEN, CWEN-A'). 쉼표/슬래시로 나눠 각각을 독립된 티커로 취급한다 —
    안 나누면 'UA/UAA' 같은 문자열 통째로 하나의 가짜 티커가 되어, 실제로는
    존재하는 UA/UAA 각각의 소속 이력이 조용히 유실된다.
    """
    return [_normalize_ticker(part) for part in _MULTI_TICKER_SPLIT.split(str(raw)) if part.strip()]


def fetch_changes_table(index_key: str, raw_dir: Path, force: bool = False) -> pd.DataFrame:
    """Wikipedia 변경 이력 표를 가져온다 (raw HTML은 us_universe.py와 같은 캐시 재사용).

    반환은 (date, added, removed) 롱 포맷이며, added/removed는 각각 최대 1개
    티커만 담는다 — 원본 셀에 여러 티커가 같이 적혀있으면 여러 행으로 펼친다.
    """
    cache_dir = raw_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"wikipedia_{index_key}.html"

    if cache_path.exists() and not force:
        html = cache_path.read_text(encoding="utf-8")
    else:
        resp = requests.get(SP_INDEX_PAGES[index_key], headers=wiki_headers(), timeout=20)
        resp.raise_for_status()
        html = resp.text
        cache_path.write_text(html, encoding="utf-8")

    tables = pd.read_html(StringIO(html))
    changes = tables[1]
    changes.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in changes.columns
    ]
    date_col, added_col, removed_col = changes.columns[0], changes.columns[1], changes.columns[3]

    rows = []
    for _, raw_row in changes.iterrows():
        date_str = _clean_date_str(raw_row[date_col])
        for ticker in _split_tickers(raw_row[added_col]) if pd.notna(raw_row[added_col]) else []:
            rows.append({"date": date_str, "added": ticker, "removed": None})
        for ticker in _split_tickers(raw_row[removed_col]) if pd.notna(raw_row[removed_col]) else []:
            rows.append({"date": date_str, "added": None, "removed": ticker})

    out = pd.DataFrame(rows, columns=["date", "added", "removed"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    return out


def build_membership_intervals(
    current_constituents: set[str], changes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """변경 이력을 재생해 종목별 소속 구간(들)을 재구성한다.

    반환: (intervals[ticker, start_date, end_date(NaT=현재까지 소속)], log_start_date)
    log_start_date 이전 시점의 소속 여부는 이 함수로 알 수 없다 — 호출부에서
    log_start_date 이전 구간은 라벨링 대상에서 제외해야 한다.
    """
    changes = changes.sort_values("date")
    log_start_date = changes["date"].min()

    # 1) 최신 -> 과거로 이력을 되돌려 log_start_date 시점(로그 시작 직전)의 구성종목 집합을 구한다.
    baseline = set(current_constituents)
    for _, row in changes.sort_values("date", ascending=False).iterrows():
        if row["added"]:
            baseline.discard(row["added"])
        if row["removed"]:
            baseline.add(row["removed"])

    # 2) 과거 -> 현재로 재생하며 각 종목의 소속 구간을 기록한다.
    open_intervals: dict[str, pd.Timestamp] = {t: log_start_date for t in baseline}
    closed: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    for _, row in changes.iterrows():
        d = row["date"]
        if row["removed"] and row["removed"] in open_intervals:
            start = open_intervals.pop(row["removed"])
            closed.append((row["removed"], start, d))
        if row["added"]:
            open_intervals[row["added"]] = d  # 재편입 케이스도 새 구간으로 정상 처리됨

    rows = [{"ticker": t, "start_date": s, "end_date": e} for t, s, e in closed]
    rows += [{"ticker": t, "start_date": s, "end_date": pd.NaT} for t, s in open_intervals.items()]
    intervals = pd.DataFrame(rows, columns=["ticker", "start_date", "end_date"])
    return intervals, log_start_date


def build_all_index_intervals(
    raw_dir: Path,
) -> dict[str, tuple[pd.DataFrame, pd.Timestamp]]:
    """S&P 500/400/600 각각의 point-in-time 소속 구간을 구축한다."""
    from src.data_collection.us_universe import fetch_index_constituents

    result = {}
    for index_key in ("sp500", "sp400", "sp600"):
        current = set(fetch_index_constituents(index_key, raw_dir))
        changes = fetch_changes_table(index_key, raw_dir)
        intervals, log_start = build_membership_intervals(current, changes)
        result[index_key] = (intervals, log_start)
    return result


def compute_coverage_days(
    ticker: str,
    index_intervals: dict[str, tuple[pd.DataFrame, pd.Timestamp]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    """[start, end] 구간 중 이 종목이 3개 지수 중 하나에라도 소속돼 있었다고
    검증되는 날짜 수(달력일)를 센다 — 유니버스 선정 시 "현재는 시총 상위지만
    point-in-time 이력이 거의 없는" 종목을 걸러내는 데 쓴다.
    """
    intervals_for_ticker = []
    for _, (intervals, log_start) in index_intervals.items():
        sub = intervals[intervals["ticker"] == ticker]
        for _, iv in sub.iterrows():
            iv_start = max(iv["start_date"], log_start, start)
            iv_end = min(iv["end_date"] if pd.notna(iv["end_date"]) else end, end)
            if iv_start <= iv_end:
                intervals_for_ticker.append((iv_start, iv_end))

    if not intervals_for_ticker:
        return 0

    intervals_for_ticker.sort()
    merged = [intervals_for_ticker[0]]
    for s, e in intervals_for_ticker[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + pd.Timedelta(days=1):
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    return sum((e - s).days + 1 for s, e in merged)


def label_point_in_time_us(
    ohlcv: pd.DataFrame, ticker: str, index_intervals: dict[str, tuple[pd.DataFrame, pd.Timestamp]]
) -> pd.DataFrame:
    """ohlcv(date 컬럼 포함, 단일 종목)에 point-in-time size_id를 부여한다.

    세 지수의 소속 구간을 모두 확인해 어느 시점에 어느 지수(대/중/소)에
    속했는지로 size_id를 정한다. 어느 지수에도 소속이 확인되지 않는 시점(특히
    S&P 600은 이력이 2019-12부터만 있음)의 행은 라벨을 붙이지 않고 제외한다.
    """
    size_id_by_index = {"sp500": 2, "sp400": 1, "sp600": 0}

    result = ohlcv.sort_values("date").copy()
    result["size_id"] = pd.NA

    for index_key, (intervals, log_start) in index_intervals.items():
        ticker_intervals = intervals[intervals["ticker"] == ticker]
        if ticker_intervals.empty:
            continue
        for _, iv in ticker_intervals.iterrows():
            start = max(iv["start_date"], log_start)
            end = iv["end_date"] if pd.notna(iv["end_date"]) else pd.Timestamp.max
            mask = (result["date"] >= start) & (result["date"] <= end) & result["size_id"].isna()
            result.loc[mask, "size_id"] = size_id_by_index[index_key]

    result = result.dropna(subset=["size_id"]).copy()
    result["size_id"] = result["size_id"].astype(int)
    return result
