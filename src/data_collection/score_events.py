"""이벤트 원문 감성 스코어링 오케스트레이션 — 모듈 3.

`collect_events.py`가 만든 `processed/events_{us,kr}.parquet`의 각 행에
대해 원문을 가져와 감성 점수를 매기고(`event_sentiment_scoring.py`의
`score_us_filing`/`score_kr_disclosure`, 둘 다 캐시 우선) 결과를
`processed/events_sentiment_{us,kr}.parquet`에 저장한다.

문서 단위 캐시(`raw/{us,kr}/events/sentiment/*.json`)가 이미 재실행 시
API/모델 재호출을 막아주므로, 이 스크립트가 중간에 죽어도 재실행하면
이미 처리된 문서는 건너뛰고 나머지만 처리한다(추가 체크포인트 불필요 —
캐시 자체가 체크포인트 역할을 함). `progress_path`를 주면 N건마다
진행 상황(처리 건수/실패 건수)을 JSON으로 남겨 오래 걸리는 실행을
바깥에서 모니터링할 수 있게 한다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.config import get_data_root
from src.data_collection.event_sentiment_scoring import score_kr_disclosure, score_us_filing

_PROGRESS_EVERY = 200


def _write_progress(progress_path: Path | None, market: str, done: int, total: int, failed: int, started: float) -> None:
    if progress_path is None:
        return
    payload = {
        "market": market,
        "done": done,
        "total": total,
        "failed": failed,
        "elapsed_sec": round(time.time() - started, 1),
    }
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(progress_path)


def _load_cached_ids(cache_dir: Path, prefix: str) -> set[str]:
    """`{cache_dir}/{prefix}{id}.json` 캐시 파일 목록을 한 번에 읽어 id 집합으로 반환한다.

    Google Drive(FUSE 마운트)는 파일 하나마다 존재 확인(stat)을 하면 왕복이
    느려서(Colab 세션 재연결 직후 실측: 이미 캐시된 200건 확인에 158초) 수천 건
    규모에서 눈에 띄게 느려진다. 디렉터리 목록 조회 한 번으로 대체하면 파일당
    개별 `.exists()` 호출을 없앨 수 있다.
    """
    if not cache_dir.exists():
        return set()
    suffix_len = len(".json")
    return {p.name[len(prefix) : -suffix_len] for p in cache_dir.glob(f"{prefix}*.json")}


def score_events_us(
    data_root: Path | None = None, limit: int | None = None, progress_path: Path | None = None
) -> pd.DataFrame:
    data_root = data_root or get_data_root()
    raw_dir = data_root / "raw" / "us"
    events = pd.read_parquet(data_root / "processed" / "events_us.parquet")
    if limit:
        events = events.head(limit)

    cache_dir = raw_dir / "events" / "sentiment"
    cached_ids = _load_cached_ids(cache_dir, "us_")

    started = time.time()
    rows = []
    failed = []
    for i, (_, e) in enumerate(events.iterrows(), start=1):
        try:
            accession = e["accessionNumber"]
            if accession in cached_ids:
                result = json.loads((cache_dir / f"us_{accession}.json").read_text(encoding="utf-8"))
            else:
                result = score_us_filing(cik=e["cik"], accession_number=accession, primary_document=e["primaryDocument"], raw_dir=raw_dir)
            rows.append({"ticker": e["ticker"], "accessionNumber": accession, **result})
        except Exception as ex:
            failed.append((e["accessionNumber"], str(ex)))
        if i % _PROGRESS_EVERY == 0 or i == len(events):
            _write_progress(progress_path, "us", i, len(events), len(failed), started)
            print(f"[score_events_us] {i}/{len(events)} 처리 (실패 {len(failed)}건, {round(time.time() - started, 1)}s 경과)")

    if failed:
        print(f"[score_events_us] 실패 {len(failed)}건 (스킵, 원문/캐시 없으면 다음 실행 시 재시도됨): {failed[:10]}{'...' if len(failed) > 10 else ''}")
    if not rows:
        raise RuntimeError("US 감성 스코어링 결과가 비어 있음 — 전체 실패")

    combined = pd.DataFrame(rows)
    processed_dir = data_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "events_sentiment_us.parquet"
    combined.to_parquet(output_path, index=False)
    print(f"[score_events_us] {len(combined)}건 저장 -> {output_path}")
    return combined


def score_events_kr(
    data_root: Path | None = None, limit: int | None = None, progress_path: Path | None = None
) -> pd.DataFrame:
    data_root = data_root or get_data_root()
    raw_dir = data_root / "raw" / "kr"
    events = pd.read_parquet(data_root / "processed" / "events_kr.parquet")
    if limit:
        events = events.head(limit)

    cache_dir = raw_dir / "events" / "sentiment"
    cached_ids = _load_cached_ids(cache_dir, "kr_")

    started = time.time()
    rows = []
    failed = []
    for i, (_, e) in enumerate(events.iterrows(), start=1):
        try:
            rcept_no = e["rcept_no"]
            if rcept_no in cached_ids:
                result = json.loads((cache_dir / f"kr_{rcept_no}.json").read_text(encoding="utf-8"))
            else:
                result = score_kr_disclosure(rcept_no=rcept_no, raw_dir=raw_dir)
            rows.append({"ticker": e["ticker"], "rcept_no": rcept_no, **result})
        except Exception as ex:
            failed.append((e["rcept_no"], str(ex)))
        if i % _PROGRESS_EVERY == 0 or i == len(events):
            _write_progress(progress_path, "kr", i, len(events), len(failed), started)
            print(f"[score_events_kr] {i}/{len(events)} 처리 (실패 {len(failed)}건, {round(time.time() - started, 1)}s 경과)")

    if failed:
        print(f"[score_events_kr] 실패 {len(failed)}건 (스킵, 원문/캐시 없으면 다음 실행 시 재시도됨): {failed[:10]}{'...' if len(failed) > 10 else ''}")
    if not rows:
        raise RuntimeError("KR 감성 스코어링 결과가 비어 있음 — 전체 실패")

    combined = pd.DataFrame(rows)
    processed_dir = data_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "events_sentiment_kr.parquet"
    combined.to_parquet(output_path, index=False)
    print(f"[score_events_kr] {len(combined)}건 저장 -> {output_path}")
    return combined


if __name__ == "__main__":
    data_root = get_data_root()
    score_events_us(data_root, progress_path=data_root / "reports" / "score_events_us_progress.json")
    score_events_kr(data_root, progress_path=data_root / "reports" / "score_events_kr_progress.json")
