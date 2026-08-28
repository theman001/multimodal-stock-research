import json

import pandas as pd

from src.data_collection import score_events


def test_load_cached_ids_returns_empty_set_for_missing_dir(tmp_path):
    assert score_events._load_cached_ids(tmp_path / "nope", "us_") == set()


def test_load_cached_ids_strips_prefix_and_suffix(tmp_path):
    cache_dir = tmp_path / "sentiment"
    cache_dir.mkdir()
    (cache_dir / "us_0001-15-000001.json").write_text("{}", encoding="utf-8")
    (cache_dir / "us_0002-15-000002.json").write_text("{}", encoding="utf-8")
    assert score_events._load_cached_ids(cache_dir, "us_") == {"0001-15-000001", "0002-15-000002"}


def test_score_events_us_reads_cache_directly_without_calling_score_us_filing(tmp_path, monkeypatch):
    """Google Drive처럼 파일당 존재확인(stat)이 느린 환경에서, 이미 캐시된 건은
    score_us_filing()을 호출하지 않고(내부에서 또 .exists() 체크를 하므로) 캐시
    디렉터리에서 직접 읽어야 한다 — 실측으로 이 경로가 느린 걸 확인해서 추가."""
    data_root = tmp_path
    raw_dir = data_root / "raw" / "us"
    processed_dir = data_root / "processed"
    cache_dir = raw_dir / "events" / "sentiment"
    cache_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    cached_result = {"score": 0.5, "num_sentences": 3, "num_windows": 1}
    (cache_dir / "us_0001-15-000001.json").write_text(json.dumps(cached_result), encoding="utf-8")

    pd.DataFrame(
        [{"ticker": "AAPL", "cik": 1, "accessionNumber": "0001-15-000001", "primaryDocument": "d.htm"}]
    ).to_parquet(processed_dir / "events_us.parquet", index=False)

    def _boom(**kwargs):
        raise AssertionError("캐시가 있는데 score_us_filing이 호출됨")

    monkeypatch.setattr(score_events, "score_us_filing", _boom)

    result = score_events.score_events_us(data_root=data_root)
    assert len(result) == 1
    assert result.iloc[0]["score"] == 0.5


def test_score_events_us_uses_committed_parquet_when_json_cache_missing(tmp_path, monkeypatch):
    """per-doc JSON 캐시(raw/.../sentiment/*.json)는 git에 커밋되지 않아 새
    환경에선 비어 있다 — 그럴 때 커밋된 events_sentiment_us.parquet 자체를
    캐시로 써서, 이미 스코어링된 문서를 FinBERT로 재계산하지 않아야 한다
    (안 그러면 매일 07:00 cron이 전체 7,900여 건을 재스코어링해 몇 시간 걸림)."""
    data_root = tmp_path
    processed_dir = data_root / "processed"
    processed_dir.mkdir(parents=True)
    # JSON 캐시 디렉터리는 아예 없음 (새 clone 상태)

    pd.DataFrame(
        [
            {"ticker": "AAPL", "cik": 1, "accessionNumber": "old-1", "primaryDocument": "d.htm"},
            {"ticker": "AAPL", "cik": 1, "accessionNumber": "new-1", "primaryDocument": "d.htm"},
        ]
    ).to_parquet(processed_dir / "events_us.parquet", index=False)
    pd.DataFrame(
        [{"ticker": "AAPL", "accessionNumber": "old-1", "score": 0.3, "num_sentences": 4, "num_windows": 2}]
    ).to_parquet(processed_dir / "events_sentiment_us.parquet", index=False)

    calls = []

    def _fake_score(*, cik, accession_number, primary_document, raw_dir):
        calls.append(accession_number)
        return {"score": -0.1, "num_sentences": 2, "num_windows": 1}

    monkeypatch.setattr(score_events, "score_us_filing", _fake_score)

    result = score_events.score_events_us(data_root=data_root)
    assert calls == ["new-1"]  # 신규 1건만 모델 호출, 기존 건은 parquet에서 재사용
    assert len(result) == 2
    assert result.set_index("accessionNumber").loc["old-1", "score"] == 0.3


def test_score_events_kr_reads_cache_directly_without_calling_score_kr_disclosure(tmp_path, monkeypatch):
    data_root = tmp_path
    raw_dir = data_root / "raw" / "kr"
    processed_dir = data_root / "processed"
    cache_dir = raw_dir / "events" / "sentiment"
    cache_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    cached_result = {"score": -0.2, "num_sentences": 5, "num_windows": 2}
    (cache_dir / "kr_20150102800003.json").write_text(json.dumps(cached_result), encoding="utf-8")

    pd.DataFrame([{"ticker": "000270", "corp_code": "00106641", "rcept_no": "20150102800003"}]).to_parquet(
        processed_dir / "events_kr.parquet", index=False
    )

    def _boom(**kwargs):
        raise AssertionError("캐시가 있는데 score_kr_disclosure가 호출됨")

    monkeypatch.setattr(score_events, "score_kr_disclosure", _boom)

    result = score_events.score_events_kr(data_root=data_root)
    assert len(result) == 1
    assert result.iloc[0]["score"] == -0.2
