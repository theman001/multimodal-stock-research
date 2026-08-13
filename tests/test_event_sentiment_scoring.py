import json
from types import SimpleNamespace

import pytest
import torch

from src.data_collection import event_sentiment_scoring as ess


def test_polarity_indices_finds_by_name_case_insensitive_finbert_order():
    """영어 FinBERT 순서: {0:Neutral,1:Positive,2:Negative}."""
    id2label = {0: "Neutral", 1: "Positive", 2: "Negative"}
    pos_idx, neg_idx = ess._polarity_indices(id2label)
    assert (pos_idx, neg_idx) == (1, 2)


def test_polarity_indices_finds_by_name_case_insensitive_kr_finbert_order():
    """KR-FinBERT는 순서가 다르다: {0:negative,1:neutral,2:positive} — 인덱스
    하드코딩이 아니라 라벨 이름으로 찾아야 한다는 걸 검증."""
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    pos_idx, neg_idx = ess._polarity_indices(id2label)
    assert (pos_idx, neg_idx) == (2, 0)


def test_score_document_returns_none_score_for_empty_text():
    """원문이 없으면(빈 문자열) 점수를 매기지 않고 None을 반환한다 — 모델 호출도 없어야 함."""
    result = ess.score_document("", model=None, tokenizer=None)
    assert result == {"score": None, "num_sentences": 0, "num_windows": 0}


def test_score_document_aggregates_window_scores_as_mean(monkeypatch):
    """윈도우별 점수의 평균을 문서 점수로 집계한다."""
    monkeypatch.setattr(ess, "score_windows", lambda windows, model, tokenizer: [0.5, -0.1, 0.8])
    text = "One sentence here. Two sentence here. Three sentence here. Four sentence here."
    result = ess.score_document(text, model=object(), tokenizer=object(), window_size=3, stride=1)
    assert result["score"] == pytest.approx((0.5 - 0.1 + 0.8) / 3)
    assert result["num_sentences"] == 4
    assert result["num_windows"] == 2  # window=3, stride=1, 문장 4개 -> [1,2,3],[2,3,4]


def test_score_windows_returns_empty_list_for_empty_windows():
    assert ess.score_windows([], model=None, tokenizer=None) == []


def test_score_windows_chunks_into_mini_batches_and_preserves_order():
    """윈도우가 많은 문서(큰 exhibit 등)를 한 번에 거대 배치로 넣으면 CPU에서
    비정상적으로 느려지는 걸 파일럿에서 실제로 확인해서(846문장/844윈도우 문서)
    미니배치로 나눠 처리하게 했다 — 청크로 나눠도 결과는 전체를 한 번에 넣은
    것과 동일해야 한다(순서 보존, 값 동일)."""
    windows = [str(i) for i in range(10)]  # 각 윈도우를 자기 인덱스 문자열로 표현

    class _FakeTokenizer:
        def __call__(self, chunk, **kwargs):
            calls.append(len(chunk))
            return {"ids": torch.tensor([[int(w)] for w in chunk])}

    class _FakeModel:
        config = SimpleNamespace(id2label={0: "negative", 1: "neutral", 2: "positive"})

        def __call__(self, ids):
            n = ids.shape[0]
            logits = torch.zeros(n, 3)
            for row in range(n):
                logits[row, 2] = ids[row, 0].item()  # 인덱스가 클수록 positive 쏠림
            return SimpleNamespace(logits=logits)

    calls: list[int] = []
    scores = ess.score_windows(windows, model=_FakeModel(), tokenizer=_FakeTokenizer(), batch_size=3)

    assert calls == [3, 3, 3, 1]  # 10개를 batch_size=3으로 나누면 청크 4개
    assert len(scores) == 10
    # positive 로짓이 클수록 softmax 이후 P(positive)-P(negative)도 커야 하므로 단조증가
    assert scores == sorted(scores)


def test_score_us_filing_returns_cached_result_without_fetching(tmp_path, monkeypatch):
    """캐시가 있으면 원문 fetch/모델 로드를 아예 하지 않는다."""
    raw_dir = tmp_path
    cache_dir = raw_dir / "events" / "sentiment"
    cache_dir.mkdir(parents=True)
    cached = {"score": 0.42, "num_sentences": 3, "num_windows": 1}
    (cache_dir / "us_0001-15-000001.json").write_text(json.dumps(cached), encoding="utf-8")

    def _boom(*args, **kwargs):
        raise AssertionError("캐시가 있는데 fetch가 호출됨")

    monkeypatch.setattr(ess, "fetch_us_filing_text", _boom)
    monkeypatch.setattr(ess, "_load_us_model", _boom)

    result = ess.score_us_filing(cik=1, accession_number="0001-15-000001", primary_document="d.htm", raw_dir=raw_dir)
    assert result == cached


def test_score_kr_disclosure_returns_cached_result_without_fetching(tmp_path, monkeypatch):
    raw_dir = tmp_path
    cache_dir = raw_dir / "events" / "sentiment"
    cache_dir.mkdir(parents=True)
    cached = {"score": -0.2, "num_sentences": 5, "num_windows": 3}
    (cache_dir / "kr_20150102800003.json").write_text(json.dumps(cached), encoding="utf-8")

    def _boom(*args, **kwargs):
        raise AssertionError("캐시가 있는데 fetch가 호출됨")

    monkeypatch.setattr(ess, "fetch_kr_disclosure_text", _boom)
    monkeypatch.setattr(ess, "_load_kr_model", _boom)

    result = ess.score_kr_disclosure(rcept_no="20150102800003", raw_dir=raw_dir)
    assert result == cached
