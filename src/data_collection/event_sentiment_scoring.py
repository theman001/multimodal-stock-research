"""이벤트 원문 감성 스코어링 — 모델 로딩 + 추론 + 캐싱 (모듈 3).

문장 단위/슬라이딩 윈도우 로직(순수 함수, 무거운 ML 의존성 없음)은
`event_sentiment.py`에 있다. 이 모듈은 실제 FinBERT/KR-FinBERT 모델을
로드해 윈도우들을 스코어링하고, 문서 단위로 집계한 뒤 원문 캐싱과 동일한
캐시-우선 패턴으로 결과를 저장한다.

라벨 공간이 모델마다 다르다(직접 확인 완료):
- 영어 FinBERT(`yiyanghkust/finbert-tone`): {0: Neutral, 1: Positive, 2: Negative}
  — `AutoConfig`가 `model_type`을 인식 못 해 `pipeline()`/`AutoModel`이 실패한다.
  `BertForSequenceClassification`/`BertTokenizerFast`로 직접 로드해야 한다.
- 한국어 KR-FinBERT(`snunlp/KR-FinBert-SC`): {0: negative, 1: neutral, 2: positive}
  — 이건 `AutoModelForSequenceClassification`/`AutoTokenizer`로 정상 로드된다.
순서가 다르므로 인덱스를 하드코딩하지 않고 `id2label`에서 이름으로 찾는다.

집계 방식: 윈도우별 `P(Positive) - P(Negative)`를 계산해 평균한다(-1~1).
원문이 비었거나(예: DART status 014로 캐싱된 빈 문자열) 문장이 하나도 안
걸러지면 점수를 매기지 않고 `score=None`을 반환한다 — Phase 1의
"모르면 버린다" 원칙과 동일(handoff 문서 Step 2 참고).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizerFast,
)

from src.data_collection.event_sentiment import sliding_windows, split_sentences
from src.data_collection.event_text import fetch_kr_disclosure_text, fetch_us_filing_text

US_MODEL_NAME = "yiyanghkust/finbert-tone"
KR_MODEL_NAME = "snunlp/KR-FinBert-SC"

# int8 동적 양자화(onednn 엔진)는 CPU 전용 최적화다(로컬 4코어 환경에서 마련한 것).
# GPU(CUDA)가 있으면 양자화 없이 fp32로 GPU에 올리는 편이 훨씬 빠르고, 애초에
# 양자화된 커널은 CUDA 텐서로 옮길 수 없다(onednn 백엔드가 CPU 전용). 그래서
# 디바이스를 감지해서 분기한다 — 스코어링 방법론(윈도우/집계)은 그대로 두고
# 추론 실행 위치·정밀도만 환경에 맞춘다.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _quantize_for_cpu(model):
    """Linear 레이어를 int8 동적 양자화한다 — CPU 코어가 4개뿐인 환경에서
    (파일럿 실측 결과 US 7,944건 전체 실행 시 약 10일 예상) 프로세스
    병렬화는 코어 경합만 생길 뿐 이득이 없어서, 스코어링 자체를 계산량
    측면에서 가볍게 만드는 쪽으로 접근했다. 가중치만 int8로 바꾸는 post-training
    dynamic quantization이라 정확도 손실이 거의 없다(직접 확인: 동일 문장
    fp32 vs quantized 확률이 소수점 7자리까지만 다름). window_size/stride/
    문장 분리 등 스코어링 "방법론"은 전혀 건드리지 않고 추론 실행 방식만
    바꾼다.

    quantized 엔진은 명시적으로 골라야 한다 — 기본 엔진(x86/fbgemm)은
    이 머신에서 벤치마크해보니 1.15배밖에 안 빨라지는데, `onednn`으로
    바꾸면 1.94배가 나온다(실제 측정: 96개 윈도우 배치 추론 fp32 19.07s ->
    onednn 9.82s). PyTorch 기본값이 최선이 아닐 수 있으니 추측하지 않고
    엔진별로 직접 재봤다.
    """
    torch.backends.quantized.engine = "onednn"
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


@lru_cache(maxsize=1)
def _load_us_model():
    """FinBERT(영어). config.json에 model_type이 없어 AutoModel/pipeline이
    실패하므로(실제로 확인된 에러) Bert 클래스를 직접 지정해서 로드한다."""
    model = BertForSequenceClassification.from_pretrained(US_MODEL_NAME)
    tokenizer = BertTokenizerFast.from_pretrained(US_MODEL_NAME)
    model.eval()
    if DEVICE.type == "cuda":
        model = model.to(DEVICE)
    else:
        model = _quantize_for_cpu(model)
    return model, tokenizer


@lru_cache(maxsize=1)
def _load_kr_model():
    """KR-FinBERT. FinBERT(영어)와 달리 AutoModel 계열로 정상 로드된다(확인 완료)."""
    model = AutoModelForSequenceClassification.from_pretrained(KR_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(KR_MODEL_NAME)
    model.eval()
    if DEVICE.type == "cuda":
        model = model.to(DEVICE)
    else:
        model = _quantize_for_cpu(model)
    return model, tokenizer


def _polarity_indices(id2label: dict[int, str]) -> tuple[int, int]:
    """id2label에서 positive/negative 라벨 인덱스를 이름으로 찾는다(대소문자 무관).

    FinBERT(영어)는 {0:Neutral,1:Positive,2:Negative}, KR-FinBERT는
    {0:negative,1:neutral,2:positive}로 순서가 달라서 인덱스 하드코딩 금지.
    """
    lower = {label.lower(): idx for idx, label in id2label.items()}
    return lower["positive"], lower["negative"]


# 첨부가 큰 filing(계약서 등)은 윈도우가 수백~수천 개까지 나온다 — 전부 한 번에
# 배치로 넣으면 가장 긴 윈도우 기준으로 패딩된 거대 텐서가 생겨(파일럿 중 실제로
# 846문장/844윈도우짜리 8-K exhibit에서 확인) CPU에서 비정상적으로 느려진다.
# 윈도우 구성/집계 수식(평균)은 그대로 두고, 추론만 작은 묶음으로 나눠 처리한다 —
# 결과는 수학적으로 전체를 한 번에 넣은 것과 동일하다(순서만 배치 단위로 나뉠 뿐).
# GPU는 VRAM이 넉넉하고 병렬 처리량이 커서 CPU용 16보다 훨씬 큰 배치가 유리하다.
# L4(24GB VRAM) 기준 128은 BERT-base(fp32, seq_len<=512)에 여유 있게 들어간다 —
# 윈도우가 수백~천 개인 큰 exhibit 문서일수록 배치가 클수록 이득이 커진다.
_INFERENCE_BATCH_SIZE = 128 if DEVICE.type == "cuda" else 16


@torch.no_grad()
def score_windows(windows: list[str], model, tokenizer, batch_size: int = _INFERENCE_BATCH_SIZE) -> list[float]:
    """윈도우들을 미니배치로 나눠 추론해 각각 P(Positive) - P(Negative)를 반환한다."""
    if not windows:
        return []
    pos_idx, neg_idx = _polarity_indices(model.config.id2label)
    scores: list[float] = []
    for i in range(0, len(windows), batch_size):
        chunk = windows[i : i + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        scores.extend((probs[:, pos_idx] - probs[:, neg_idx]).tolist())
    return scores


def score_document(text: str, model, tokenizer, window_size: int = 3, stride: int = 1) -> dict:
    """원문 하나를 문장분리 -> 슬라이딩 윈도우 -> 스코어링 -> 평균 집계한다."""
    sentences = split_sentences(text)
    if not sentences:
        return {"score": None, "num_sentences": 0, "num_windows": 0}
    windows = sliding_windows(sentences, window_size=window_size, stride=stride)
    scores = score_windows(windows, model, tokenizer)
    return {
        "score": sum(scores) / len(scores),
        "num_sentences": len(sentences),
        "num_windows": len(windows),
    }


def score_us_filing(
    cik: int, accession_number: str, primary_document: str, raw_dir: Path, force: bool = False
) -> dict:
    """US 8-K 원문을 (필요시) 가져와 감성 점수를 매기고 캐싱한다 (캐시 우선).

    `force`는 감성 점수 캐시에만 적용된다 — 원문 텍스트는 이미
    `fetch_us_filing_text`가 자체 캐시를 갖고 있으므로 그대로 재사용한다.
    """
    cache_dir = raw_dir / "events" / "sentiment"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"us_{accession_number}.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    text = fetch_us_filing_text(cik, accession_number, primary_document, raw_dir)
    model, tokenizer = _load_us_model()
    result = score_document(text, model, tokenizer)
    cache_path.write_text(json.dumps(result), encoding="utf-8")
    return result


def score_kr_disclosure(rcept_no: str, raw_dir: Path, force: bool = False) -> dict:
    """KR 공시 원문을 (필요시) 가져와 감성 점수를 매기고 캐싱한다 (캐시 우선)."""
    cache_dir = raw_dir / "events" / "sentiment"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"kr_{rcept_no}.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    text = fetch_kr_disclosure_text(rcept_no, raw_dir)
    model, tokenizer = _load_kr_model()
    result = score_document(text, model, tokenizer)
    cache_path.write_text(json.dumps(result), encoding="utf-8")
    return result
