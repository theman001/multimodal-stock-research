"""이벤트(공시/8-K) 원문 감성 스코어링 — 모듈 3.

FinBERT 계열 모델은 문장 단위(Financial PhraseBank)로 파인튜닝돼 있어서,
법률 boilerplate와 실제 보도자료가 섞인 512토큰짜리 긴 문서를 통째로 넣으면
신호가 뭉개져 전부 Neutral로 나온다(파일럿 검증 중 실제 AAPL 8-K로 확인 —
문장 단위로는 Positive 87~99%가 나오는 내용이 문서 전체로는 Neutral 100%가
됐음). 그래서 문장 단위로 쪼갠 뒤 슬라이딩 윈도우(윈도우 3문장, stride 1 —
사용자 지정)로 겹쳐 묶어 각각 스코어링하고, 그 결과를 평균해 문서 하나의
감성 점수로 집계한다.
"""
from __future__ import annotations

import re

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_MIN_SENTENCE_LENGTH = 5  # 너무 짧은 조각(표 데이터 파편 등)은 문장으로 안 침


def split_sentences(text: str, min_length: int = _MIN_SENTENCE_LENGTH) -> list[str]:
    """마침표/느낌표/물음표 뒤 공백 기준으로 문장을 나눈다.

    완벽한 문장 분리기는 아니다(한글/영어 공용의 단순 규칙 기반) — 표 데이터가
    섞이면 숫자 나열이 "문장"으로 잡히기도 하지만, 목적이 완벽한 NLU가 아니라
    통계적 톤 신호 추출이라 이 정도 노이즈는 허용 범위로 본다(Phase 1의
    51~55%를 현실적 한계로 받아들인 것과 같은 기준).
    """
    raw = _SENTENCE_SPLIT_PATTERN.split(text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= min_length]


def sliding_windows(sentences: list[str], window_size: int = 3, stride: int = 1) -> list[str]:
    """문장 리스트를 겹치는 윈도우로 묶는다 (사용자 지정: window=3, stride=1).

    예: 문장 6개, window=3, stride=1 -> [1,2,3],[2,3,4],[3,4,5],[4,5,6] (4개).
    문장 수가 window_size 이하면 전체를 윈도우 1개로 취급한다(짧은 공시 대응).
    """
    if not sentences:
        return []
    if len(sentences) <= window_size:
        return [" ".join(sentences)]
    return [
        " ".join(sentences[i : i + window_size])
        for i in range(0, len(sentences) - window_size + 1, stride)
    ]
