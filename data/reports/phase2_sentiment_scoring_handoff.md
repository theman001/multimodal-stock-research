# Phase 2 모듈 3 (감성 스코어링) — 세션 인계 문서

생성 시각: 2026-08-12. 컨텍스트 관리를 위해 이 지점부터 새 세션으로 이어감.

## 지금까지 확정된 것 (재검토·재설계 불필요 — 그대로 실행)

### 1. 의존성
- `torch==2.13.0+cpu`, `transformers==5.15.0` 설치 완료, `requirements.txt`에 반영됨.
- GPU 없음(CPU 전용) — 사용자가 "시간은 얼마나 걸려도 상관없다"고 명시적으로 확인함. 속도 최적화보다 정확성 우선.

### 2. 원문 수집 (완료, 재사용만 하면 됨)
`src/data_collection/event_text.py`:
- `fetch_us_filing_text(cik, accession_number, primary_document, raw_dir, force=False) -> str`
  — SEC EDGAR Archives에서 primaryDocument + 텍스트성 exhibit(최대 5개)을 가져와 정제된 텍스트로 캐싱(`{raw_dir}/events/text/us_{accession_number}.txt`).
  - **파일럿 중 실제로 발견한 버그 4건, 전부 수정 완료**(디버깅 재시도 금지 — 이미 해결됨):
    1. 8-K 커버페이지의 정형 법률 문구가 512토큰을 거의 다 차지 → `_strip_us_cover_page_boilerplate()`로 "Item N" 지점부터 시작하도록 자름
    2. XBRL 자동생성 뷰어 페이지(`R1.htm`, `R2.htm`...)가 진짜 exhibit 자리를 밀어냄 → `_XBRL_VIEWER_PATTERN`으로 제외
    3. `{accession-no}.txt`(전체 제출묶음)가 실제 문서가 아니라 SGML 헤더/placeholder만 있음 → 제외
    4. `{accession-no}-index.html`(Filing Detail 페이지)도 네비게이션 텍스트뿐 → 제외
  - 위 3~4는 `_SEC_SYSTEM_FILE_PATTERN`으로 통합 제외 처리됨.
- `fetch_kr_disclosure_text(rcept_no, raw_dir, force=False) -> str`
  — DART `document.xml` API(rcept_no 기반)로 ZIP을 받아 **EUC-KR**로 디코딩 후 정제된 텍스트로 캐싱(`{raw_dir}/events/text/kr_{rcept_no}.txt`).
  - 일부 rcept_no는 원문 파일이 없음(status 014) → 정상 케이스로 빈 문자열 캐싱, 에러 아님.
- 두 함수 다 캐시 우선(이미 있으면 네트워크 호출 없이 파일에서 읽음), `_html_to_text()`로 script/style 제거 후 텍스트만 추출.
- 테스트: `tests/test_event_text.py` (전부 통과 상태).

### 3. 문장 분리 + 슬라이딩 윈도우 (완료, 재사용만 하면 됨)
`src/data_collection/event_sentiment.py`:
- `split_sentences(text, min_length=5) -> list[str]` — 마침표/느낌표/물음표 뒤 공백 기준 분리.
- `sliding_windows(sentences, window_size=3, stride=1) -> list[str]` — **사용자가 직접 지정한 방식**: 문장 6개면 (1,2,3)/(2,3,4)/(3,4,5)/(4,5,6)처럼 겹쳐서 묶음. 문장이 window_size 이하면 전체를 윈도우 1개로.
- 테스트: `tests/test_event_sentiment.py` (6건, 전부 통과).

### 4. 왜 문서 전체가 아니라 문장 단위+슬라이딩 윈도우인가 (재론의 불필요)
FinBERT(`yiyanghkust/finbert-tone`)로 실제 AAPL 8-K를 스코어링해본 결과, 512토큰 안에 실제 보도자료 내용("Apple Expands Capital Return Program... increase of 50 percent...")이 정상적으로 들어왔는데도 **문서 전체를 하나로 넣으면 Neutral 100%**가 나왔다. 반면 **같은 문장을 개별로 넣으면 Positive 87~99%**가 정확히 나왔다. FinBERT-tone은 Financial PhraseBank(개별 문장 데이터셋)로 파인튜닝된 모델이라 문서 단위 분류에 안 맞는다는 걸 실증적으로 확인함. 사용자와 논의 후 문장 단위 슬라이딩 윈도우(중첩) + 집계 방식으로 최종 확정됨 — 이 부분을 다시 검토하거나 "문서 전체 넣기"로 되돌리지 말 것.

### 5. FinBERT 로드 방법 (완료, 재사용만 하면 됨)
```python
from transformers import BertForSequenceClassification, BertTokenizerFast
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
tokenizer = BertTokenizerFast.from_pretrained('yiyanghkust/finbert-tone')
```
**주의**: `pipeline('text-classification', model='yiyanghkust/finbert-tone')`나 `AutoModel.from_pretrained(...)`는 **실패한다** — 이 모델의 `config.json`에 `model_type` 필드가 없어서 `AutoConfig`가 인식을 못 함(실제로 확인된 에러: `ValueError: Unrecognized model in yiyanghkust/finbert-tone`). 반드시 `BertForSequenceClassification`/`BertTokenizerFast`로 직접 클래스를 지정해서 우회할 것.
- 라벨: `model.config.id2label` = `{0: 'Neutral', 1: 'Positive', 2: 'Negative'}` (확인 완료).
- 추론 예시: `inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)` → `logits = model(**inputs).logits` → `torch.softmax(logits, dim=-1)`.

## 지금부터 해야 할 것 (실제 작업)

### Step 1 — KR-FinBERT 로드 확인
`snunlp/KR-FinBert-SC` 모델을 로드해본다. FinBERT(영어)와 같은 `model_type` 누락 문제가 있을 수 있으니, 먼저 `AutoModel`/`pipeline`으로 시도해보고 실패하면 위와 동일하게 `BertForSequenceClassification`/`BertTokenizerFast` 직접 로드로 우회한다. **라벨 순서가 FinBERT와 다를 수 있으니 `model.config.id2label`을 반드시 확인**하고 코드에 반영할 것(추측하지 말 것).

### Step 2 — 전체 스코어링 파이프라인 구현
새 모듈(예: `src/data_collection/event_sentiment.py`에 이어서 작성하거나 별도 모듈)에 다음을 구현:
1. 원문 텍스트 가져오기(`fetch_us_filing_text`/`fetch_kr_disclosure_text`, 이미 있음)
2. `split_sentences()` → `sliding_windows(window_size=3, stride=1)` (이미 있음)
3. 각 윈도우를 해당 언어 모델(US=FinBERT, KR=KR-FinBERT)로 스코어링
4. 윈도우들의 점수를 집계해 문서 하나의 감성 점수로 만든다 — 권장 방식: `mean(P(Positive) - P(Negative))` (윈도우별로 계산 후 평균), 범위 -1~1. 문장이 하나도 안 걸러지는 경우(빈 텍스트, status 014로 빈 문자열인 KR 공시 등) 처리도 필요(예: `None`/`NaN` 반환, 스코어링 시도하지 않음 — 있지도 않은 원문에 억지로 점수 매기지 말 것, Phase 1의 "모르면 버린다" 원칙과 동일).
5. **캐싱 필수** — 종목/공시 단위로 이미 만든 `fetch_*` 함수들의 캐시-우선 패턴을 그대로 따를 것(예: `{raw_dir}/events/sentiment/us_{accession_number}.json` 또는 parquet에 점수 + 윈도우 개수 등 저장). 3만여 건에 대해 추론을 반복 실행하는 건 시간 낭비이므로 재실행 시 캐시를 반드시 활용해야 한다.
6. 순수 로직(집계 함수 등)은 반드시 유닛테스트 작성.

### Step 3 — 소량 파일럿으로 실측 시간 확인
전체 실행 전에 예컨대 US 20건 + KR 20건 정도로 실제 걸리는 시간을 재서 전체 규모(US 7,944건 + KR 22,490건, 문장 수에 따라 문서당 윈도우 여러 개)로 확장했을 때 대략 얼마나 걸릴지 가늠해본다. **속도가 느려도 진행을 막을 필요는 없음** — 사용자가 이미 시간 제약 없음을 확인함. 참고용 정보 제공 목적.

### Step 4 — 전체 규모 실행
US 60종목 7,944건 + KR 60종목 22,490건 전체에 대해 원문 수집(아직 안 한 건들) + 문장 분리 + 슬라이딩 윈도우 + 스코어링 + 캐싱을 실행한다. 오래 걸리는 작업이므로 백그라운드로 돌리고 주기적으로(과도하게 자주 말고) 진행 상황을 확인할 것.

### Step 5 — 마무리
- 결과를 검증(스코어 분포가 한쪽으로 쏠리지 않는지, 몇 개 샘플을 실제로 원문과 대조해 방향이 맞는지 등 — 이번 세션에서 계속 해온 "실제로 맞는지 확인" 습관을 유지할 것).
- `progress_log.json` 갱신(이 문서와 같은 위치에 이어서 기록).
- `data/reports/` 에 결과 리포트 작성(기존 `phase2_event_collection_report.md`처럼 발견한 이슈·수정 내역 포함).
- **모듈 4(이벤트 피처 집계 + `features.parquet` 병합)는 시작하지 말 것** — 사용자가 모듈 3 결과를 검토한 뒤 원 세션에서 별도로 승인할 예정.

## 하지 말아야 할 것
- 문장 단위 슬라이딩 윈도우 설계를 다시 논의하거나 "역시 문서 전체가 낫지 않을까" 하며 되돌리지 말 것 — 이미 실증 데이터로 검증하고 사용자 승인까지 받은 결정.
- window_size/stride 값(3, 1)을 임의로 바꾸지 말 것 — 사용자가 직접 지정한 값.
- Claude API를 감성 스코어링에 쓰지 말 것(CLAUDE.md 원칙 — 로컬 모델로 전체 이력 커버, Claude API는 소규모 정성 비교용으로만 고려 대상).
- 모듈 1·2(이벤트 수집, 타임스탬프 정렬) 코드를 재검토/재작업하지 말 것 — 이미 검증 3라운드(1차 1회, 2차 4회, 3차 3회)를 거쳐 버그 5건 발견·수정 완료된 안정 상태.
