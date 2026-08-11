# Multimodal Stock Research

시장 카테고리(국내/해외) × 기업 규모 카테고리(대형/중소형)를 메타 특성(`Market_Id`, `Size_Id`)으로 주입해, 기술적 지표와 뉴스 이벤트의 통계적 유의성이 케이스별로 어떻게 달라지는지 검증하는 리서치 파이프라인입니다.

에이전트(Claude Code) 운영 규칙과 데이터 분류 기준, 데이터 누수 방지 원칙은 [CLAUDE.md](./CLAUDE.md)에 있습니다 — 코드를 작성하기 전에 반드시 그 파일을 기준으로 삼으세요.

## Phase Roadmap

- **Phase 1 (진행 중)**: 차트 중심 조건부 연결 학습 MVP
  - 국내(KOSPI, pykrx)/해외(S&P 500·400·600, yfinance) 데이터 수집 + `Market_Id`/`Size_Id` 메타 태그 결합
  - 데이터 누수 없는 시계열 분할(Train 80% / Test 20%) 및 모델 학습
  - 수수료·슬리피지 반영 백테스팅 엔진
- **Phase 2 (예정)**: 다국어 뉴스 이벤트 드리븐 레이어 융합 (FinBERT / KR-FinBERT / Claude API 감성 점수)

## 실행 환경

- 로컬 개발 및 Google Colab(+ code-server) 양쪽에서 실행 가능하도록 `DATA_ROOT` 경로를 자동 분기합니다 (`CLAUDE.md` 참조).
- 세션이 중간에 끊겨도 `progress_log.json`을 통해 이어받기가 가능합니다.

## Setup

```bash
pip install -r requirements.txt
pytest tests/
```
