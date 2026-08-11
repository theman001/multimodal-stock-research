# CLAUDE.md — 멀티모달 주식 리서치 시스템 (Multimodal Stock Research)

이 파일은 이 저장소에서 작업하는 모든 Claude Code 세션이 **매 세션 시작 시 가장 먼저 읽어야 하는** 운영 규칙입니다. 아래 규칙은 협의를 통해 확정된 것이므로, 특별한 지시가 없는 한 임의로 변경하지 마세요.

**이 문서는 자기완결적입니다.** 저장소에 `plan/` 폴더가 있다면 설계 배경과 세부 근거가 담겨 있어 참고하면 좋지만, `plan/`은 git에 커밋되지 않는 로컬 전용 폴더라 새로 clone한 환경에는 없을 수 있습니다. 코드를 작성하는 데 필요한 모든 수치·규칙은 이 파일 안에 다 있어야 하며, `plan/`이 없다는 이유로 작업을 멈추지 마세요.

## 세션 시작 시 행동 순서

1. 이 파일(CLAUDE.md) 전체를 읽는다
2. `progress_log.json`을 읽고 `next_action`부터 이어서 진행한다 (아래 "진행 로그 프로토콜" 참고)
3. `plan/`이 존재하면 해당 모듈의 상세 문서를 참고한다 (없어도 무방 — 이 파일이 우선)
4. 작업은 원자 단위로 진행한다 — 한 세션에 "데이터 수집 모듈"처럼 하나의 하위 모듈만 완료하고, 다음 지시를 기다린다

## 프로젝트 비전

**최종 목표는 스스로 매수/매도/보유를 판단하는 AI Agent입니다.** 그 판단의 기반이 되는 신호를 Phase 1~2에서 통계적으로 검증하는 것이 지금 단계의 목적입니다: 단순 가격 예측이 아니라 "시장 분류 × 기업 규모 카테고리에 따라 기술적 지표와 뉴스 이벤트의 통계적 유의성이 어떻게 달라지는가"를 검증하는 리서치 파이프라인입니다. 데이터셋을 케이스별로 쪼개서 각개 학습시키지 않고, 하나의 데이터셋에 `Market_Id`(국내/해외), `Size_Id`(대형/중형/소형) **메타 특성**을 주입해 조건부 연관성을 학습시키는 것이 핵심 사상입니다.

## 전체 로드맵과 현재 스코프: Phase 1만 진행

- **Phase 1 (현재)**: 차트 중심 조건부 연결 학습 MVP — 데이터 수집, 메타 태그 결합, 누수 없는 시계열 분할/학습, 수수료·슬리피지 반영 백테스팅. 산출물은 "익일 등락 확률" 신호
- **Phase 2 (금지 — 아직 시작하지 말 것)**: 다국어 뉴스/이벤트 신호 융합 (감성 점수 + 실시간 해석). Phase 1 Definition of Done을 전부 충족하고 사용자가 명시적으로 승인하기 전에는 뉴스/감성 관련 코드를 작성하지 않는다.
- **Phase 3 (금지 — 아직 계획도 상세화하지 말 것)**: RL 트레이딩 에이전트 — Phase 1의 확률 신호 + Phase 2의 해석을 state로 받아 매수/매도/보유 정책을 학습하는 signal-conditioned RL. 원본 차트/뉴스로 처음부터 학습하는 end-to-end RL이 아님 — 이미 검증된 신호 위에서 학습하므로 순수 RL보다 다루기 쉬운 구조로 설계할 것. Phase 2 완료 후 착수
- **Phase 4 (금지, 필요시에만)**: 실시간 운영 루프. 학습과 별개의 배포 문제로 취급하고 Phase 3 이후 별도 승인 하에 진행

각 Phase는 이전 Phase의 Definition of Done을 충족하고 사용자가 명시적으로 승인해야 다음으로 진행한다. 지금 세션에서 Phase 2/3/4의 코드나 상세 설계를 미리 만들지 않는다 — 스코프 확장은 반드시 사용자 지시로만 시작한다.

## 환경 설정

- Python 3.11+ 가정. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- 테스트 실행: `pytest tests/ -v` — 새 기능 완료 시점마다 반드시 통과해야 함 (특히 누수 관련 테스트는 절대 skip하지 않는다)
- 의존성 추가 시 `requirements.txt`에 반영

## 저장소 레이아웃

```
src/data_collection/   # pykrx(KR) / yfinance(US) 수집 스크립트
src/features/          # 기술지표 + 메타피처 + 타겟 생성
src/models/            # XGBoost 학습/추론
src/backtest/          # 수수료·슬리피지 반영 백테스팅, 유의성 검정
tests/                 # 누수 검증 등 유닛테스트
plan/                  # (git 제외) 로컬 설계 노트 — 없어도 무방
.claude/skills/leakage-guard/  # 데이터 누수 셀프체크 스킬
progress_log.json      # 세션 재개용 상태 파일
```

## Phase 1 구체 사양 (임의 변경 금지 — 바꾸려면 먼저 사용자에게 확인)

### 종목 유니버스
- 국내: KOSPI 대형/중형/소형 각 20종목 (시가총액 순위 상위, 하단 "분류 기준" 참고)
- 해외: S&P 500/400/600 각 20종목
- **총 120종목**

### 기간
- 2015-01-01 ~ 스크립트 실행 시점

### 저장 형식
- Parquet. `${DATA_ROOT}/raw/{kr,us}/`(원본 캐시), `${DATA_ROOT}/processed/`(가공본), `${DATA_ROOT}/checkpoints/`(모델), `${DATA_ROOT}/reports/`(백테스트 리포트) — 전부 `.gitignore` 처리됨, git에 데이터/체크포인트를 커밋하지 않는다

### 타겟 변수
```
target_t = 1 if Close_{t+1} / Close_t - 1 > 0 else 0
```

### 피처 (모두 절대가격이 아닌 비율/변화율로 — 국내/해외 가격 스케일 차이가 `Market_Id`와 혼동되는 것을 방지)
- `ma5_ratio`, `ma20_ratio`, `ma60_ratio` = `MA_n / Close - 1`
- `rsi14` (표준 RSI 14)
- `macd`, `macd_signal`, `macd_hist` (12, 26, 9)
- `bb_pctb` (Bollinger %B, 20, 2σ)
- `vol_chg5` = `Volume / Volume.shift(5) - 1`
- `volatility20` (20일 일간수익률 표준편차)
- `momentum5`, `momentum20`, `momentum60` = `Close / Close.shift(N) - 1`
- 메타피처: `market_id`(0=해외/1=국내), `size_id`(0=소형/1=중형/2=대형) — XGBoost에 `dtype="category"` + `enable_categorical=True`로 전달

### Train/Test 분할
- 전역 날짜 기준(종목별 아님) 시간순 80% / 20%
- Train 끝 ~ Test 시작 사이 **60 캘린더데이 embargo** (최장 롤링 윈도우 MA60 대응)
- 스케일러는 Train에만 `fit()`, Test는 `transform()`만

### 모델 베이스라인
- XGBoost 이진분류: `max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss"`
- 평가는 전체 지표 + **`Market_Id`×`Size_Id` 6개 셀별 breakdown 필수**

### 백테스팅
- 수수료/슬리피지는 함수 파라미터로 노출 (하드코딩 금지). Placeholder: 국내 왕복 약 0.03%+거래세 약 0.15~0.18%(**최신 세율은 구현 시점에 재확인 — 정책적으로 변경되어 온 값**), 해외 커미션 0 + 슬리피지 5bp
- 랜덤워크(50% 무작위, N=1000 시뮬레이션) 및 Buy&Hold 대비 비교
- 6개 셀 각각 이항검정 → **Bonferroni 보정 필수** (다중비교)

## 시장/규모 분류 기준 (반드시 이 기준을 그대로 사용 — 임의 기준 생성 금지)

### 국내 (KOSPI)
KRX 공식 순위 기준을 그대로 사용합니다:
- 코스피 시가총액 순위 **1~100위 = 대형주, 101~300위 = 중형주, 301위 이하 = 소형주**
- 순위는 **정기변경일 이전 3개월간 일평균 시가총액** 기준
- **매년 6월/12월** 정기변경 (6월물·12월물 최종거래일 익일 적용)
- pykrx로 각 정기변경 시점의 구성을 재구성해서 **그 시점 기준**으로 라벨링할 것 (아래 "데이터 누수 방지" 참조). 정확한 함수 시그니처는 pykrx 버전에 따라 다를 수 있으니 구현 시 `help()`/소스로 재확인할 것

### 해외 (미국)
S&P 지수 계열을 그대로 사용합니다. **S&P 500을 대형/중소형으로 임의 재분류하지 않습니다** — S&P 500 종목은 인덱스 편입 기준상 이미 전부 대형주입니다.
- **S&P 500 구성종목 = 대형주**
- **S&P MidCap 400 구성종목 = 중형주**
- **S&P SmallCap 600 구성종목 = 소형주**
- 각 지수의 **그 시점 구성종목**을 사용할 것 (오늘 시점 구성종목을 과거 데이터에 소급 적용 금지)
- **알려진 제약**: yfinance는 지수의 과거 시점 구성종목을 제공하지 않는다. MVP 1차는 "현재 구성종목을 전체 기간에 소급 적용"으로 진행하되, 이는 survivorship bias가 있는 known limitation으로 코드 주석·리포트에 반드시 명시할 것. 정확한 point-in-time 재구성(Wikipedia 변경이력 파싱 등)은 1차 결과가 유의미할 때 후속 개선으로 진행

## 데이터 누수(Look-ahead Bias) 방지 — 위반 시 반드시 중단하고 재작업

1. **스케일러**: `fit()`은 Train split에만 적용. Test/Val에는 `transform()`만.
2. **롤링 지표**(이동평균, RSI 등): 반드시 `shift()`/`rolling()`(과거 데이터만 포함, `center=True` 절대 금지)으로 미래 값이 섞이지 않도록 처리하고, 이를 검증하는 유닛 테스트를 `tests/`에 작성할 것 — "N일차 행의 피처는 N일차 이전 데이터만으로 계산되었는가"를 assert.
3. **`Market_Id`/`Size_Id` 메타 피처 자체도 누수 대상입니다**: 반드시 "그 날짜 시점에 유효했던 분류"를 사용할 것. 오늘 기준 대형주라고 해서 5년 전 데이터에도 대형주로 라벨링하면 안 됩니다.
4. **뉴스 타임스탬프 정렬**(Phase 2 대비 사전 원칙): 장마감 후 발행된 뉴스가 당일 종가 예측에 선반영되지 않도록 발행 시각과 거래 시각을 엄격히 정렬.
5. 작업 중 위 규칙 위반 가능성이 보이면 **코드를 계속 진행하지 말고** `.claude/skills/leakage-guard`의 체크리스트로 먼저 점검한다.

## 경로 규칙 (`DATA_ROOT`)

- 로컬 개발: 저장소 기준 `./data` (git에는 커밋하지 않음, `.gitignore` 처리됨)
- Colab 실행: `/content/drive` 존재 여부로 감지 → `/content/drive/MyDrive/주식프로젝트/`
- 코드에서는 하드코딩하지 말고 `DATA_ROOT` 환경변수 또는 자동 감지 로직으로 분기할 것

## 진행 로그 프로토콜 (`progress_log.json`)

세션이 중간에 끊겨도(Colab 세션 만료 등) 이어받기가 가능하도록 하는 상태 파일입니다.

- **세션 시작 시 가장 먼저 이 파일을 읽고**, `next_action`부터 이어서 진행할 것. `status: "done"`인 step은 재실행하지 말 것.
- 각 step 완료 시 **원자적으로 갱신**할 것 (임시 파일에 쓰고 rename — 쓰는 도중 세션이 끊겨도 손상된 파일이 남지 않도록).
- 스키마:
  ```json
  {
    "phase": "1",
    "last_completed_step": "step_name or null",
    "steps": [
      {"step": "...", "status": "done|in_progress|failed", "timestamp": "ISO8601", "output": "생성된 파일 경로"}
    ],
    "next_action": "다음에 할 일 한 문장",
    "blockers": []
  }
  ```

## 실행 통제 (권한 가이드)

- **자동 허용 대상**: `pip install`(requirements.txt 범위 내), 저장소 내부 `python`/`pytest` 실행
- **승인 필요 대상**: 파괴적 셸 명령(`rm -rf` 등), `requirements.txt`에 없는 외부 네트워크 호출, `git push`, 원격 저장소 관련 작업
- 한 번에 여러 Phase/여러 모듈을 동시에 진행하라는 지시가 없는 한, 하나의 원자적 작업 단위로 쪼개서 수행할 것

## 모델/평가 원칙 (Phase 1 백테스팅 설계 시 참고)

- 방향성 적중률 50~55% 수준을 현실적 한계로 인정하고, 이를 넘는 정확도를 억지로 만들려 하지 말 것. 크게 벗어나는 셀이 있으면 과최적화/누수 의심 신호로 취급하고 누수 테스트를 재확인할 것
- 수수료(국내외 세금 포함) + 슬리피지를 파라미터화해 보수적으로 차감
- Random walk 대비 통계적으로 유의미한 초과수익(알파)인지 검증할 것
- `Market_Id` × `Size_Id` 조합별로 유의성을 각각 검정할 경우 다중비교 문제가 생기므로 보정(Bonferroni)을 적용할 것
- Train/Test 경계 부근에서 롤링 피처로 인한 누수가 없도록 embargo 구간을 둘 것

## Phase 1 Definition of Done

- [ ] 120종목 point-in-time `Market_Id`/`Size_Id` 라벨링 완료 (결측/오분류 0건)
- [ ] `tests/`의 모든 누수 검증 유닛테스트 통과
- [ ] 베이스라인 XGBoost 체크포인트 + 메트릭(6셀 breakdown 포함) 저장
- [ ] 백테스트 리포트 생성 (수수료/슬리피지 반영, 랜덤워크 대비 유의성 검정 + 다중비교 보정 포함)
- [ ] `progress_log.json`의 모든 step이 `done`, `next_action`이 "Phase 2 착수 여부 사용자 확인 대기"로 갱신

## 관련 스킬

- `.claude/skills/leakage-guard` — 데이터 수집/피처 엔지니어링/분할 코드를 작성하거나 리뷰할 때, 또는 `progress_log.json`에 해당 step을 `done`으로 표시하기 전에 반드시 이 스킬의 체크리스트를 실행할 것
