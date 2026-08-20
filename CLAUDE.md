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

## 전체 로드맵과 현재 스코프: Phase 3 완료, Phase 4 KR/US 트랙 코드 구현(1~5단계) 둘 다 완료 — 양쪽 다 6단계(무인 운영)만 대기 중

- **Phase 1 (완료, 2026-08-12 사용자 승인)**: 차트 중심 조건부 연결 학습 MVP — 데이터 수집, 메타 태그 결합, 누수 없는 시계열 분할/학습, 수수료·슬리피지 반영 백테스팅. 산출물은 "익일 등락 확률" 신호(model_v3). walk-forward CV·하이퍼파라미터 탐색으로 51~53% 방향성 정확도가 기술지표 기반 신호의 현실적 천장임을 확인.
- **Phase 2 (완료, 2026-08-17)**: 뉴스/이벤트 신호 융합. Definition of Done 전 항목 충족(모듈 1~5). 이벤트/감성 피처를 추가해도 XGBoost 익일 방향성 예측은 노이즈 수준 이상 개선되지 않는다는 결론(`data/reports/phase2_retraining_comparison_report.md`) — model_v3(13피처)를 공식 산출물로 유지. 상세 사양은 `plan/08_phase2_news_events.md` 참고.
- **Phase 3 (완료, 2026-08-20)**: RL 트레이딩 에이전트. 2026-08-18 전체 검토에서 `panel.py::score_model_v3_probabilities()`가 model_v3에 raw 미스케일 피처를 넣던 **CRITICAL 버그**를 발견(raw vs 올바른 스케일링 입력의 예측 상관계수 0.044)해 수정했고, 그 버그로 학습됐던 구 정책은 폐기했다. 2026-08-19 04:03~14:45(약 10시간42분)에 수정된 코드로 6개 정책(폴드1~5 + 공식 단일분할)을 `resume=False`로 전부 재학습, 이어서 재평가해 **train/eval 입력분포가 일치하는 신뢰 가능한 결과**를 확보했다. 결론: 공식분할 RL 누적수익률 101.03%(분류기 27.09%, Buy&Hold 89.55% 대비 높음)이나, 평균 보유종목수 57.7/120·평균 현금비중 0.45%로 "정교한 타이밍"이 아니라 "항상 널리 분산해 최대 투자 상태 유지"에 가까운 저정교도 전략으로 수렴한 것으로 해석됨 — 절대수익률을 그대로 "학습된 알파"로 보지 않음. 상세는 `data/reports/phase3_rl_backtest_report_v1.md`(해석 유의사항 포함), `data/reports/phase3_reward_clipping_investigation.md`, 설계는 `plan/09_phase3_rl_agent_design.md` 참고.
- **Phase 4 (설계 확정 2026-08-20, KR/US 트랙 코드 구현 둘 다 완료 2026-08-20)**: 실시간 운영 루프. `plan/10_phase4_live_operations_design.md`에 상세 설계 확정. 코드 구현은 KR(KIS)/US(Alpaca) 두 개의 독립 세션으로 병렬 진행 — US 트랙은 공유 기반(observation/infer/allocation/safety/notify/broker base) + `broker/alpaca.py` + `decide_us.py`/`execute_us.py` 전부 구현·review-loop 검증·테스트 통과 완료(cron 미설치, Alpaca 키·Mattermost webhook 설정 대기). KR 트랙도 별도 세션에서 착수해 공유 기반을 재작성 없이 그대로 재사용 + KR 고유의 데이터 증분화 버그 수정(`collect_kr.py`/`events_kr.py`) + `broker/kis.py`(KIS Open API — TR_ID/응답 필드명은 실제 앱키로 재확인 필요, notional 주문 미지원이라 `get_current_price()`로 체결 직전 실시간 현재가 조회 후 정수 수량 변환) + `decide_kr.py`/`execute_kr.py` 전부 구현·review-loop 검증(각 파일마다 최소 2라운드, execute_kr.py에서 KIS 고유의 가격조회 실패 격리 버그 발견·수정)·테스트 통과 완료(cron 미설치, KIS 앱키·Mattermost webhook 설정 대기). **양쪽 다 무인 운영은 별도 사용자 승인 필요** — 이 세션이 결정하지 않음. 상세는 아래 "Phase 4 구체 사양"과 "Phase 4 Definition of Done" 참고

각 Phase는 이전 Phase의 Definition of Done을 충족하고 사용자가 명시적으로 승인해야 다음으로 진행한다. Phase 3는 설계 확정(2026-08-17 "Phase3는 구체적인 계획을 먼저 세우자") 이후, 원자적 모듈(panel/obs_scaler → trading_env → 학습 파이프라인+실학습 → 평가) 각각을 사용자가 "진행해"로 승인하며 완료했다 — 이 방식이 실제 구현 승인 절차였다. Phase 4는 코드나 상세 설계를 아직 미리 만들지 않는다 — 스코프 확장은 반드시 사용자 지시로만 시작한다.

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
- Parquet. `${DATA_ROOT}/raw/{kr,us}/`(원본 캐시), `${DATA_ROOT}/processed/`(가공본), `${DATA_ROOT}/checkpoints/`(모델), `${DATA_ROOT}/reports/`(백테스트 리포트).
- **2026-08-18부터 `data/`를 git에 커밋한다**(사용자 명시 지시 — Colab에서 매번 파일 업로드 없이 바로 clone만으로 필요한 데이터가 갖춰지게 하려는 목적). 그 이전에는 전부 `.gitignore` 처리돼 커밋하지 않았음. 커밋 시점 크기 참고: 376MB, 약 4,900개 파일 — `features.parquet`/체크포인트처럼 자주 재생성되는 바이너리가 매번 새 버전으로 히스토리에 쌓이므로, 저장소가 실제 폴더 크기보다 빠르게 불어난다는 점을 인지하고 진행할 것.

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

- 로컬 개발: 저장소 기준 `./data` (2026-08-18부터 git에 커밋됨 — "저장 형식" 절 참고)
- Colab 실행: `/content/drive` 존재 여부로 감지 → `/content/drive/MyDrive/주식프로젝트/multimodal-stock-research/`
- 코드에서는 하드코딩하지 말고 `DATA_ROOT` 환경변수 또는 자동 감지 로직으로 분기할 것
- **Colab에서는 저장소 자체를 매 세션 GitHub에서 새로 `git clone`하는 것을 기본으로 한다** (Drive FUSE 마운트 위에서 `.git`을 직접 다루면 느리고 불안정함). 즉 저장소 코드는 세션마다 사라져도 되는 휘발성으로 취급하고, **데이터·체크포인트·진행 상태처럼 반드시 남아야 하는 것만 `DATA_ROOT`(Drive)에 둔다.**
- **주의**: `data/`가 git에 커밋돼 있어도(위 참고) Colab에서 Drive를 마운트하면 `get_data_root()`는 여전히 Drive 경로를 우선한다 — clone된 저장소 안의 `./data`를 자동으로 쓰지 않는다. 그래서 Colab에서 매 파일 업로드 없이 바로 쓰려면, clone 직후 `./data`를 Drive의 `DATA_ROOT` 경로로 한 번 복사(`cp -r`)해줘야 한다. 이렇게 해야 (1) git clone만으로 파일이 안 빠지고 (2) 학습 중 체크포인트도 Drive에 남아 세션이 끊겨도 살아남는다는 두 목적을 동시에 만족한다.

## 진행 로그 프로토콜 (`progress_log.json`)

세션이 중간에 끊겨도(Colab 세션 만료 등) 이어받기가 가능하도록 하는 상태 파일입니다.

- **파일 위치는 저장소 안이 아니라 `${DATA_ROOT}/progress_log.json`이다.** 저장소는 매 세션 새로 clone될 수 있는 휘발성이므로, 저장소 안에 두면 세션이 끊기는 순간 마지막 상태가 통째로 유실된다. 로컬 개발 시에는 `./data/progress_log.json`.
- **세션 시작 시 가장 먼저 `${DATA_ROOT}/progress_log.json`을 읽고**, `next_action`부터 이어서 진행할 것. `status: "done"`인 step은 재실행하지 말 것. **파일이 없으면** (최초 실행) 아래 스키마로 새로 생성해서 시작한다.
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

- [x] 120종목 point-in-time `Market_Id`/`Size_Id` 라벨링 완료 (결측/오분류 0건)
- [x] `tests/`의 모든 누수 검증 유닛테스트 통과
- [x] 베이스라인 XGBoost 체크포인트 + 메트릭(6셀 breakdown 포함) 저장 (model_v3)
- [x] 백테스트 리포트 생성 (수수료/슬리피지 반영, 랜덤워크 대비 유의성 검정 + 다중비교 보정 포함, backtest_report_v3.md)
- [x] `progress_log.json`의 모든 step이 `done`, 2026-08-12 사용자가 "Phase2로 넘어가자"로 명시 승인

## Phase 2 구체 사양 (2026-08-12 확정 — 임의 변경 금지, 바꾸려면 `plan/08_phase2_news_events.md` 참고 후 사용자 승인)

상세 근거(데이터소스 조사 결과 등)는 `plan/08_phase2_news_events.md`에 있지만, 코드 작성에 필요한 확정 수치는 이 섹션에 자기완결적으로 둔다.

### 이벤트 데이터 소스 (일반 뉴스 스크레이핑 아님 — 공식 공시 API)
- **한국**: DART Open API(`opendart.fss.or.kr`, 금융감독원 공식, 무료 인증키). 공시목록 API는 `corp_code` 지정 시 기간 제한 없음.
- **미국**: SEC EDGAR(`data.sec.gov`, 공식, 키 불필요). `submissions/CIK##########.json`으로 회사별 전체 filing 이력 조회.
- 근거: 네이버 뉴스 API(과거기간 검색 불가), NewsAPI.org(무료 티어 최근 24개월만), Alpha Vantage(25 req/day) 등은 120종목×11년 스코프를 무료·합법·재현 가능하게 커버할 수 없음을 조사로 확인. 크롤링은 이번 세션에서 KRX ToS 위반 차단을 실제로 겪은 전례가 있어 배제.

### 이벤트 유형
- 한국: DART 주요사항보고서·수시공시(`pblntf_ty` B/I — 확인 완료, 정기공시/지분공시 등은 제외). `pblntf_ty`는 DART 응답 `list` 배열에는 없는 요청 전용 파라미터라, 값마다 별도 요청을 보내야 한다(응답을 받은 뒤 클라이언트에서 거를 방법이 없음 — 처음에 이걸 몰라서 무필터로 수집했다가 재수집한 적 있음).
- 미국: 8-K(Current Report) 중심, 필요시 10-K/10-Q 보강

### 감성 스코어링
- 영어: FinBERT(예: `yiyanghkust/finbert-tone`), 한국어: `snunlp/KR-FinBert-SC` — 둘 다 HuggingFace 무료 로컬 모델로 전체 이력을 스코어링한다.
- Claude API는 전체 이력 스코어링에 쓰지 않는다(비용/속도) — 소규모 정성 비교 용도로만 고려.

### 타임스탬프 정렬 규칙 (데이터 누수 방지 원칙 4 이행 — 반드시 유닛테스트로 검증)
- 한국: DART 응답의 `rcept_dt`는 날짜만 제공(시:분:초 없음, API 문서로 확인 완료) → **접수일자 다음 거래일(D+1)부터** 이벤트 피처에 반영.
- 미국: EDGAR 응답의 `acceptanceDateTime`은 초 단위 포함 → 정규장 마감(America/New_York) **이전 접수는 당일, 이후 접수는 다음 거래일**부터 반영. 마감 기준은 평소 16:00이지만, NYSE 조기마감일(독립기념일 전날/추수감사절 다음날/크리스마스이브)은 13:00으로 낮춘다 — 고정 날짜 목록이 아니라 `trading_days` 캘린더로 "그 공휴일 직전 마지막 거래일"을 계산해서 공휴일-주말 겹침으로 관찰일이 밀리는 해(예: 2026년 7/4가 토요일)까지 반영한다(`src/data_collection/event_timing.py`). 최초 구현엔 이 조기마감 예외가 없어서 실제 데이터 3건(TSLA/JPM 2017-07-03, VNOM 2019-07-03)이 잘못 배정되고 있었음을 리뷰로 발견해 수정함.
- 시각이 불확실/애매하면 항상 더 보수적인(늦은) 쪽으로 배정한다.

### 피처 결합
- 기존 `FEATURE_COLUMNS`에 이벤트 피처를 추가해 동일 XGBoost 파이프라인으로 재학습(별도 앙상블 아님).
- 이벤트가 없는 날은 NaN이 아니라 "이벤트 없음"을 뜻하는 중립값(0)으로 채운다 — Phase 1의 "NaN row는 drop" 원칙을 그대로 적용하면 대부분의 행이 사라지므로 예외로 둔다.
- 후보 피처: `event_count_5d`, `event_sentiment_latest`, `event_sentiment_mean5d`, `days_since_last_event` (구현 중 조정 가능)

## Phase 2 Definition of Done

- [x] KR(DART)/US(EDGAR) ticker↔corp_code/CIK 매핑 완료, 120종목 커버리지 확인
- [x] 이벤트 원문 수집 + raw 캐싱(KR 22,490건/US 7,944건), 타임스탬프 정렬 규칙 구현 + 누수 검증 유닛테스트 통과 — 검증 라운드는 `data/reports/phase2_event_collection_report.md` 참고
- [x] 감성 스코어링(US 7,944건/KR 22,490건 전체, FinBERT/KR-FinBERT + 문장 단위 슬라이딩 윈도우) — 검증 내역은 `data/reports/phase2_sentiment_scoring_report.md` 참고
- [x] 이벤트 피처 집계, 기존 `features.parquet`과 병합 (모듈 4) — 상세 내역은 `data/reports/phase2_event_feature_engineering_report.md` 참고
- [x] 이벤트 피처 포함 재학습을 walk-forward CV로 Phase 1 베이스라인과 비교, 리포트 작성(개선이 노이즈 수준이면 미채택도 유효한 결론) — 결론: 노이즈 수준 차이, 미채택. 상세 내역은 `data/reports/phase2_retraining_comparison_report.md` 참고
- [x] `progress_log.json` 갱신, `next_action`이 "Phase 2 결과 확인 후 Phase 3 착수 여부 사용자 확인 대기"로 갱신

## Phase 3 구체 사양 (2026-08-17 설계 확정 — 코드 구현은 별도 승인 필요, 상세 근거는 `plan/09_phase3_rl_agent_design.md`)

**핵심 결정 (미정 항목 확정)**: 처음부터 120종목 전체 포트폴리오 에이전트(단일종목 MVP 아님) / 이산 매수·매도·보유(포지션 사이징은 v1 범위 밖) / 이벤트·감성 피처를 state에 포함(모듈 5의 분류 무용 결과와 별개로 RL엔 다르게 쓰일 수 있다는 가설, ablation으로 추후 검증) / PPO(`stable-baselines3==2.9.0` + `gymnasium==1.3.0`, 이 저장소 Python 3.14 venv에서 설치 가능함을 dry-run으로 확인).

### 관측(observation) 공간
`gymnasium.spaces.Box(shape=(120*D + 4,))`. 티커별 블록(D=20, 이벤트 피처 포함 시 24) = `BASE_FEATURE_COLUMNS`(13) + `model_v3` P(up)(1, 오프라인 배치 계산, env.step()마다 재추론 안 함) + `EVENT_FEATURE_COLUMNS`(0/4, 생성자 플래그로 토글) + `market_id`/`size_id`(각 1) + `holding_flag`/`unrealized_return`/`position_weight`(각 1, env 계산) + `valid_mask`(1). 포트폴리오 스칼라 4개(`cash_weight`, `nav_ratio`, `n_positions_held/120`, `step_frac`) 추가. 티커 순서는 `${DATA_ROOT}/checkpoints/rl_ticker_universe.json`에 고정 저장(`MARKET_ID_CATEGORIES`/`SIZE_ID_CATEGORIES`와 동일한 이유). 그 날 row가 없는 티커는 0-채움 + `valid_mask=0`(forward-fill 금지 — 암묵적 정보 누수 방지). `MAX_REALIZED_RETURN_GAP_DAYS=10`(`simulate.py`, 재사용) 넘게 마스킹되면 마지막 유효가로 강제 청산.

### 행동(action) 공간
`gymnasium.spaces.MultiDiscrete([3]*120)`, 티커별 독립 3지 선택(`SELL=0, HOLD=1, BUY=2`), 고정 티커 순서, 롱 온리.

### 자본배분
NAV 무차원(1.0 시작). 그 스텝 BUY 신호 중 미보유 티커만 대상으로 `equal_share = 가용현금/len(buy_tickers)`, 종목당 `min(equal_share, MAX_POSITION_WEIGHT)`(기본 0.05, 파라미터화). 캡으로 남는 현금은 재분배 안 함(보수적). 보유 중 BUY는 no-op, SELL은 전량 청산만. 에피소드 마지막 스텝은 행동과 무관하게 강제 전량 청산.

### 보상 함수
학습 신호 `reward_t = ln(NAV_{t+1}/NAV_t)`, `REWARD_CLIP=0.15`로 클리핑(실제 NAV/포지션 장부는 클리핑 없이 정확히 추적 — `info["nav"]`/`info["raw_reward"]`가 참값). 리포트용 지표는 `simulate.py::cumulative_return()`에 그대로 통과(클리핑 안 된 참값 NAV 시계열 사용). **비용은 `simulate.py`처럼 매일 부과하지 않고 포지션을 열 때/닫을 때만** `round_trip_cost_for_market()`(`costs.py`, 요율 재사용)의 절반씩 부과 — 진입→청산 사이클 총비용은 Phase 1과 동일하게 유지, 타이밍만 다름(RL은 진짜 포지션 상태가 있으므로).

### PPO 하이퍼파라미터 (실측 검증됨 — 임의 변경 금지, 근거는 `data/reports/phase3_reward_clipping_investigation.md`)
`learning_rate=3e-5`(SB3 기본값 3e-4는 이 문제 규모(관측 ~2,884차원, 120개 동시 행동헤드)에서 approx_kl이 이터레이션마다 폭주하는 걸 실측으로 확인 — 절대 기본값으로 되돌리지 말 것), `target_kl=3.0`(MultiDiscrete(120)이라 SB3가 120개 독립 카테고리 분포의 KL을 합산 보고하므로 단일 행동공간 기준값의 약 120배 스케일), `policy_kwargs=dict(net_arch=[256,256])`. `n_envs=8`(`DummyVecEnv`) + `n_steps=256`(버퍼=2048)이 로컬 4코어 환경 기준 실측 검증된 조합(13.90ms/step, n_envs=1 대비 2.59배).

### 에피소드/평가
학습은 `EPISODE_LENGTH_DAYS=252`, 폴드 train 구간 내 무작위 시작(embargo 침범 방지 가드 포함). 평가는 결정적 단일 패스. `generate_walk_forward_folds()`(`walk_forward.py`, 그대로 재사용)로 5폴드 정의, 폴드마다 새로 학습. 공식 헤드라인 비교는 `train.parquet`/`test.parquet`(2024-06-25~2026-08-07, `backtest_report_v3.md`와 동일 구간)에서 별도로 1회.

### 파일 레이아웃 (제안)
`src/rl/{panel,obs_scaler,trading_env,train_agent,evaluate}.py`, `tests/test_trading_env.py`, `tests/test_rl_panel_leakage.py`.

### RL 특화 누수 주의사항 (가장 비자명한 것 하나)
SB3의 `VecNormalize`는 기본적으로 온라인으로 통계를 갱신하는데, 평가 중에도 켜두면 이전 결정에 이후 타임스텝 정보가 섞여든다 — `scaling.py`의 "Train에만 fit" 원칙을 `obs_scaler.py`에도 그대로 적용해 평가 중엔 고정 스케일러만 사용.

### 평가/비교
`test.parquet` 구간에서 RL 정책 vs `backtest_report_v3.md`의 분류기 전략(재계산 없이 인용) vs 랜덤워크/Buy&Hold(재사용) 비교. 두 전략의 일별수익률 *차이*에 대한 paired bootstrap이 새로 필요 → `significance.py`에 `paired_bootstrap_return_diff_ci()` 추가 제안. **미해결로 명시**: 기존 6셀 Bonferroni 독립 검정은 공동 현금 제약이 있는 포트폴리오 정책엔 독립성 가정이 깨져 그대로 재사용 불가 — 실제 리포트 작성 시점에 별도 판단.

### 연산 자원
로컬 CPU 파일럿 우선, 벽시계 시간 감당 안 되면 기존 Colab GPU 워크플로(디바이스 자동감지 패턴) 그대로 적용.

## Phase 3 Definition of Done

- [x] `src/rl/panel.py` + `obs_scaler.py` 구현, 누수 검증 유닛테스트(`tests/test_rl_panel_leakage.py`, `tests/test_rl_obs_scaler.py`) 통과 — 실제 데이터로 `build_panel()` 실행 검증 완료(2026-08-17)
- [x] `src/rl/trading_env.py`(Gymnasium 환경) 구현, reset/step/비용/강제청산 로직 유닛테스트(`tests/test_trading_env.py`) 통과 — 실제 120종목 패널로 무작위 정책 스모크 테스트 + all-HOLD NAV 불변 검증 완료(2026-08-17)
- [x] PPO 학습 파이프라인(`src/rl/train_agent.py`) 구현, CPU 파일럿으로 학습 가능성 확인(필요시 Colab GPU 전환) — 파일럿 결과 병목은 환경이 아니라 정책망 추론(배치크기 1 순차 추론). `train_policy(n_envs=N)`로 DummyVecEnv 배치화 적용, 로컬 4코어 기준 n_envs=8에서 2.59배 개선(36.01→13.90ms/step, 2026-08-17). **전체 실행 첫 시도에서 PPO 학습이 실제로 불안정(approx_kl 폭주 9.9→183.2)함을 발견해 중단·조사** — 데이터 결함이 아니라(042660/CORT 극단치 둘 다 진짜 시세변동으로 확인) `learning_rate` 기본값(3e-4)이 이 문제 규모(관측 ~2,884차원, 120개 동시 행동헤드)엔 과도했던 게 원인. `3e-5`로 낮추자 완전 안정화(10개 이터레이션 내내 approx_kl 0.07~0.09, explained_variance -1.82→0.78 꾸준히 개선). `PPO_DEFAULT_PARAMS`에 `learning_rate=3e-5`/`target_kl=3.0`(2차 안전장치) 반영, `trading_env.py`에 `REWARD_CLIP=0.15`도 함께 추가. 상세 조사 과정은 `data/reports/phase3_reward_clipping_investigation.md` 참고. **전체 5폴드+공식분할 실학습 완료**(2026-08-18, 02:03~13:14, 약 11.2시간) — `data/checkpoints/rl_policy_fold{1..5}.zip`+`rl_policy_v1.zip` 전부 저장, 전체 1,464개 이터레이션에 걸쳐 approx_kl 0.065~0.226 범위 유지(재발 없음, 안정화 조치가 장시간 실행에서도 유효함을 확인). **2026-08-18 전체 검토로 model_v3_prob 채널 스케일링 버그를 발견해 수정**, 이 학습 결과(구 체크포인트)는 폐기하고 **2026-08-19 04:03~14:45(약 10시간42분) 수정된 코드로 재학습 완료** — `data/checkpoints/rl_policy_fold{1..5}.zip`+`rl_policy_v1.zip`이 현재 신뢰 가능한 최종 체크포인트
- [x] Walk-forward 5폴드 + 공식 단일분할(test.parquet 구간) 평가 완료, Phase 1 분류기 전략/랜덤워크/Buy&Hold 대비 비교 리포트 작성(paired bootstrap 포함) — `src/backtest/significance.py::paired_bootstrap_return_diff_ci()` 신규 추가, `src/rl/evaluate.py` 구현. **2026-08-18 전체 검토로 `score_model_v3_probabilities()`의 CRITICAL 스케일링 버그를 발견·수정**(raw vs 스케일링 입력 예측 상관계수 0.044, 스케일링 후 model_v3 원래 정확도 0.5169399830938293과 정확히 일치 확인) — 그 버그로 학습됐던 구 정책·리포트는 무효화하고 폐기. **2026-08-19 04:03~14:45 수정된 코드로 6개 정책 재학습, 2026-08-20 재평가 완료** — train/eval 입력분포가 이제 일치하므로 아래 수치는 신뢰 가능. 공식분할 RL 누적수익률 101.03%(분류기 27.09%, Buy&Hold 89.55% 대비 높음)이나, 평균 보유종목수 57.7/120·평균 현금비중 0.45%로 실제로는 정교한 타이밍이 아니라 "항상 광범위하게 분산투자한 채 최대 노출 유지"로 수렴한 저정교도 전략으로 확인됨(랜덤워크 벤치마크는 매일 재추첨이라 지속보유 전략엔 약한 기준선 — 같은 구간 랜덤워크 최댓값은 14.0%뿐). 절대수익률을 "학습된 알파"로 해석하지 않음. 상세는 `data/reports/phase3_rl_backtest_report_v1.md` 참고
- [x] `progress_log.json` 갱신, `next_action`이 "Phase 3 결과 확인 후 Phase 4 착수 여부 사용자 확인 대기"로 갱신

## Phase 4 구체 사양 (2026-08-20 설계 확정 — 임의 변경 금지, 상세 근거는 `plan/10_phase4_live_operations_design.md`)

**목표**: Phase 3에서 신뢰 가능하게 재학습된 `data/checkpoints/rl_policy_v1.zip`을 매일 자동으로 서빙해 모의투자(→ 향후 승인 시 실거래) 결정을 자동화한다.

**핵심 결정 (임의 변경 금지)**:
1. 모의투자(paper) 우선, 실거래(live) 자동집행도 **동일 로직에서** 가능해야 한다 — 의사결정 로직(관측 생성→정책 추론→목표 포지션 계산)은 모의/실전 공용이고, 최종 "브로커에 주문을 보낸다" 단계만 `mode: Literal["paper","live"]` 스위치로 분기. **기본값은 절대 `live=True`가 아니어야 하며, live 경로를 실제로 호출하는 건 이번 구현 범위에서 제외**(별도 승인 후).
2. 로컬 머신 cron 실행(클라우드 서버 아님).
3. DART/EDGAR 공시 + FinBERT 감성도 매일 갱신해서 RL 관측에 포함(`include_event_features=True`로 학습됐으므로).
4. **v1은 단일 시장만** — `TradingEnv`의 자본배분(`equal_share`)이 120종목 단일 무차원 현금 풀 공유를 가정하는데, 실거래는 한국(원화 KIS 계좌)/미국(달러 Alpaca 계좌)이 물리적으로 분리돼 이 가정과 안 맞음. 이를 회피하기 위해 v1은 한 시장만.
5. **KR 트랙(KIS)과 US 트랙(Alpaca)을 별도 세션으로 분리 개발** — 공유 기반(`src/live/observation.py`, `infer.py`, `allocation.py`, `safety.py`, `notify.py`, `broker/base.py`)만 먼저 완성하면 이후 트랙별 파일이 겹치지 않는다.
6. 알림은 **Mattermost webhook 직접 사용**(n8n 경유는 부가 홉이라 필요할 때 침묵 실패 지점이 될 수 있어 배제).
7. **Phase 4는 순수 서빙(serving)만, 재학습(training)은 범위 밖** — `rl_policy_v1.zip`을 고정 가중치로 `model.predict()`만 수행. 주기적 재학습은 명시적 결정보류(모의투자 로그로 성능 저하가 실측되면 그때 별도 논의).
8. **"롱 온리" → "롱 베이스, 숏은 전략적으로 허용"(2026-08-20 사용자 결정, US 트랙 세션에서 확정)** — 범위는 **관측 계층 한정**이다. `rl_policy_v1.zip`은 여전히 SELL/HOLD/BUY 3지 선택뿐이라 스스로 숏에 진입하지 않는다(`TradingEnv`/`allocation.py` 모두 SELL=롱 포지션 전량 청산으로만 동작, 숏 오픈 경로 없음 — Phase 3 액션 공간 자체는 안 바뀜). `src/live/observation.py`의 `HeldPosition.qty<0`(숏)을 에러 없이 받아들이고 `unrealized_return`/`position_value` 부호를 정확히 계산하도록 대비해둔 것 — 브로커 계좌에 이미 존재할 수 있는 숏 포지션(수동 개입 등)을 크래시 없이 관측에 반영하기 위함이다. **정책이 능동적으로 숏에 진입하는 기능은 액션 공간 자체를 바꿔야 하는 Phase 3 재학습 범위**라 이번 Phase 4 세션(순수 서빙) 밖이고, 아직 별도로 스코프되지 않은 향후 과제로 남겨둔다.

**아키텍처 요약**: `daily_pipeline.py`(증분 데이터 수집, 기존 `fetch_us_ohlcv`/`fetch_all_daily_snapshots`/`fetch_kr_disclosures`/`fetch_us_filings`의 "캐시 있으면 그대로 반환" 버그 수정 필요) → `observation.py`(오늘자 단일 관측벡터, `TradingEnv._build_observation()`과 동일 채널 순서이나 동적 필드는 브로커 계좌 조회로 계산) → `infer.py`(`load_checkpoint`+`model.predict`) → `allocation.py`(`TradingEnv.step()` 배분 로직의 순수 함수 재구현) → `broker/{kis,alpaca}.py`(`BrokerAdapter` 공통 인터페이스, 모드는 생성자 파라미터로만 분기) → `safety.py`(중복실행 락/한도/재구성 체크) → `decide_{kr,us}.py`(07:00 KST cron, 결정만 저장)/`execute_{kr,us}.py`(시장별 집행 시각 cron) → `notify.py`(Mattermost).

**미해결 항목(명시적으로 열어둠)**: `step_frac`/`nav_ratio`는 252일 에피소드 학습 전제라 라이브의 "에피소드 없음" 상황에 잠정 규칙 필요(모의투자로 사후 검증). 결정 시각과 체결 시각 사이 오버나이트 갭은 학습 가정 비용에 없음(실제 vs 가정 비용을 리포트에 병기). 브로커 계좌 개설/API 키, rate limit, 실거래 전환 기준은 사용자/모의투자 로그 기반 재논의 대상.

## Phase 4 Definition of Done (KR/US 트랙별로 별도 체크, 공유 기반 항목은 1회만)

- [x] `src/live/observation.py`가 배치 경로(`src/rl/panel.py::build_grid`)와 수치 일치 (공유, 1회만) — US 트랙 세션에서 구현 완료(2026-08-20). `tests/test_live_observation.py`로 검증, review-loop 다수 라운드로 NaN/Inf·부호·마스킹 처리 버그 수정. KR 트랙이 그대로 재사용할 것(재작성 금지)
- [x] `src/live/infer.py` + `src/live/allocation.py`가 `TradingEnv` 배분 로직과 일치 (공유, 1회만) — US 트랙 세션에서 구현 완료(2026-08-20), `tests/test_live_infer.py`/`tests/test_live_allocation.py`로 검증. KR 트랙이 그대로 재사용할 것(재작성 금지)
- [x] 데이터 증분 수집 버그 수정 + 회귀 테스트(해당 트랙의 시장: KR은 `collect_kr.py`/`events_kr.py`, US는 `ohlcv.py`/`events_us.py`) — **US/KR 둘 다 완료**(US 2026-08-20 `fetch_us_ohlcv`/`fetch_us_filings`; KR 2026-08-20 `fetch_all_daily_snapshots`/`extract_universe_ohlcv`/`fetch_kr_disclosures`, review-loop 2라운드로 `end_date` 계약 위반 등 실제 버그 2건 추가 발견·수정)
- [x] 브로커 어댑터(`kis.py` 또는 `alpaca.py`, 모의투자 도메인) mock 기반 유닛테스트 통과 — **US/KR 둘 다 완료**(US `alpaca.py` 2026-08-20; KR `kis.py` 2026-08-20 — OAuth2 토큰 발급/hashkey/HTTP 200이어도 rt_cd 기반 실패 판정 등 KIS 특유 처리 포함, TR_ID/응답 필드명은 실제 앱키로 재확인 필요라고 모듈 docstring에 명시. notional 주문 미지원이라 execute_kr.py가 쓰는 `get_current_price()`도 함께 구현, review-loop로 output 필드 누락 시 KeyError 대신 명확한 에러로 처리하도록 수정)
- [x] `src/live/safety.py`가 실제로 주문을 차단하는 테스트 통과 — US 트랙에서 구현+검증 완료(2026-08-20, `tests/test_live_safety.py`). safety.py 자체는 공유 모듈이라 KR도 재작성 없이 그대로 재사용(2026-08-20 `decide_kr.py`/`execute_kr.py`에서 track="kr"로 호출)
- [ ] `decide_*.py`/`execute_*.py` cron 등록, 최소 N영업일(사용자와 재합의 필요) 무인 모의투자 운영, 크래시 0건 — **US/KR 둘 다 코드 구현+테스트+cron 파일 준비까지 완료**, 실제 cron 등록·무인 운영은 둘 다 미착수. US: `decide_us.py`/`execute_us.py`/`notify.py`(2026-08-20), Alpaca API 키/.env·MATTERMOST_WEBHOOK_URL 설정이 선행 필요. KR: `decide_kr.py`/`execute_kr.py`(2026-08-20, notify.py는 재작성 없이 그대로 재사용 — Mattermost 알림 연동은 이미 두 스크립트에 내장됨), KIS 앱키/.env 설정이 선행 필요 — 둘 다 사용자 액션 대기 중. `cron_us.txt`/`cron_kr.txt`에 crontab 스니펫 준비만 해두고 시스템 crontab에는 설치 안 함(무인 자동 실행 시작은 별도 명시적 승인 필요)
- [ ] 일일 리포트(결정/체결/실현손익 vs 시뮬레이션 가정 비교) 축적, 실거래 전환은 이 DoD 충족과 무관하게 별도 명시적 승인 — 무인 운영이 시작돼야 축적 가능, 아직 미착수
- [x] `progress_log.json` 갱신 — 매 원자적 단계·review-loop 라운드마다 갱신 중

## 관련 스킬

- `.claude/skills/leakage-guard` — 데이터 수집/피처 엔지니어링/분할 코드를 작성하거나 리뷰할 때, 또는 `progress_log.json`에 해당 step을 `done`으로 표시하기 전에 반드시 이 스킬의 체크리스트를 실행할 것
