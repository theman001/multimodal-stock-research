---
name: leakage-guard
description: multimodal-stock-research 저장소에서 데이터 수집(src/data_collection), 피처 엔지니어링(src/features), Train/Test 분할 코드를 작성하거나 수정할 때, 또는 progress_log.json의 해당 step을 "done"으로 표시하기 전에 반드시 실행하는 데이터 누수(look-ahead bias) 셀프 점검 체크리스트.
---

# Leakage Guard

이 프로젝트(`CLAUDE.md` 참고)는 데이터 누수에 특히 취약한 3개 지점을 갖고 있다: (1) 롤링 기술지표, (2) `Market_Id`/`Size_Id` 메타피처의 point-in-time 라벨링, (3) 스케일러 fit 범위. 코드를 "완료"로 표시하기 전에 아래 체크리스트를 순서대로 실행한다.

## 1. 롤링 지표 shift 점검

```bash
grep -n "rolling(" src/features/*.py
grep -n "center=True" src/features/*.py   # 결과가 있으면 반드시 제거 — 미래 데이터 포함됨
grep -n "shift(-" src/features/*.py        # 타겟 계산(target) 외에 음수 shift가 있으면 누수 의심
```

- `rolling(...).mean()` 등은 기본적으로 현재 행 포함 과거만 사용하므로 그 자체로는 안전하지만 `center=True`는 절대 금지
- `shift(-N)`(미래 참조)은 타겟 변수 계산에만 허용된다. 피처 계산 코드에 있으면 즉시 수정

## 2. Point-in-time 라벨링 점검

- `size_id`/`market_id`를 만드는 코드가 "현재 시점 스냅샷 1개"를 전체 기간에 broadcast하고 있지 않은지 확인 (예: `df["size_id"] = 2` 같은 상수 할당은 위험 신호)
- 실제로 종목 하나를 골라, 정기변경일(6월/12월) 전후로 `size_id` 값이 코드 상에서 바뀔 수 있는 구조인지 확인. 안 바뀌면 point-in-time이 아니라 오늘 기준 고정 라벨을 쓰고 있다는 뜻
- `tests/test_point_in_time_labels.py`가 존재하고 통과하는지 확인, 없으면 작성 요구

## 3. 스케일러 fit 범위 점검

```bash
grep -n "\.fit(" src/**/*.py src/*.py 2>/dev/null
```

- 각 `.fit(` 호출의 인자가 Train 서브셋 변수인지 확인 (전체 데이터프레임이나 Test가 섞인 변수로 `fit`하면 누수)
- Test/Val에는 `.transform()`만 있어야 하며 `.fit_transform()`이 Test 쪽에 있으면 즉시 수정

## 4. Train/Test 경계 embargo 점검

- Train 최대 날짜와 Test 최소 날짜 사이 간격이 60 캘린더데이 이상인지 확인 (`CLAUDE.md`의 MA60 대응 embargo 규칙)
- `tests/test_train_test_embargo.py` 통과 확인

## 5. 테스트 실행

```bash
pytest tests/ -v
```

전부 통과해야 `progress_log.json`의 해당 step을 `done`으로 표시할 수 있다. 하나라도 실패하면 `status: "failed"`로 기록하고 원인을 `blockers`에 남긴 뒤 수정한다.

## 이 체크리스트를 건너뛰면 안 되는 이유

이 프로젝트의 핵심 가설("Market_Id/Size_Id 조건에 따라 지표 유의성이 달라진다")은, 메타피처 자체나 롤링 피처에 미래 정보가 섞이면 검증 자체가 무의미해진다. 특히 방향성 적중률이 55%를 크게 넘는 결과가 나오면 모델이 좋아서가 아니라 누수일 가능성을 먼저 의심할 것 (`CLAUDE.md` 모델/평가 원칙 참고).
