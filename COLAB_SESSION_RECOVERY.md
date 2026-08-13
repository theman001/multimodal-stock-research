# Colab 세션 끊김 복구 가이드

캐시(`${DATA_ROOT}/raw/{us,kr}/events/{text,sentiment}/*.{txt,json}`)는 문서
1건이 끝날 때마다 즉시 Drive에 저장되므로, 세션이 끊겨도 손실은 최대 처리 중이던
1건뿐이다. 아래 순서대로 재연결하면 이어서 진행된다.

## 1. 노트북 셀에서 실행 (Drive 마운트는 반드시 여기서)

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. 쉘(터미널)에서 한 번에 실행

```bash
cd /content
git clone https://github.com/theman001/multimodal-stock-research.git repo && cd repo
pip install -r requirements.txt
# Colab 기본 torch(2.11.x)용으로 깔린 torchvision/torchaudio가 requirements.txt의
# torch==2.13.0과 CUDA 버전이 안 맞아 transformers import가 깨짐(BertForSequenceClassification
# 로드 시 내부적으로 둘 다 거쳐감) — 이 프로젝트는 vision/audio 기능을 안 쓰므로 둘 다 제거.
pip uninstall -y torchvision torchaudio

export DATA_ROOT="/content/drive/MyDrive/주식프로젝트/multimodal-stock-research"
mkdir -p "$DATA_ROOT/reports"   # 없으면 진행상황 체크포인트 쓸 때 FileNotFoundError
echo "US sentiment: $(ls "$DATA_ROOT/raw/us/events/sentiment" | wc -l)건"
echo "KR sentiment: $(ls "$DATA_ROOT/raw/kr/events/sentiment" | wc -l)건"
```

캐시 개수가 0이면 `DATA_ROOT` 경로/마운트를 먼저 의심할 것.

**`.env`는 clone에 안 딸려온다**(`.gitignore` 처리됨) — `SEC_EDGAR_USER_AGENT`, `DART_API_KEY`가
없으면 네트워크 요청도 안 하고 즉시 실패한다(실패 건수가 수백 건씩 순식간에 뛰면 이거다). 로컬
`.env`를 `repo/` 안에 업로드하거나 `export SEC_EDGAR_USER_AGENT=... DART_API_KEY=...`로 직접
설정할 것. 다 됐으면 실행:
```bash
python -m src.data_collection.score_events
```

## 로컬 백업이 필요할 때

```bash
# 로컬에서 아카이브 생성
tar -czf sentiment_cache.tar.gz \
  data/raw/us/events/text data/raw/us/events/sentiment \
  data/raw/kr/events/text data/raw/kr/events/sentiment \
  data/processed/events_us.parquet data/processed/events_kr.parquet
```

```bash
# Colab 쉘에서 해제 (data/ 껍질을 벗기며 DATA_ROOT 바로 아래로)
tar -xzf "/content/sentiment_cache.tar.gz" -C "$DATA_ROOT" --strip-components=1
```

## 흔한 실수

- Python 코드(`drive.mount`, `import os` 등)는 노트북 셀 전용 — 터미널에 붙여넣으면 `bash: import: command not found`.
- `--strip-components=1` 빠뜨리면 `${DATA_ROOT}/data/raw/...`로 한 겹 밀려서 캐시를 못 찾음.
- 재연결 시 GPU 재배정은 보장되지 않음(속도만 영향, 데이터 손실과 무관).
- `ModuleNotFoundError: Could not import module 'BertForSequenceClassification'`가 나면 torch와 torchvision/torchaudio(Colab 기본 CUDA 버전으로 고정됨) 간 CUDA 버전 불일치다 — `pip uninstall -y torchvision torchaudio`로 해결(위 셀 블록에 이미 포함). 하나만 지우면 다음 실행에서 나머지 하나가 또 걸린다.
