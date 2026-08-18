"""Phase 3 — 120종목 포트폴리오 RL 환경을 위한 관측 패널 구성.

CLAUDE.md "Phase 3 구체 사양" > "관측(observation) 공간" 기준. `features.parquet`을
고정된 `(date, ticker)` 그리드로 재구성해서, 종목별 정적 피처 블록(D_static차원) x
120종목 x 전체 거래일의 3차원 배열을 만든다. 여기서 "정적"이란 포트폴리오 상태(보유
여부/평가손익/비중)와 달리 시장 데이터만으로 결정되는 부분이라는 뜻 — 그 3개
필드는 `src/rl/trading_env.py`(Phase 3 다음 모듈)가 스텝마다 계산해서 이 뒤에
이어붙인다.

D_static = BASE_FEATURE_COLUMNS(13) + model_v3_prob(1) + EVENT_FEATURE_COLUMNS(0/4,
include_event_features로 토글) + market_id(1) + size_id(1) + valid_mask(1)
= 17 또는 21. (여기에 trading_env.py가 3개를 더해 CLAUDE.md에 명시된 D=20/24가 된다.)

핵심 불변식(leakage-guard):
- 그 날 그 티커의 row가 없으면 0-채움 + valid_mask=0. **forward-fill 금지** —
  과거 값을 그대로 끌어오면 "마지막 관측 이후 며칠 지났는지"가 다른 필드들에
  암묵적으로 새어들어간다(event_features.py가 NO_PRIOR_EVENT_SENTINEL을 쓴 것과
  동일한 이유로, valid_mask를 "정보 없음"의 유일한 명시적 채널로 둔다).
- model_v3는 13개 기술지표만으로 학습됐다 — `src.models.train._prepare_X`는 Phase 2
  모듈 4 이후 FEATURE_COLUMNS(17개)를 쓰므로 그대로 재사용하면 model_v3에 맞지
  않는 컬럼 수가 들어간다. 이 파일에서 model_v3 전용으로 13개만 별도 준비한다.
- **model_v3는 스케일링된 입력으로 학습됐다.** `src/models/build_split.py`가
  `fit_scaler`/`transform`(Train에만 fit)으로 표준화한 `train.parquet`을 model_v3가
  학습했으므로, 추론 시에도 반드시 같은 방식으로 스케일링해야 한다 — raw 값을
  그대로 넣으면 트리 분기 임계값이 완전히 다른 분포를 보고 사실상 무작위에
  가까운 예측이 나온다(실측: raw 입력과 스케일링된 입력의 예측 상관계수 0.044,
  거의 무관함. 코드화 전 리뷰로 발견해 수정 — data/reports/phase3_reward_clipping_investigation.md
  옆에 별도 기록 없이 이 주석과 커밋 메시지로 남김). `data/checkpoints/scaler.pkl`은
  Phase 2 모듈 4 이후 FEATURE_COLUMNS(17개) 기준으로 다시 fit됐지만, StandardScaler는
  컬럼별로 독립적으로 평균/표준편차를 계산하므로 13개 BASE_FEATURE_COLUMNS에
  대한 변환값은 model_v3가 원래 학습 때 본 것과 동일하다(실측으로 확인: 이
  스케일러로 변환한 test.parquet 구간 정확도가 model_v3.meta.json의 원래 기록
  0.5169399830938293과 정확히 일치).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.config import get_data_root
from src.features.indicators import BASE_FEATURE_COLUMNS, EVENT_FEATURE_COLUMNS
from src.models.scaling import load_scaler, transform as scale_transform
from src.models.train import MARKET_ID_CATEGORIES, META_COLUMNS, SIZE_ID_CATEGORIES

MODEL_V3_PROB_COLUMN = "model_v3_prob"


@dataclass
class Panel:
    dates: np.ndarray  # (T,) datetime64[ns], 오름차순, features.parquet의 unique date
    tickers: list[str]  # 길이 N, 고정 순서(TICKER_UNIVERSE)
    feature_names: list[str]  # 길이 D_static, 마지막 원소는 항상 "valid_mask"
    features: np.ndarray  # (T, N, D_static) float32
    close: np.ndarray  # (T, N) float32, 해당 (date,ticker)에 유효한 row가 없으면 NaN
    valid_mask: np.ndarray  # (T, N) float32(0.0/1.0) — features[..., -1]과 동일한 값
    market_id: np.ndarray  # (N,) int64, 티커별 고정 시장 코드(0=US,1=KR).
    # features[..., market_id_idx]도 같은 정보를 담지만 그건 날짜별 그리드라
    # 마스킹된(0-채움) 행에서 읽으면 진짜 값 대신 0(US처럼 보임)이 나온다 —
    # trading_env.py의 거래비용 계산처럼 "그 티커가 어느 시장인지"가 마스킹
    # 여부와 무관하게 항상 정확해야 하는 곳에선 이 필드를 써야 한다.


def static_feature_names(include_event_features: bool) -> list[str]:
    """정적 피처 블록의 채널 이름을 순서대로 반환한다(마지막은 항상 valid_mask)."""
    names = list(BASE_FEATURE_COLUMNS) + [MODEL_V3_PROB_COLUMN]
    if include_event_features:
        names = names + list(EVENT_FEATURE_COLUMNS)
    return names + ["market_id", "size_id", "valid_mask"]


def _ticker_universe_path(data_root: Path) -> Path:
    return data_root / "checkpoints" / "rl_ticker_universe.json"


def save_ticker_universe(tickers: list[str], data_root: Path) -> None:
    """행동 인덱스 i가 항상 같은 티커를 가리키도록 순서를 고정 저장한다.

    src.models.train의 MARKET_ID_CATEGORIES/SIZE_ID_CATEGORIES가 카테고리 코드를
    명시 고정해 train/test 간 암묵적으로 어긋나는 걸 막은 것과 동일한 이유.
    """
    path = _ticker_universe_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tickers, ensure_ascii=False, indent=2), encoding="utf-8")


def load_ticker_universe(data_root: Path) -> list[str]:
    return json.loads(_ticker_universe_path(data_root).read_text(encoding="utf-8"))


def _prepare_x_for_model_v3(df: pd.DataFrame) -> pd.DataFrame:
    X = df[BASE_FEATURE_COLUMNS + META_COLUMNS].copy()
    X["market_id"] = pd.Categorical(X["market_id"], categories=MARKET_ID_CATEGORIES)
    X["size_id"] = pd.Categorical(X["size_id"], categories=SIZE_ID_CATEGORIES)
    return X


def score_model_v3_probabilities(features_df: pd.DataFrame, data_root: Path) -> np.ndarray:
    """features_df의 각 행에 대해 model_v3의 P(up)을 오프라인 배치로 계산한다.

    trading_env.py가 매 step()마다 재추론하지 않도록 패널 구성 시점에 한 번만
    계산한다. features_df와 같은 행 순서로 반환하므로(조인이 아니라 컬럼 선택만
    수행) 정렬 문제가 애초에 발생하지 않는다.

    model_v3는 스케일링된 입력으로 학습됐으므로(모듈 docstring 참고), 여기서도
    반드시 같은 스케일러로 transform()한 뒤에 넣어야 한다 — raw 값을 넣으면 안 됨.
    """
    model = XGBClassifier()
    model.load_model(str(data_root / "checkpoints" / "model_v3.json"))

    scaler = load_scaler(data_root / "checkpoints" / "scaler.pkl")
    scaled_df = scale_transform(features_df, scaler)

    X = _prepare_x_for_model_v3(scaled_df)
    return model.predict_proba(X)[:, 1]


def build_grid(
    features_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    tickers: list[str],
    include_event_features: bool = True,
) -> Panel:
    """순수 함수: 이미 model_v3_prob 컬럼이 붙은 features_df + close가 있는
    ohlcv_df를 고정 (date, tickers) 그리드로 재구성한다. 파일 IO나 모델 로딩이
    없어 독립적으로 유닛테스트하기 쉽다(build_panel()과의 관계는
    indicators.compute_features()와 build_features.py의 관계와 동일).
    """
    names = static_feature_names(include_event_features)
    value_cols = [c for c in names if c != "valid_mask"]

    # np.array(sorted(...))는 pandas Timestamp의 리스트를 dtype=object 배열로
    # 만들어버릴 수 있다(datetime64[ns]가 아니게 됨) — 그러면 이후
    # obs_scaler.fit_obs_scaler의 `panel.dates <= cutoff` 같은 비교가 조용히
    # 깨진다(실제로 실 데이터로 겪은 버그). pd.DatetimeIndex로 명시 변환해 확실히
    # datetime64[ns] 배열이 되게 한다.
    dates = pd.DatetimeIndex(sorted(features_df["date"].unique())).to_numpy()
    full_index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    feat_indexed = features_df.set_index(["date", "ticker"])[value_cols]
    if feat_indexed.index.duplicated().any():
        raise ValueError("features_df에 (date, ticker) 중복 행이 있음 — 패널 구성 불가")
    feat_full = feat_indexed.reindex(full_index)

    # market_id는 모든 유효 행에 항상 값이 있으므로(결측 없는 메타컬럼), 이 컬럼의
    # NaN 여부만으로 "그 (date,ticker)에 원래 row가 있었는가"를 판정할 수 있다.
    valid = feat_full["market_id"].notna().to_numpy()
    feat_full = feat_full.fillna(0.0)

    n_dates, n_tickers, n_value_cols = len(dates), len(tickers), len(value_cols)
    grid = np.zeros((n_dates * n_tickers, n_value_cols + 1), dtype=np.float32)
    grid[:, :n_value_cols] = feat_full.to_numpy(dtype=np.float32)
    grid[:, n_value_cols] = valid.astype(np.float32)
    features_grid = grid.reshape(n_dates, n_tickers, n_value_cols + 1)
    valid_mask_grid = valid.astype(np.float32).reshape(n_dates, n_tickers)

    close_indexed = ohlcv_df.set_index(["date", "ticker"])["close"]
    if close_indexed.index.duplicated().any():
        raise ValueError("ohlcv_df에 (date, ticker) 중복 행이 있음 — 패널 구성 불가")
    close_full = close_indexed.reindex(full_index)
    close_grid = close_full.to_numpy(dtype=np.float32).reshape(n_dates, n_tickers)

    market_id_array = _ticker_market_ids(features_df, tickers)

    return Panel(
        dates=dates,
        tickers=list(tickers),
        feature_names=names,
        features=features_grid,
        close=close_grid,
        valid_mask=valid_mask_grid,
        market_id=market_id_array,
    )


def _ticker_market_ids(features_df: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    """티커별 market_id를 날짜 그리드와 무관하게(마스킹에 영향받지 않게) 구한다.

    features[..., market_id_idx]는 그 날짜의 row가 없으면 0으로 채워지므로,
    하필 그 티커가 마스킹된 행에서 market_id를 읽으면 실제로 KR(1)인 티커가
    US(0)처럼 보이는 오류가 생길 수 있다 — 이 함수는 그 문제를 원천 차단한다.
    """
    per_ticker = features_df.groupby("ticker")["market_id"]
    if not per_ticker.nunique().eq(1).all():
        raise ValueError("일부 티커의 market_id가 시점에 따라 다름 — 예상치 못한 데이터")
    lookup = per_ticker.first()
    return lookup.reindex(tickers).to_numpy(dtype=np.int64)


def build_panel(data_root: Path | None = None, include_event_features: bool = True) -> Panel:
    """Phase 3 관측 패널 진입점.

    ${DATA_ROOT}/processed/features.parquet + ohlcv_meta.parquet을 읽어 티커
    유니버스를 고정 저장하고, model_v3 확률을 배치로 계산한 뒤 build_grid()로
    (T, N, D_static) 패널을 만든다.
    """
    data_root = data_root or get_data_root()
    processed = data_root / "processed"

    features_df = pd.read_parquet(processed / "features.parquet")
    ohlcv_df = pd.read_parquet(processed / "ohlcv_meta.parquet")[["date", "ticker", "close"]]

    tickers = sorted(features_df["ticker"].unique())
    save_ticker_universe(tickers, data_root)

    features_df = features_df.copy()
    features_df[MODEL_V3_PROB_COLUMN] = score_model_v3_probabilities(features_df, data_root)

    return build_grid(features_df, ohlcv_df, tickers, include_event_features=include_event_features)
