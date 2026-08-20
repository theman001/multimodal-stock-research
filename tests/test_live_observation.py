"""build_today_observation()(라이브 단일일 경로)이 build_grid()(배치 경로)와
수치까지 동일한 정적 블록을 내는지 검증한다 — Phase 4 DoD "observation.py가
배치 경로와 수치 일치" 항목. model_v3 자체의 정확성은 tests/test_rl_panel_leakage.py가
이미 검증하므로, 여기서는 tests/test_rl_panel_leakage.py::test_score_model_v3_probabilities_scales_before_predicting와
동일한 패턴(XGBClassifier를 결정적 스텁으로 monkeypatch)으로 model_v3 로딩
자체를 우회한다."""
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import src.live.observation as observation_module
import src.rl.panel as panel_module
from src.features.indicators import BASE_FEATURE_COLUMNS, EVENT_FEATURE_COLUMNS
from src.live.observation import HeldPosition, build_today_observation
from src.rl.obs_scaler import fit_obs_scaler, save_obs_scaler, transform_obs_features
from src.rl.panel import MODEL_V3_PROB_COLUMN, build_grid, save_ticker_universe, static_feature_names

TICKERS = ["T0", "T1"]  # T0=US(market_id=0), T1=KR(market_id=1)
CHECKPOINT_NAME = "rl_policy_v1"


class _DeterministicModel:
    """실제 model_v3.json을 로드하지 않고, 입력 X의 첫 컬럼 값으로 결정적 prob을
    만든다(같은 (date,ticker) 행이면 배치/라이브 경로 어디서 스코어링해도 항상
    같은 값이 나옴 — score_model_v3_probabilities는 행 단위 stateless라 이
    성질이 성립해야 한다는 게 이 테스트의 전제)."""

    def load_model(self, path):
        pass

    def predict_proba(self, X):
        base = X[BASE_FEATURE_COLUMNS[0]].to_numpy(dtype=np.float64)
        prob = 1.0 / (1.0 + np.exp(-base))
        return np.column_stack([1 - prob, prob])


def _make_features_row(date: str, ticker: str, ticker_idx: int, day_idx: int) -> dict:
    signal = day_idx + ticker_idx * 10.0
    rec = {c: signal + i * 0.1 for i, c in enumerate(BASE_FEATURE_COLUMNS)}
    rec.update({c: signal * 0.5 for c in EVENT_FEATURE_COLUMNS})
    rec["date"] = pd.Timestamp(date)
    rec["ticker"] = ticker
    rec["market_id"] = 0 if ticker == "T0" else 1
    rec["size_id"] = 2
    return rec


def _setup_data_root(tmp_path, dates: list[str], monkeypatch, missing: set[tuple[str, str]] = frozenset()):
    """dates x TICKERS 전체 조합으로 features.parquet/ohlcv_meta.parquet을 만들되
    missing에 있는 (date,ticker)는 빼서 그날 그 티커가 마스킹되게 한다."""
    monkeypatch.setattr(panel_module, "XGBClassifier", _DeterministicModel)

    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir(parents=True)

    feature_rows, ohlcv_rows = [], []
    for day_idx, date in enumerate(dates):
        for ticker_idx, ticker in enumerate(TICKERS):
            if (date, ticker) in missing:
                continue
            feature_rows.append(_make_features_row(date, ticker, ticker_idx, day_idx))
            ohlcv_rows.append({"date": pd.Timestamp(date), "ticker": ticker, "close": 100.0 + day_idx + ticker_idx})

    features_df = pd.DataFrame(feature_rows)
    ohlcv_df = pd.DataFrame(ohlcv_rows)
    features_df.to_parquet(processed / "features.parquet", index=False)
    ohlcv_df.to_parquet(processed / "ohlcv_meta.parquet", index=False)

    # model_v3 스케일러(score_model_v3_probabilities가 요구) — 이 테스트는
    # _DeterministicModel이 실제 스케일링 결과를 쓰지 않으므로 identity에
    # 가까운 스케일러면 충분하다.
    model_v3_scaler = StandardScaler()
    model_v3_scaler.fit(pd.DataFrame({c: [0.0, 1.0] for c in list(BASE_FEATURE_COLUMNS) + list(EVENT_FEATURE_COLUMNS)}))
    with open(checkpoints / "scaler.pkl", "wb") as f:
        pickle.dump(model_v3_scaler, f)

    save_ticker_universe(TICKERS, tmp_path)

    # 참조용 배치 패널(build_grid 직접 호출) — 이 테스트의 "정답"
    full_features = features_df.copy()
    full_features[MODEL_V3_PROB_COLUMN] = panel_module.score_model_v3_probabilities(full_features, tmp_path)
    batch_panel = build_grid(full_features, ohlcv_df, TICKERS, include_event_features=True)

    obs_scaler = fit_obs_scaler(batch_panel, train_end_date=batch_panel.dates[-1])
    save_obs_scaler(obs_scaler, checkpoints / f"{CHECKPOINT_NAME}_scaler.pkl")

    scaled_batch = transform_obs_features(batch_panel, obs_scaler)
    return batch_panel, scaled_batch


def test_matches_batch_panel_static_block_with_no_positions(tmp_path, monkeypatch):
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    target_date = "2024-01-03"
    batch_panel, scaled_batch = _setup_data_root(tmp_path, dates, monkeypatch)
    t = list(batch_panel.dates).index(pd.Timestamp(target_date).to_datetime64())

    result = build_today_observation(
        data_root=tmp_path,
        checkpoints_dir=tmp_path / "checkpoints",
        checkpoint_name=CHECKPOINT_NAME,
        target_date=target_date,
        broker_positions={},
        broker_cash=1.0,
        nav_anchor=1.0,
        include_event_features=True,
    )

    d_static = len(static_feature_names(include_event_features=True))
    n = len(TICKERS)
    d_per_ticker = d_static + 3
    per_ticker_actual = result.obs[: n * d_per_ticker].reshape(n, d_per_ticker)

    expected_static = scaled_batch[t]  # (N, D_static)
    expected_dynamic = np.zeros((n, 3), dtype=np.float32)  # 포지션 없음 -> holding/return/weight 전부 0
    expected_per_ticker = np.concatenate(
        [expected_static[:, :-1], expected_dynamic, expected_static[:, -1:]], axis=1
    )

    np.testing.assert_allclose(per_ticker_actual, expected_per_ticker, atol=1e-5)
    assert result.tickers == TICKERS
    np.testing.assert_array_equal(result.valid_mask, np.array([True, True]))
    np.testing.assert_array_equal(result.market_id, np.array([0, 1]))


def test_dynamic_fields_reflect_broker_positions(tmp_path, monkeypatch):
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    target_date = "2024-01-03"
    _setup_data_root(tmp_path, dates, monkeypatch)

    # T0 종가(2024-01-03) = 100.0 + day_idx(2) + ticker_idx(0) = 102.0 (_make_features_row/ohlcv 생성 규칙)
    result = build_today_observation(
        data_root=tmp_path,
        checkpoints_dir=tmp_path / "checkpoints",
        checkpoint_name=CHECKPOINT_NAME,
        target_date=target_date,
        broker_positions={"T0": HeldPosition(qty=2.0, avg_entry_price=100.0)},
        broker_cash=50.0,
        nav_anchor=1.0,
        include_event_features=True,
    )

    close_t0 = 102.0
    expected_position_value = 2.0 * close_t0  # 204.0
    expected_nav = 50.0 + expected_position_value  # 254.0
    expected_unrealized = close_t0 / 100.0 - 1.0  # 0.02

    d_static = len(static_feature_names(include_event_features=True))
    d_per_ticker = d_static + 3
    per_ticker_actual = result.obs[: len(TICKERS) * d_per_ticker].reshape(len(TICKERS), d_per_ticker)
    holding_col, unrealized_col, weight_col = d_static - 1, d_static, d_static + 1

    assert per_ticker_actual[0, holding_col] == pytest.approx(1.0)
    assert per_ticker_actual[0, unrealized_col] == pytest.approx(expected_unrealized, abs=1e-5)
    assert per_ticker_actual[0, weight_col] == pytest.approx(expected_position_value / expected_nav, abs=1e-5)
    assert per_ticker_actual[1, holding_col] == pytest.approx(0.0)  # T1은 미보유

    assert result.nav == pytest.approx(expected_nav)
    assert result.cash == pytest.approx(50.0)
    assert result.position_value[0] == pytest.approx(expected_position_value)
    assert result.position_value[1] == pytest.approx(0.0)  # T1은 미보유

    portfolio_scalars = result.obs[-4:]
    cash_weight, nav_ratio, n_positions_frac, step_frac = portfolio_scalars
    assert cash_weight == pytest.approx(50.0 / expected_nav, abs=1e-5)
    assert nav_ratio == pytest.approx(expected_nav / 1.0 - 1.0, abs=1e-5)  # nav_anchor=1.0
    assert n_positions_frac == pytest.approx(0.5)  # 2종목 중 1종목 보유
    assert step_frac == pytest.approx(0.0)  # 라이브엔 에피소드가 없음(잠정 규칙)


def test_missing_today_features_raises_clear_error(tmp_path, monkeypatch):
    dates = ["2024-01-01", "2024-01-02"]  # target_date(01-03)가 아예 없음
    _setup_data_root(tmp_path, dates, monkeypatch)

    with pytest.raises(ValueError, match="오늘자 행이 없음"):
        build_today_observation(
            data_root=tmp_path,
            checkpoints_dir=tmp_path / "checkpoints",
            checkpoint_name=CHECKPOINT_NAME,
            target_date="2024-01-03",
            broker_positions={},
            broker_cash=1.0,
            nav_anchor=1.0,
            include_event_features=True,
        )


def test_lookback_window_resolves_market_id_for_ticker_masked_today(tmp_path, monkeypatch):
    """T1(KR)이 target_date 당일엔 row가 없어도(예: 그날만 데이터 공백), 최근
    lookback_days 안의 다른 날짜에 row가 있으면 market_id를 정상적으로 resolve해야
    한다(오늘 하루만 잘라 넘기면 panel.py::_ticker_market_ids가 깨지는 문제를
    회피하기 위한 lookback 윈도우 설계 — 모듈 docstring 참고)."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    target_date = "2024-01-03"
    _setup_data_root(tmp_path, dates, monkeypatch, missing={(target_date, "T1")})

    result = build_today_observation(
        data_root=tmp_path,
        checkpoints_dir=tmp_path / "checkpoints",
        checkpoint_name=CHECKPOINT_NAME,
        target_date=target_date,
        broker_positions={},
        broker_cash=1.0,
        nav_anchor=1.0,
        include_event_features=True,
    )

    assert result.market_id[1] == 1  # T1은 여전히 KR로 정확히 resolve됨
    assert result.valid_mask[1] == False  # 하지만 오늘 row는 없으므로 마스킹됨(거래 불가)


def test_masked_held_position_carries_forward_last_known_price(tmp_path, monkeypatch):
    """T0을 보유 중인데 target_date 당일 가격이 마스킹되면, 0으로 떨어뜨리지 않고
    윈도우 안 마지막 유효 종가를 그대로 들고 가야 한다(TradingEnv의 mark-to-market과
    동일 취지) — 그렇지 않으면 그 포지션의 실제 가치가 NAV에서 통째로 사라진다
    (재검토로 발견한 버그의 회귀 테스트)."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    target_date = "2024-01-03"
    _setup_data_root(tmp_path, dates, monkeypatch, missing={(target_date, "T0")})

    # T0의 2024-01-02 종가 = 100.0 + day_idx(1) + ticker_idx(0) = 101.0 (_setup_data_root 규칙)
    result = build_today_observation(
        data_root=tmp_path,
        checkpoints_dir=tmp_path / "checkpoints",
        checkpoint_name=CHECKPOINT_NAME,
        target_date=target_date,
        broker_positions={"T0": HeldPosition(qty=2.0, avg_entry_price=100.0)},
        broker_cash=50.0,
        nav_anchor=1.0,
        include_event_features=True,
    )

    last_known_close_t0 = 101.0
    expected_position_value = 2.0 * last_known_close_t0  # 202.0
    expected_nav = 50.0 + expected_position_value  # 252.0 — 0으로 떨어졌다면 50.0이었을 것
    expected_unrealized = last_known_close_t0 / 100.0 - 1.0

    d_static = len(static_feature_names(include_event_features=True))
    per_ticker_actual = result.obs[: len(TICKERS) * (d_static + 3)].reshape(len(TICKERS), d_static + 3)
    holding_col, unrealized_col, weight_col = d_static - 1, d_static, d_static + 1

    assert result.nav == pytest.approx(expected_nav)
    assert per_ticker_actual[0, holding_col] == pytest.approx(1.0)
    assert per_ticker_actual[0, unrealized_col] == pytest.approx(expected_unrealized, abs=1e-5)
    assert per_ticker_actual[0, weight_col] == pytest.approx(expected_position_value / expected_nav, abs=1e-5)
    # allocation.py::compute_target_allocations()가 SELL 처리 시 이 값을 그대로
    # position_values 딕셔너리에 넣어 쓸 수 있어야 한다(result.close[0]은 NaN이라
    # 이 용도로 못 씀 — 재검토로 발견한 API 설계 갭의 회귀 테스트).
    assert np.isnan(result.close[0])
    assert result.position_value[0] == pytest.approx(expected_position_value)


def test_non_finite_static_block_raises_instead_of_returning_bad_obs(tmp_path, monkeypatch):
    """지금까지 추가한 입력(broker_cash/nav_anchor/포지션) 검증을 전부 우회하는
    다른 경로(재사용 중인 transform_obs_features가 반환한 정적 블록 자체에
    이상치가 있는 경우 등)로 NaN이 새어들어와도 조용히 반환되면 안 된다 —
    최종 출력 자체를 마지막으로 한 번 더 검증하는 안전망(재검토로 발견)."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    _setup_data_root(tmp_path, dates, monkeypatch)

    real_transform = observation_module.transform_obs_features

    def _corrupting_transform(panel, scaler):
        out = real_transform(panel, scaler)
        out[..., 0] = np.nan  # 입력 검증으로는 못 잡는 경로를 시뮬레이션
        return out

    monkeypatch.setattr(observation_module, "transform_obs_features", _corrupting_transform)

    with pytest.raises(ValueError, match="유한하지 않은"):
        build_today_observation(
            data_root=tmp_path,
            checkpoints_dir=tmp_path / "checkpoints",
            checkpoint_name=CHECKPOINT_NAME,
            target_date="2024-01-03",
            broker_positions={},
            broker_cash=1.0,
            nav_anchor=1.0,
            include_event_features=True,
        )


def test_zero_avg_entry_price_does_not_produce_nan(tmp_path, monkeypatch):
    """avg_entry_price=0(비정상 데이터)은 division by zero로 NaN/Inf를 만들면
    안 된다 — 정책 입력에 NaN이 흘러들어가면 예측 자체가 오염된다."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    target_date = "2024-01-03"
    _setup_data_root(tmp_path, dates, monkeypatch)

    result = build_today_observation(
        data_root=tmp_path,
        checkpoints_dir=tmp_path / "checkpoints",
        checkpoint_name=CHECKPOINT_NAME,
        target_date=target_date,
        broker_positions={"T0": HeldPosition(qty=5.0, avg_entry_price=0.0)},
        broker_cash=50.0,
        nav_anchor=1.0,
        include_event_features=True,
    )

    assert np.isfinite(result.obs).all()
    d_static = len(static_feature_names(include_event_features=True))
    per_ticker_actual = result.obs[: len(TICKERS) * (d_static + 3)].reshape(len(TICKERS), d_static + 3)
    assert per_ticker_actual[0, d_static] == pytest.approx(0.0)  # unrealized_return 가드됨


def test_nan_broker_cash_raises_instead_of_corrupting_whole_observation(tmp_path, monkeypatch):
    """broker_cash가 NaN이면 nav 전체가 오염되고, 그 nav로 나누는
    cash_weight/nav_ratio/position_weight까지 120종목 전체 관측이 NaN으로
    물든다 — 개별 필드가 아니라 관측벡터 전체가 망가지는 문제라 조용히
    넘어가지 말고 여기서 막아야 한다(재검토로 발견한 버그의 회귀 테스트)."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    _setup_data_root(tmp_path, dates, monkeypatch)

    with pytest.raises(ValueError, match="broker_cash"):
        build_today_observation(
            data_root=tmp_path,
            checkpoints_dir=tmp_path / "checkpoints",
            checkpoint_name=CHECKPOINT_NAME,
            target_date="2024-01-03",
            broker_positions={},
            broker_cash=float("nan"),
            nav_anchor=1.0,
            include_event_features=True,
        )


@pytest.mark.parametrize("bad_anchor", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_nav_anchor_raises_instead_of_corrupting_nav_ratio(tmp_path, monkeypatch, bad_anchor):
    """nav_anchor는 호출자가 상태 파일 등에서 읽어오는 값이라 broker_cash와
    달리 브로커 API 응답은 아니지만, 손상된 상태 파일이면 똑같이 NaN/Inf/
    비양수가 들어올 수 있다. nav_ratio 채널 하나만 오염되는 것처럼 보여도
    신경망은 입력을 층마다 섞으므로 전체 행동 예측이 오염될 수 있다(재검토로
    발견한 버그의 회귀 테스트)."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    _setup_data_root(tmp_path, dates, monkeypatch)

    with pytest.raises(ValueError, match="nav_anchor"):
        build_today_observation(
            data_root=tmp_path,
            checkpoints_dir=tmp_path / "checkpoints",
            checkpoint_name=CHECKPOINT_NAME,
            target_date="2024-01-03",
            broker_positions={},
            broker_cash=1.0,
            nav_anchor=bad_anchor,
            include_event_features=True,
        )


def test_nan_position_qty_raises(tmp_path, monkeypatch):
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    _setup_data_root(tmp_path, dates, monkeypatch)

    with pytest.raises(ValueError, match="T0"):
        build_today_observation(
            data_root=tmp_path,
            checkpoints_dir=tmp_path / "checkpoints",
            checkpoint_name=CHECKPOINT_NAME,
            target_date="2024-01-03",
            broker_positions={"T0": HeldPosition(qty=float("inf"), avg_entry_price=100.0)},
            broker_cash=1.0,
            nav_anchor=1.0,
            include_event_features=True,
        )


def test_zero_qty_with_nan_avg_entry_price_does_not_raise(tmp_path, monkeypatch):
    """qty==0(완전 청산된 잔여 항목)이면 avg_entry_price는 애초에 계산에
    쓰이지 않는다 — 브로커가 그런 항목에 의미 없는 avg_entry_price(NaN 등)를
    돌려주는 흔한 경우까지 걸려 매일 불필요하게 막히면 안 된다(재검토로 발견)."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    _setup_data_root(tmp_path, dates, monkeypatch)

    result = build_today_observation(
        data_root=tmp_path,
        checkpoints_dir=tmp_path / "checkpoints",
        checkpoint_name=CHECKPOINT_NAME,
        target_date="2024-01-03",
        broker_positions={"T0": HeldPosition(qty=0.0, avg_entry_price=float("nan"))},
        broker_cash=1.0,
        nav_anchor=1.0,
        include_event_features=True,
    )
    assert result.nav == pytest.approx(1.0)


def test_non_positive_lookback_days_raises(tmp_path, monkeypatch):
    """lookback_days<=0은 사실상 '오늘 하루만' 넘기는 것과 같아져서, 이 함수가
    애초에 피하려 했던 market_id resolve 실패 버그가 재현된다 — 조용히
    허용하면 안 된다(재검토로 발견)."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    _setup_data_root(tmp_path, dates, monkeypatch)

    with pytest.raises(ValueError, match="lookback_days"):
        build_today_observation(
            data_root=tmp_path,
            checkpoints_dir=tmp_path / "checkpoints",
            checkpoint_name=CHECKPOINT_NAME,
            target_date="2024-01-03",
            broker_positions={},
            broker_cash=1.0,
            nav_anchor=1.0,
            include_event_features=True,
            lookback_days=0,
        )


def test_short_position_computes_correctly_signed_pnl_and_value(tmp_path, monkeypatch):
    """"롱 베이스, 숏은 전략적으로 허용"(2026-08-20 결정) — 관측 계층은 음수
    qty(숏)를 에러 없이 받아들이되, position_value/unrealized_return의 부호가
    실제 손익 방향과 일치해야 한다. 가격이 오르면 롱은 이익·숏은 손실인데,
    price/entry-1 공식을 부호 보정 없이 그대로 쓰면 숏의 손실을 이익으로
    잘못 표시하게 된다."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    target_date = "2024-01-03"
    _setup_data_root(tmp_path, dates, monkeypatch)

    # T0 종가(2024-01-03) = 102.0 (_setup_data_root 규칙) — 진입가(100.0) 대비 +2%
    result = build_today_observation(
        data_root=tmp_path,
        checkpoints_dir=tmp_path / "checkpoints",
        checkpoint_name=CHECKPOINT_NAME,
        target_date=target_date,
        broker_positions={"T0": HeldPosition(qty=-2.0, avg_entry_price=100.0)},
        broker_cash=300.0,
        nav_anchor=1.0,
        include_event_features=True,
    )

    close_t0 = 102.0
    expected_position_value = -2.0 * close_t0  # -204.0 — 숏은 음수 시가평가
    expected_nav = 300.0 + expected_position_value  # 96.0
    expected_unrealized = -(close_t0 / 100.0 - 1.0)  # -0.02 — 가격이 올라 숏은 손실(음수)

    assert result.position_value[0] == pytest.approx(expected_position_value)
    assert result.nav == pytest.approx(expected_nav)

    d_static = len(static_feature_names(include_event_features=True))
    per_ticker_actual = result.obs[: len(TICKERS) * (d_static + 3)].reshape(len(TICKERS), d_static + 3)
    holding_col, unrealized_col, weight_col = d_static - 1, d_static, d_static + 1

    assert per_ticker_actual[0, holding_col] == pytest.approx(1.0)  # 방향과 무관하게 "보유 중"
    assert per_ticker_actual[0, unrealized_col] == pytest.approx(expected_unrealized, abs=1e-5)
    assert per_ticker_actual[0, weight_col] == pytest.approx(expected_position_value / expected_nav, abs=1e-5)
    assert np.isfinite(result.obs).all()


def test_short_position_with_masked_price_combines_carry_forward_and_sign_fix(tmp_path, monkeypatch):
    """숏 포지션(부호 보정 필요)이면서 동시에 오늘 가격이 마스킹된(carry-forward
    필요) 경우 — 두 수정이 독립적으로 합성되는지 확인. 둘 다 순차적으로 적용되는
    코드 경로라 상호작용 버그가 있을 가능성은 낮지만, 재검토 과정에서 발견한
    두 별개 수정의 조합이라 명시적으로 잠가둔다."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    target_date = "2024-01-03"
    _setup_data_root(tmp_path, dates, monkeypatch, missing={(target_date, "T0")})

    # T0의 2024-01-02 종가 = 101.0(carry-forward 대상), 진입가 100.0 대비 +1%
    result = build_today_observation(
        data_root=tmp_path,
        checkpoints_dir=tmp_path / "checkpoints",
        checkpoint_name=CHECKPOINT_NAME,
        target_date=target_date,
        broker_positions={"T0": HeldPosition(qty=-2.0, avg_entry_price=100.0)},
        broker_cash=300.0,
        nav_anchor=1.0,
        include_event_features=True,
    )

    last_known_close_t0 = 101.0
    expected_position_value = -2.0 * last_known_close_t0  # -202.0
    expected_nav = 300.0 + expected_position_value  # 98.0
    expected_unrealized = -(last_known_close_t0 / 100.0 - 1.0)  # -0.01

    assert result.position_value[0] == pytest.approx(expected_position_value)
    assert result.nav == pytest.approx(expected_nav)

    d_static = len(static_feature_names(include_event_features=True))
    per_ticker_actual = result.obs[: len(TICKERS) * (d_static + 3)].reshape(len(TICKERS), d_static + 3)
    unrealized_col = d_static
    assert per_ticker_actual[0, unrealized_col] == pytest.approx(expected_unrealized, abs=1e-5)
    assert np.isfinite(result.obs).all()


def test_unknown_ticker_in_broker_positions_raises(tmp_path, monkeypatch):
    """broker_positions에 120종목 유니버스 밖 티커가 섞여 있으면(수동 거래,
    스핀오프 등) 그 포지션의 실제 가치가 관측벡터에 반영될 방법이 없어 NAV가
    조용히 틀어진다 — 무시하지 말고 막아야 한다(재검토로 발견)."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    _setup_data_root(tmp_path, dates, monkeypatch)

    with pytest.raises(ValueError, match="ZZZZ"):
        build_today_observation(
            data_root=tmp_path,
            checkpoints_dir=tmp_path / "checkpoints",
            checkpoint_name=CHECKPOINT_NAME,
            target_date="2024-01-03",
            broker_positions={"ZZZZ": HeldPosition(qty=1.0, avg_entry_price=100.0)},
            broker_cash=1.0,
            nav_anchor=1.0,
            include_event_features=True,
        )


def test_unknown_ticker_with_zero_qty_does_not_raise(tmp_path, monkeypatch):
    """브로커가 완전 청산 직후에도 한동안 qty=0인 잔여 포지션 항목을 돌려주는
    경우가 흔하다 — NAV에 영향이 없는데도 유니버스 밖이라는 이유로 매일
    막히면 안 된다(재검토로 발견)."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    _setup_data_root(tmp_path, dates, monkeypatch)

    result = build_today_observation(
        data_root=tmp_path,
        checkpoints_dir=tmp_path / "checkpoints",
        checkpoint_name=CHECKPOINT_NAME,
        target_date="2024-01-03",
        broker_positions={"ZZZZ": HeldPosition(qty=0.0, avg_entry_price=100.0)},
        broker_cash=1.0,
        nav_anchor=1.0,
        include_event_features=True,
    )
    assert result.nav == pytest.approx(1.0)


def test_timezone_aware_target_date_matches_naive_equivalent(tmp_path, monkeypatch):
    """향후 KST cron이 tz-aware Timestamp를 넘길 수 있다 — features.parquet의
    tz-naive 날짜와 비교가 조용히 어긋나거나 하루 밀리면 안 된다."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    _setup_data_root(tmp_path, dates, monkeypatch)

    kwargs = dict(
        data_root=tmp_path,
        checkpoints_dir=tmp_path / "checkpoints",
        checkpoint_name=CHECKPOINT_NAME,
        broker_positions={},
        broker_cash=1.0,
        nav_anchor=1.0,
        include_event_features=True,
    )
    naive_result = build_today_observation(target_date="2024-01-03", **kwargs)
    tz_aware_result = build_today_observation(
        target_date=pd.Timestamp("2024-01-03 07:00:00", tz="Asia/Seoul"), **kwargs
    )

    np.testing.assert_allclose(naive_result.obs, tz_aware_result.obs, atol=1e-8)
