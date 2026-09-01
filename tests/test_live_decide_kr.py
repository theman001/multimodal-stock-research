"""decide_kr.py 통합 테스트 — test_live_decide_us.py와 동일 구조(공유 관측/
추론/배분/안전장치 모듈을 그대로 검증), market 관련 부분만 KR로 교체.
네트워크를 타는 수집 함수(collect_kr/collect_events_kr/score_events_kr)만
monkeypatch로 no-op 처리하고, 나머지는 전부 실제로 돌린다."""
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import src.live.decide_kr as decide_kr_module
import src.rl.panel as panel_module
from src.features.indicators import BASE_FEATURE_COLUMNS, EVENT_FEATURE_COLUMNS
from src.live.allocation import TargetOrder
from src.live.broker.base import BrokerAdapter, OrderResult
from src.live.decide_kr import decision_path, run_decide
from src.live.observation import HeldPosition
from src.rl.obs_scaler import fit_obs_scaler, save_obs_scaler, transform_obs_features
from src.rl.panel import MODEL_V3_PROB_COLUMN, build_grid, save_ticker_universe
from src.rl.train_agent import save_checkpoint, train_policy

TICKERS = ["005930", "000660"]  # 둘 다 KR(market_id=1) — decide_kr.py 테스트 범위
N_WARMUP_DAYS = 65


class _FakeBroker(BrokerAdapter):
    def __init__(self, positions=None, cash=1000.0, nav=1000.0, market_open=True):
        super().__init__(mode="paper")
        self._positions = positions or {}
        self._cash = cash
        self._nav = nav
        self._market_open = market_open
        self.orders_submitted: list[tuple] = []

    def get_positions(self):
        return dict(self._positions)

    def get_cash(self):
        return self._cash

    def get_nav(self):
        return self._nav

    def place_market_order(self, ticker, side, qty=None, notional=None, client_order_id=None):
        self.orders_submitted.append((ticker, side, qty, notional))
        return OrderResult(ticker=ticker, side=side, status="filled", filled_qty=qty, filled_avg_price=100.0)

    def is_market_open(self):
        return self._market_open

    def count_open_orders(self):
        return 0


class _DeterministicModelV3:
    """실제 model_v3.json을 로드하지 않는 결정적 스텁(tests/test_live_observation.py와 동일 패턴)."""

    def load_model(self, path):
        pass

    def predict_proba(self, X):
        base = X[BASE_FEATURE_COLUMNS[0]].to_numpy(dtype=np.float64)
        prob = 1.0 / (1.0 + np.exp(-base))
        return np.column_stack([1 - prob, prob])


def _make_ohlcv(tickers, n_days, market_id=1, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(0)
    rows = []
    for ticker in tickers:
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, size=n_days))
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date, "ticker": ticker, "market_id": market_id, "size_id": 2,
                    "open": close[i], "high": close[i] * 1.01, "low": close[i] * 0.99,
                    "close": close[i], "volume": 1_000_000, "market_cap": 1e10,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    checkpoints = tmp_path / "checkpoints"
    processed.mkdir(parents=True)
    checkpoints.mkdir(parents=True)

    # collect_kr/collect_events_kr/score_events_kr는 실제 네트워크(KRX/DART)를
    # 부르므로 no-op으로 대체한다 — 이 테스트가 검증하려는 건 그 함수들의
    # 정확성이 아니라(각자 별도 테스트 있음) decide_kr.py가 그것들을 올바른
    # 순서로 오케스트레이션하고, 나머지(merge/observation/infer/allocation/
    # safety)와 올바르게 이어붙이는지다.
    monkeypatch.setattr(decide_kr_module, "collect_kr", lambda data_root: None)
    monkeypatch.setattr(decide_kr_module, "collect_events_kr", lambda data_root: None)
    monkeypatch.setattr(decide_kr_module, "score_events_kr", lambda data_root: None)

    # ohlcv_meta_kr.parquet를 직접 준비 -> merge_ohlcv_meta()가 진짜로 돌아
    # ohlcv_meta.parquet을 만든다(mock 아님).
    ohlcv = _make_ohlcv(TICKERS, N_WARMUP_DAYS, market_id=1)
    ohlcv.to_parquet(processed / "ohlcv_meta_kr.parquet", index=False)

    # 배치 features.parquet — 실제 build_features()처럼 오늘(마지막 거래일) 행은 뺀다.
    monkeypatch.setattr(panel_module, "XGBClassifier", _DeterministicModelV3)
    frames = []
    from src.features.indicators import compute_features

    for ticker in TICKERS:
        g = ohlcv.loc[ohlcv["ticker"] == ticker]
        computed = compute_features(g).dropna(subset=BASE_FEATURE_COLUMNS + ["target"])
        frames.append(computed[["date", "ticker", "market_id", "size_id", *BASE_FEATURE_COLUMNS, "target"]])
    features = pd.concat(frames, ignore_index=True)
    features["target"] = features["target"].astype(int)
    features.to_parquet(processed / "features.parquet", index=False)

    # 이벤트 파일 — 비어 있어도 되지만 스키마는 맞춰야 함(collect_events_kr이
    # no-op이므로 events_kr.parquet/events_sentiment_kr.parquet 자체는 미리
    # 준비해줘야 함). join_key는 rcept_no(US의 accessionNumber에 대응).
    empty_events = pd.DataFrame(columns=["ticker", "corp_code", "pblntf_ty", "report_nm", "rcept_no", "rcept_dt", "flr_nm"])
    empty_events.to_parquet(processed / "events_kr.parquet", index=False)
    pd.DataFrame(columns=["ticker", "rcept_no", "score"]).to_parquet(
        processed / "events_sentiment_kr.parquet", index=False
    )

    # model_v3 스케일러
    model_v3_scaler = StandardScaler()
    model_v3_scaler.fit(pd.DataFrame({c: [0.0, 1.0] for c in list(BASE_FEATURE_COLUMNS) + list(EVENT_FEATURE_COLUMNS)}))
    with open(checkpoints / "scaler.pkl", "wb") as f:
        pickle.dump(model_v3_scaler, f)

    save_ticker_universe(TICKERS, tmp_path)

    # RL 정책 체크포인트 — 실제로 작게 학습(test_live_infer.py와 동일 패턴).
    full_features = features.copy()
    # 배치 경로엔 event feature 컬럼이 없으므로 0으로 채워 패널 스키마를 맞춘다.
    for col in EVENT_FEATURE_COLUMNS:
        full_features[col] = 0.0
    full_features[MODEL_V3_PROB_COLUMN] = panel_module.score_model_v3_probabilities(full_features, tmp_path)
    ohlcv_for_panel = ohlcv[["date", "ticker", "close"]]
    batch_panel = build_grid(full_features, ohlcv_for_panel, TICKERS, include_event_features=True)

    from src.rl.trading_env import TradingEnv

    obs_scaler = fit_obs_scaler(batch_panel, train_end_date=batch_panel.dates[-1])
    model, _ = train_policy(
        batch_panel,
        train_end_idx=len(batch_panel.dates) - 1,
        total_timesteps=32,
        episode_length_days=min(3, len(batch_panel.dates)),
        seed=0,
        ppo_params={"n_steps": 16, "batch_size": 8},
    )
    save_checkpoint(
        model, obs_scaler, checkpoints, "rl_policy_v1", {"include_event_features": True}
    )

    return tmp_path


def test_run_decide_produces_decision_file_with_expected_structure(data_root):
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)
    # target_date는 ohlcv의 마지막 거래일이어야 한다(오늘자 행 계산 대상).
    ohlcv = pd.read_parquet(data_root / "processed" / "ohlcv_meta_kr.parquet")
    target_date = ohlcv["date"].max()

    path = run_decide(data_root=data_root, target_date=target_date, broker=broker)

    assert path == decision_path(data_root, target_date)
    assert path.exists()
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["target_date"] == str(target_date.date())
    assert "orders" in payload
    for order in payload["orders"]:
        assert order["market_id"] == 1  # TICKERS가 전부 KR
        assert order["side"] in ("buy", "sell")


def test_run_decide_does_not_call_broker_place_order(data_root):
    """decide_kr.py는 절대 주문을 내지 않는다(plan/10 §B-6 — 결정만 저장)."""
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)
    ohlcv = pd.read_parquet(data_root / "processed" / "ohlcv_meta_kr.parquet")
    target_date = ohlcv["date"].max()

    run_decide(data_root=data_root, target_date=target_date, broker=broker)

    assert broker.orders_submitted == []


def test_run_decide_bootstraps_state_on_first_run(data_root):
    from src.live.safety import load_state

    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)
    ohlcv = pd.read_parquet(data_root / "processed" / "ohlcv_meta_kr.parquet")
    target_date = ohlcv["date"].max()

    assert load_state(data_root, "kr") is None
    run_decide(data_root=data_root, target_date=target_date, broker=broker)

    state = load_state(data_root, "kr")
    assert state is not None
    assert state.nav_anchor == pytest.approx(1000.0)


def test_run_decide_raises_on_reconciliation_mismatch(data_root):
    """어제 마지막으로 기록된 상태와 브로커가 지금 보고하는 상태가 크게
    다르면(수동 개입 등) 결정을 만들지 않고 막아야 한다."""
    from src.live.safety import LiveState, save_state

    ohlcv = pd.read_parquet(data_root / "processed" / "ohlcv_meta_kr.parquet")
    target_date = ohlcv["date"].max()

    # 어제 상태를 미리 기록(현금 1000) — 오늘 브로커는 현금 100만 보고(큰 괴리)
    save_state(data_root, "kr", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    broker = _FakeBroker(positions={}, cash=100.0, nav=100.0)

    with pytest.raises(RuntimeError, match="재구성 불일치"):
        run_decide(data_root=data_root, target_date=target_date, broker=broker)


def test_run_decide_suppresses_buys_when_kill_switch_triggers(data_root):
    """전일 대비 NAV가 크게 하락했으면 신규 BUY가 억제돼야 한다.

    킬스위치는 obs_result.nav(observation.py가 cash+position_value로 직접
    재계산한 값 — broker.get_nav()가 아님)와 state.nav를 비교한다. 재구성
    체크는 cash/positions만 비교하므로, state.cash/positions를 브로커의
    현재 값과 동일하게 맞춘 채 state.nav만 크게 높게 기록해두면(예: 그
    사이 포지션이 있었다가 청산됐다고 가정) 재구성은 통과하면서 킬스위치만
    독립적으로 발동시킬 수 있다."""
    from src.live.safety import LiveState, save_state

    ohlcv = pd.read_parquet(data_root / "processed" / "ohlcv_meta_kr.parquet")
    target_date = ohlcv["date"].max()

    save_state(data_root, "kr", LiveState(positions={}, cash=1000.0, nav=2000.0, nav_anchor=2000.0, updated_at="2024-01-01"))
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)  # obs_result.nav = cash+0 = 1000.0 -> 전일(2000) 대비 -50%

    import json

    path = run_decide(data_root=data_root, target_date=target_date, broker=broker)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["suppress_buys"] is True
    assert all(o["side"] != "buy" for o in payload["orders"])


def test_run_decide_sends_warning_notification_when_kill_switch_triggers(data_root):
    """plan/10 §B-5: 킬스위치 발동은 명시적으로 알림이 요구되는 지점이다."""
    from src.live.safety import LiveState, save_state

    ohlcv = pd.read_parquet(data_root / "processed" / "ohlcv_meta_kr.parquet")
    target_date = ohlcv["date"].max()

    save_state(data_root, "kr", LiveState(positions={}, cash=1000.0, nav=2000.0, nav_anchor=2000.0, updated_at="2024-01-01"))
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)

    calls = []
    original = decide_kr_module.send_notification

    def _capture(text, level="info"):
        calls.append((text, level))
        return True

    decide_kr_module.send_notification = _capture
    try:
        run_decide(data_root=data_root, target_date=target_date, broker=broker)
    finally:
        decide_kr_module.send_notification = original

    levels = [level for _text, level in calls]
    assert "warning" in levels
    assert any("킬스위치" in text for text, _level in calls)


def test_run_decide_sends_error_notification_on_reconciliation_failure_and_still_raises(data_root):
    from src.live.safety import LiveState, save_state

    ohlcv = pd.read_parquet(data_root / "processed" / "ohlcv_meta_kr.parquet")
    target_date = ohlcv["date"].max()

    save_state(data_root, "kr", LiveState(positions={}, cash=1000.0, nav=1000.0, nav_anchor=1000.0, updated_at="2024-01-01"))
    broker = _FakeBroker(positions={}, cash=100.0, nav=100.0)  # 큰 괴리 -> 재구성 실패

    calls = []
    original = decide_kr_module.send_notification

    def _capture(text, level="info"):
        calls.append((text, level))
        return True

    decide_kr_module.send_notification = _capture
    try:
        with pytest.raises(RuntimeError, match="재구성 불일치"):
            run_decide(data_root=data_root, target_date=target_date, broker=broker)
    finally:
        decide_kr_module.send_notification = original

    errors = [(t, lvl) for t, lvl in calls if lvl == "error"]
    assert len(errors) == 1
    assert "decide_kr 실패" in errors[0][0]
    assert any("decide 시작" in t for t, _ in calls)


def test_run_decide_sends_info_notification_on_success(data_root):
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)
    ohlcv = pd.read_parquet(data_root / "processed" / "ohlcv_meta_kr.parquet")
    target_date = ohlcv["date"].max()

    calls = []
    original = decide_kr_module.send_notification

    def _capture(text, level="info"):
        calls.append((text, level))
        return True

    decide_kr_module.send_notification = _capture
    try:
        run_decide(data_root=data_root, target_date=target_date, broker=broker)
    finally:
        decide_kr_module.send_notification = original

    assert [lvl for _, lvl in calls] == ["info", "info"]  # 시작 핑 + 완료
    assert any("decide 시작" in t for t, _ in calls)
    assert any("decide 완료" in t for t, _ in calls)


def test_run_decide_second_call_same_day_blocked_by_lock_if_concurrent(data_root):
    """중복 실행 방지 락이 실제로 걸리는지 — acquire_run_lock을 직접 잡아둔
    상태에서 run_decide를 부르면 막혀야 한다."""
    from src.live.safety import acquire_run_lock

    ohlcv = pd.read_parquet(data_root / "processed" / "ohlcv_meta_kr.parquet")
    target_date = ohlcv["date"].max()
    broker = _FakeBroker(positions={}, cash=1000.0, nav=1000.0)

    with acquire_run_lock(data_root, "decide_kr"):
        with pytest.raises(RuntimeError, match="이미 진행 중"):
            run_decide(data_root=data_root, target_date=target_date, broker=broker)
