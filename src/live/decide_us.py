"""Phase 4 US 트랙 — 매일 07:00 KST 목표 행동을 계산해 저장한다(주문 없음).

plan/10 §B-6/§C-2. 이 시각이면 전날 KR 마감분과 그날 새벽 US 마감분이 둘 다
확정돼 있다. 브로커에 주문을 내지 않는다 — 그건 execute_us.py(당일 저녁,
US 장 시작 시각)의 몫이다.

관측/추론은 120종목 전체 유니버스를 대상으로 한다(정책이 그렇게 학습됐으므로
— CLAUDE.md "Phase 4 구체 사양" 참고) — 그중 US가 아닌 티커에 대한 결정도
저장은 하되(각 주문에 `market_id`를 함께 기록), 실제로 브로커에 주문을 내는
건 execute_us.py가 `market_id==0`(US)인 것만 필터링해서 한다(plan/10 §B-6:
"저장된 결정 파일을 읽어 그 시장 소속 티커만 필터링해 브로커에 주문"). 파일명에
`us_` 접두사를 붙이는 이유: KR 트랙 세션이 같은 날짜에 대해 독립적으로
`decide_kr.py`를 돌리면 서로 다른 브로커 계좌 컨텍스트(현금/포지션)로 계산한
별개의 결정이 나오므로, 시장 구분 없는 `{date}.json` 하나를 공유하면 두
트랙이 서로 덮어쓴다.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from src.config import get_data_root
from src.data_collection.collect import collect_us
from src.data_collection.collect_events import collect_events_us
from src.data_collection.score_events import score_events_us
from src.live.allocation import compute_target_allocations
from src.live.broker.alpaca import AlpacaBroker
from src.live.broker.base import BrokerAdapter
from src.live.daily_pipeline import build_live_features_window, merge_ohlcv_meta
from src.live.infer import predict_action
from src.live.observation import build_today_observation
from src.live.safety import (
    acquire_run_lock,
    check_daily_loss_kill_switch,
    check_reconciliation,
    enforce_order_caps,
    load_or_bootstrap_state,
)

CHECKPOINT_NAME = "rl_policy_v1"
INCLUDE_EVENT_FEATURES = True  # rl_policy_v1이 include_event_features=True로 학습됨(infer.py가 불일치 시 막음)
LOOKBACK_DAYS = 30
MARKET_ID_US = 0


def _kst_today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()


def decision_path(data_root: Path, target_date: pd.Timestamp) -> Path:
    return data_root / "live" / "decisions" / f"us_{target_date.date()}.json"


def _save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp_path, path)


def run_decide(
    data_root: Path | None = None,
    target_date: pd.Timestamp | str | None = None,
    checkpoints_dir: Path | None = None,
    broker: BrokerAdapter | None = None,
) -> Path:
    """관측→추론→배분을 실행해 오늘자 결정을 저장하고 그 경로를 반환한다.
    브로커 주문은 호출하지 않는다(get_positions/get_cash/get_nav 같은 조회성
    메서드만 씀 — place_market_order를 아예 호출하지 않으므로 execute_us.py와
    달리 broker.mode="live" 주입을 막는 별도 방어가 필요 없다). `broker`를
    주면(테스트용) 그대로 쓰고, 아니면 `AlpacaBroker(mode="paper")`를 새로
    만든다 — **기본값은 절대 live가 아니다**(CLAUDE.md Phase 4 핵심 결정 1번)."""
    data_root = data_root or get_data_root()
    checkpoints_dir = checkpoints_dir or (data_root / "checkpoints")
    target_date = pd.Timestamp(target_date) if target_date is not None else _kst_today()
    broker = broker or AlpacaBroker(mode="paper")

    with acquire_run_lock(data_root, "decide_us"):
        current_positions = broker.get_positions()
        current_cash = broker.get_cash()
        current_nav_from_broker = broker.get_nav()

        state = load_or_bootstrap_state(data_root, current_positions, current_cash, current_nav_from_broker)
        check_reconciliation(current_positions, current_cash, state)

        collect_us(data_root)
        merge_ohlcv_meta(data_root)
        collect_events_us(data_root)
        score_events_us(data_root)

        features_override = build_live_features_window(
            data_root,
            target_date=target_date,
            lookback_days=LOOKBACK_DAYS,
            market_id=MARKET_ID_US,
            include_event_features=INCLUDE_EVENT_FEATURES,
        )

        obs_result = build_today_observation(
            data_root=data_root,
            checkpoints_dir=checkpoints_dir,
            checkpoint_name=CHECKPOINT_NAME,
            target_date=target_date,
            broker_positions=current_positions,
            broker_cash=current_cash,
            nav_anchor=state.nav_anchor,
            include_event_features=INCLUDE_EVENT_FEATURES,
            lookback_days=LOOKBACK_DAYS,
            features_df_override=features_override,
        )

        action = predict_action(checkpoints_dir, CHECKPOINT_NAME, obs_result.obs, INCLUDE_EVENT_FEATURES)

        suppress_buys = check_daily_loss_kill_switch(obs_result.nav, state.nav)

        current_holdings = {t for t, p in current_positions.items() if p.qty != 0}
        position_values = {
            t: float(v) for t, v in zip(obs_result.tickers, obs_result.position_value) if v != 0
        }
        orders = compute_target_allocations(
            action=action,
            tickers=obs_result.tickers,
            current_holdings=current_holdings,
            position_values=position_values,
            cash=obs_result.cash,
            nav=obs_result.nav,
            valid_mask=obs_result.valid_mask,
            has_price=obs_result.has_price,
        )
        if suppress_buys:
            orders = [o for o in orders if o.side != "buy"]
        orders = enforce_order_caps(orders, nav=obs_result.nav)

        market_id_by_ticker = dict(zip(obs_result.tickers, obs_result.market_id))
        payload = {
            "target_date": str(target_date.date()),
            "decided_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            "nav": obs_result.nav,
            "cash": obs_result.cash,
            "suppress_buys": suppress_buys,
            "orders": [
                {
                    "ticker": o.ticker,
                    "side": o.side,
                    "notional": o.notional,
                    "market_id": int(market_id_by_ticker[o.ticker]),
                }
                for o in orders
            ],
        }
        path = decision_path(data_root, target_date)
        _save_json_atomic(path, payload)

    return path


if __name__ == "__main__":
    run_decide()
