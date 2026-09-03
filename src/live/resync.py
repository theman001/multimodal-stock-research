"""브로커 실제 계좌 상태로 `state_{track}.json` 을 강제 재동기화하는 수동 복구 도구.

execute 가 미체결·크래시 등으로 state 를 실제 계좌와 어긋나게 남겨(2026-08-31/
09-02 발생) 다음 decide 가 재구성 체크로 멈췄을 때 쓴다. `nav_anchor` 와
`last_executed_date` 는 기존 값을 보존한다(state 파일이 없으면 각각 현재 nav /
None). 정상 운영 경로는 이 스크립트를 쓰지 않는다 —
`safety.resync_state_if_settled()` 가 pending_settle 플래그를 보고 자동으로 한다.

사용: python -m src.live.resync us     (또는 kr)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

from src.config import get_data_root
from src.live.safety import LiveState, load_state, save_state


def run(track: str) -> LiveState:
    if track == "us":
        from src.live.broker.alpaca import AlpacaBroker

        broker = AlpacaBroker(mode="paper")
    elif track == "kr":
        from src.live.broker.kis import KISBroker

        broker = KISBroker(mode="paper")
    else:
        raise SystemExit(f"track 은 'us' 또는 'kr' 이어야 함: {track!r}")

    data_root = get_data_root()
    prev = load_state(data_root, track)
    positions = broker.get_positions()
    cash = broker.get_cash()
    nav = broker.get_nav()
    state = LiveState(
        positions=positions,
        cash=cash,
        nav=nav,
        nav_anchor=prev.nav_anchor if prev is not None else nav,
        updated_at=datetime.now(timezone.utc).isoformat(),
        last_executed_date=prev.last_executed_date if prev is not None else None,
        pending_settle=False,
    )
    save_state(data_root, track, state)
    print(
        f"[resync] {track}: cash={cash:,.2f} nav={nav:,.2f} positions={len(positions)} "
        f"nav_anchor={state.nav_anchor:,.2f} last_executed_date={state.last_executed_date}"
    )
    return state


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "us")
