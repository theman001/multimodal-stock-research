"""resync.py — 수동 복구 도구가 브로커 상태로 state 를 덮어쓰되 기준선을 보존하는지."""
import pytest

import src.live.resync as resync_module
from src.live.observation import HeldPosition
from src.live.safety import LiveState, load_state, save_state


class _StubBroker:
    def __init__(self, positions, cash, nav):
        self._p, self._c, self._n = positions, cash, nav

    def get_positions(self):
        return dict(self._p)

    def get_cash(self):
        return self._c

    def get_nav(self):
        return self._n


@pytest.fixture(autouse=True)
def _data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(resync_module, "get_data_root", lambda: tmp_path)
    return tmp_path


def test_resync_overwrites_state_from_broker_preserving_anchor(_data_root, monkeypatch):
    save_state(
        _data_root, "us",
        LiveState(positions={}, cash=275.19, nav=99170.0, nav_anchor=100000.0,
                  updated_at="x", last_executed_date="2026-09-02", pending_settle=True),
    )
    broker = _StubBroker({"XOM": HeldPosition(24.0, 159.9)}, cash=6201.73, nav=99176.07)
    monkeypatch.setattr(resync_module, "AlpacaBroker", lambda mode="paper": broker, raising=False)
    import src.live.broker.alpaca as alpaca_mod
    monkeypatch.setattr(alpaca_mod, "AlpacaBroker", lambda mode="paper": broker)

    resync_module.run("us")

    s = load_state(_data_root, "us")
    assert s.cash == pytest.approx(6201.73)
    assert s.nav_anchor == pytest.approx(100000.0)      # 보존
    assert s.last_executed_date == "2026-09-02"          # 보존
    assert s.pending_settle is False
    assert "XOM" in s.positions


def test_resync_rejects_unknown_track(_data_root):
    with pytest.raises(SystemExit):
        resync_module.run("jp")
