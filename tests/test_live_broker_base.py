import pytest

from src.live.broker.base import BrokerAdapter


def test_cannot_instantiate_abstract_broker_adapter_directly():
    with pytest.raises(TypeError):
        BrokerAdapter()


def test_default_mode_is_paper():
    """CLAUDE.md Phase 4 핵심 결정 1번 — 기본값은 절대 live=True가 아니어야 함.
    구체 서브클래스로 최소 확인(추상 클래스 자체는 인스턴스화 불가)."""

    class _MinimalBroker(BrokerAdapter):
        def get_positions(self):
            return {}

        def get_cash(self):
            return 0.0

        def get_nav(self):
            return 0.0

        def place_market_order(self, ticker, side, qty=None, notional=None):
            raise NotImplementedError

        def is_market_open(self):
            return False

        def count_open_orders(self):
            return 0

    assert _MinimalBroker().mode == "paper"


class _SettleBroker(BrokerAdapter):
    """count_open_orders()가 호출마다 시퀀스에서 하나씩 뱉는다(마지막 값 반복)."""

    def __init__(self, seq):
        super().__init__(mode="paper")
        self._seq = list(seq)
        self.calls = 0

    def get_positions(self):
        return {}

    def get_cash(self):
        return 0.0

    def get_nav(self):
        return 0.0

    def place_market_order(self, ticker, side, qty=None, notional=None):
        raise NotImplementedError

    def is_market_open(self):
        return True

    def count_open_orders(self):
        self.calls += 1
        return self._seq.pop(0) if len(self._seq) > 1 else self._seq[0]


def test_wait_until_orders_settle_returns_zero_once_orders_clear():
    b = _SettleBroker([3, 1, 0])
    assert b.wait_until_orders_settle(timeout_s=5.0, poll_s=0.0) == 0
    assert b.calls == 3  # 0을 볼 때까지 폴링


def test_wait_until_orders_settle_returns_remaining_count_on_timeout():
    b = _SettleBroker([2])  # 계속 2건 미체결
    remaining = b.wait_until_orders_settle(timeout_s=0.02, poll_s=0.0)
    assert remaining == 2  # 타임아웃 — execute_*.py가 이걸로 경고 알림을 보냄
