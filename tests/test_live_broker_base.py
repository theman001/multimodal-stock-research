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

    assert _MinimalBroker().mode == "paper"
