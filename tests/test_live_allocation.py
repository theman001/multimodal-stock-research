import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import kr_round_trip_cost, us_round_trip_cost
from src.live.allocation import TargetOrder, compute_target_allocations
from src.rl.panel import Panel
from src.rl.trading_env import ACTION_BUY, ACTION_HOLD, ACTION_SELL, TradingEnv

US_HALF_COST = us_round_trip_cost() / 2.0
KR_HALF_COST = kr_round_trip_cost() / 2.0


def _make_panel(n_dates, close, valid_mask, market_id) -> Panel:
    n_tickers = close.shape[1]
    close = np.asarray(close, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=np.float32)
    features = np.zeros((n_dates, n_tickers, 3), dtype=np.float32)
    features[..., -1] = valid_mask
    dates = np.array(
        [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(n_dates)]
    ).astype("datetime64[ns]")
    tickers = [f"T{i}" for i in range(n_tickers)]
    return Panel(
        dates=dates,
        tickers=tickers,
        feature_names=["f0", "f1", "valid_mask"],
        features=features,
        close=close,
        valid_mask=valid_mask,
        market_id=np.array(market_id, dtype=np.int64),
    )


def test_buy_cap_matches_trading_env_numerically():
    """TradingEnv.step()과 동일 시나리오(tests/test_trading_env.py의
    test_capital_allocation_respects_cap_and_does_not_redistribute_leftover)를
    실제로 돌려, compute_target_allocations()의 BUY notional(비용 차감 전)이
    env.position_value(비용 차감 후)와 정확히 half_cost만큼만 차이 나는지
    (즉 배분 산술 자체는 완전히 동일한지) 직접 비교한다."""
    close = np.full((3, 2), 100.0)
    panel = _make_panel(3, close, np.ones((3, 2)), market_id=[0, 1])  # T0=US, T1=KR
    env = TradingEnv(panel, date_start_idx=0, date_end_idx=2, random_start=False, max_position_weight=0.3)
    env.reset()
    env.step(np.array([ACTION_BUY, ACTION_BUY]))

    orders = compute_target_allocations(
        action=np.array([ACTION_BUY, ACTION_BUY]),
        tickers=["T0", "T1"],
        current_holdings=set(),
        position_values={},
        cash=1.0,
        nav=1.0,
        valid_mask=np.array([1.0, 1.0]),
        has_price=np.array([True, True]),
        max_position_weight=0.3,
    )

    by_ticker = {o.ticker: o for o in orders}
    assert by_ticker["T0"].side == "buy"
    assert by_ticker["T1"].side == "buy"
    # equal_share=1.0/2=0.5, cap=0.3*1.0=0.3 -> 둘 다 캡에 걸림(재분배 없음)
    assert by_ticker["T0"].notional == pytest.approx(0.3)
    assert by_ticker["T1"].notional == pytest.approx(0.3)
    # env.position_value는 비용 차감 후 값이므로, notional*(1-half_cost)와 일치해야
    # "배분 산술은 동일, 비용만 라이브에서 제외했다"는 설계 의도가 실제로 맞는지 확인된다.
    assert by_ticker["T0"].notional * (1 - US_HALF_COST) == pytest.approx(env.position_value[0])
    assert by_ticker["T1"].notional * (1 - KR_HALF_COST) == pytest.approx(env.position_value[1])


def test_sell_proceeds_fund_same_step_buy():
    """TradingEnv.step()은 SELL을 먼저 정산해 그 현금을 같은 스텝의 BUY 배분에
    쓴다(trading_env.py 189-223행) — compute_target_allocations()도 동일해야 한다."""
    orders = compute_target_allocations(
        action=np.array([ACTION_SELL, ACTION_BUY]),
        tickers=["T0", "T1"],
        current_holdings={"T0"},
        position_values={"T0": 1.0},
        cash=0.0,
        nav=1.0,
        valid_mask=np.array([1.0, 1.0]),
        has_price=np.array([True, True]),
        max_position_weight=1.0,
    )
    by_ticker = {o.ticker: o for o in orders}
    assert by_ticker["T0"].side == "sell"
    assert by_ticker["T0"].notional is None
    assert by_ticker["T1"].side == "buy"
    assert by_ticker["T1"].notional == pytest.approx(1.0)  # cash_after_sells=0+1.0=1.0, 매수대상 1종목


@pytest.mark.parametrize(
    "action,holdings,valid,has_price,expected_tickers",
    [
        # 보유 중 BUY는 no-op(이미 보유 중이면 재매수 없음)
        (np.array([ACTION_BUY]), {"T0"}, np.array([1.0]), np.array([True]), set()),
        # 미보유 HOLD는 no-op
        (np.array([ACTION_HOLD]), set(), np.array([1.0]), np.array([True]), set()),
        # 미보유 SELL은 no-op(팔 게 없음)
        (np.array([ACTION_SELL]), set(), np.array([1.0]), np.array([True]), set()),
        # valid_mask=0(오늘 row 없음)인 티커는 BUY 신호가 있어도 no-op
        (np.array([ACTION_BUY]), set(), np.array([0.0]), np.array([True]), set()),
        # has_price=False(가격 없음)면 BUY도 SELL도 no-op
        (np.array([ACTION_BUY]), set(), np.array([1.0]), np.array([False]), set()),
    ],
)
def test_noop_scenarios_produce_no_orders(action, holdings, valid, has_price, expected_tickers):
    orders = compute_target_allocations(
        action=action,
        tickers=["T0"],
        current_holdings=holdings,
        position_values={"T0": 1.0} if holdings else {},
        cash=1.0,
        nav=1.0,
        valid_mask=valid,
        has_price=has_price,
        max_position_weight=1.0,
    )
    assert {o.ticker for o in orders} == expected_tickers


def test_negative_cash_produces_no_buy_orders_not_negative_notional():
    """cash/nav가 비정상적으로 음수면(재구성 불일치 등 safety.py가 원래 걸러야
    할 상황) equal_share가 음수가 될 수 있다 — min()이 그 음수를 그대로 통과시켜
    '매수 금액이 음수인 매수 주문'을 만들면 안 된다(재검토로 발견한 버그의
    회귀 테스트)."""
    orders = compute_target_allocations(
        action=np.array([ACTION_BUY]),
        tickers=["T0"],
        current_holdings=set(),
        position_values={},
        cash=-10.0,
        nav=-10.0,
        valid_mask=np.array([1.0]),
        has_price=np.array([True]),
        max_position_weight=1.0,
    )
    assert orders == []  # 음수 notional 주문이 아니라 아예 주문이 없어야 함


def test_sell_on_short_position_is_noop_with_warning_not_a_wrong_direction_order():
    """숏 포지션(position_values 음수)에 SELL 결정이 들어오면, 이 함수는 여전히
    '청산=SELL'이라는 롱 전제라 그대로 side="sell" 주문을 내면 실제로 필요한
    커버링(매수)과 반대 방향이 된다. 처음엔 예외로 막았지만(1차 수정), 그러면
    정책이 그 숏 티커에 SELL을 낼 때마다 나머지 종목들의 결정까지 매일 통째로
    막혀 '숏은 전략적으로 유지'라는 의도와 부딪힌다(재검토로 발견) — 그 티커만
    조용히 건너뛰고(주문 미생성) 경고만 남기는 걸로 수정."""
    with pytest.warns(UserWarning, match="숏"):
        orders = compute_target_allocations(
            action=np.array([ACTION_SELL, ACTION_HOLD]),
            tickers=["T0", "T1"],
            current_holdings={"T0"},
            position_values={"T0": -50.0},  # 숏 포지션의 시가평가액은 음수
            cash=0.0,
            nav=50.0,
            valid_mask=np.array([1.0, 1.0]),
            has_price=np.array([True, True]),
        )
    assert orders == []  # 잘못된 방향 주문도, 다른 주문도 없음(no-op)


def test_sell_on_long_position_still_works_after_short_guard():
    """숏 가드가 정상 롱 청산까지 막지 않는지 확인 — position_values가 양수(롱)면
    기존처럼 정상적으로 SELL 주문이 나가야 한다."""
    orders = compute_target_allocations(
        action=np.array([ACTION_SELL]),
        tickers=["T0"],
        current_holdings={"T0"},
        position_values={"T0": 50.0},
        cash=0.0,
        nav=50.0,
        valid_mask=np.array([1.0]),
        has_price=np.array([True]),
    )
    assert len(orders) == 1
    assert orders[0].side == "sell"


def test_short_sell_skip_does_not_fund_same_step_buy():
    """숏은 오늘 정산되지 않는(no-op) 포지션이므로, 그 시가평가액이
    cash_after_sells에 들어가 다른 종목의 BUY 자금으로 쓰이면 안 된다 —
    실제로 청산(현금화)되지 않았는데 그 돈을 쓴 것처럼 계산하면 이중으로
    틀어진다."""
    with pytest.warns(UserWarning, match="숏"):
        orders = compute_target_allocations(
            action=np.array([ACTION_SELL, ACTION_BUY]),
            tickers=["T0", "T1"],
            current_holdings={"T0"},
            position_values={"T0": -50.0},  # 숏, no-op으로 건너뜀
            cash=0.0,
            nav=50.0,
            valid_mask=np.array([1.0, 1.0]),
            has_price=np.array([True, True]),
            max_position_weight=1.0,
        )
    # cash_after_sells = 0.0(숏 시가평가액 제외) -> equal_share=0.0 -> alloc=0 -> BUY 주문 없음
    assert orders == []


def test_nan_cash_produces_no_buy_orders_not_nan_notional():
    """min()/max()가 NaN을 다루는 방식(Python 내장 min/max는 NaN과의 비교가 항상
    False라 첫 인자를 그대로 반환)에 우연히 기대는 안전장치라, 이 동작을 테스트로
    고정해둔다 — 나중에 np.minimum/np.maximum으로 무심코 바꾸면 이 성질이
    깨질 수 있다(NaN 비교 의미가 다름)."""
    orders = compute_target_allocations(
        action=np.array([ACTION_BUY]),
        tickers=["T0"],
        current_holdings=set(),
        position_values={},
        cash=float("nan"),
        nav=1.0,
        valid_mask=np.array([1.0]),
        has_price=np.array([True]),
        max_position_weight=1.0,
    )
    assert orders == []


def test_mismatched_array_lengths_raise_clear_error():
    """tickers/action/valid_mask/has_price는 전부 같은 순서의 티커별 배열이어야
    하는 암묵적 계약이 있다 — 길이가 어긋나면 numpy의 일반적인 broadcast
    에러 대신 어떤 파라미터가 문제인지 알려주는 이름 붙은 에러여야 한다
    (재검토로 발견)."""
    with pytest.raises(ValueError, match="길이가 서로 다름"):
        compute_target_allocations(
            action=np.array([ACTION_BUY, ACTION_HOLD]),  # 길이 2
            tickers=["T0"],  # 길이 1
            current_holdings=set(),
            position_values={},
            cash=1.0,
            nav=1.0,
            valid_mask=np.array([1.0]),
            has_price=np.array([True]),
        )


def test_returns_target_order_dataclass_instances():
    orders = compute_target_allocations(
        action=np.array([ACTION_BUY]),
        tickers=["T0"],
        current_holdings=set(),
        position_values={},
        cash=1.0,
        nav=1.0,
        valid_mask=np.array([1.0]),
        has_price=np.array([True]),
    )
    assert len(orders) == 1
    assert isinstance(orders[0], TargetOrder)
