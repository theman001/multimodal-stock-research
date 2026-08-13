"""수수료/슬리피지 (플레이스홀더 — CLAUDE.md / plan/05_backtesting.md 기준).

실제 수치는 구현 시점에 재확인이 필요하다 (특히 국내 거래세율은 정책적으로
단계적 변경되어 온 값이라 하드코딩 시 만료 위험이 있음, CLAUDE.md 명시).
전부 함수 파라미터로 노출해 하드코딩하지 않는다.
"""
from __future__ import annotations

KR_COMMISSION_RATE = 0.0003  # 왕복 매매수수료 약 0.03% (placeholder)
KR_TAX_RATE = 0.0018  # 거래세 약 0.15~0.18% 중 상단값 (placeholder — 최신 세율 재확인 필요)
US_COMMISSION_RATE = 0.0  # 대다수 리테일 브로커의 무수수료 정책 기준
US_SLIPPAGE_BPS = 5.0  # 보수적 placeholder


def kr_round_trip_cost(commission_rate: float = KR_COMMISSION_RATE, tax_rate: float = KR_TAX_RATE) -> float:
    return commission_rate + tax_rate


def us_round_trip_cost(commission_rate: float = US_COMMISSION_RATE, slippage_bps: float = US_SLIPPAGE_BPS) -> float:
    return commission_rate + slippage_bps / 10_000


def round_trip_cost_for_market(market_id: int) -> float:
    """market_id: 0=해외(US), 1=국내(KR) — CLAUDE.md 기준."""
    return kr_round_trip_cost() if market_id == 1 else us_round_trip_cost()
