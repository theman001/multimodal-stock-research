"""Phase 1 백테스팅 진입점. CLAUDE.md / plan/05_backtesting.md 기준.

Test 구간에서 최신 모델 체크포인트로 예측 → 수수료/슬리피지 반영 전략 시뮬레이션 →
Market_Id x Size_Id 6개 셀 각각에 대해 랜덤워크(N=1000) 대비 비교 + 이항검정
(Bonferroni 보정) + 부트스트랩 신뢰구간을 계산하고 리포트를 생성한다.

산출물: ${DATA_ROOT}/reports/backtest_report_v{N}.md
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import xgboost as xgb

from src.backtest.significance import BONFERRONI_ALPHA, N_CELLS, binomial_test_direction, bootstrap_return_ci
from src.backtest.simulate import (
    add_net_return,
    add_realized_return,
    buy_and_hold_return,
    cumulative_return,
    daily_portfolio_returns,
    random_walk_total_returns,
)
from src.config import get_data_root
from src.models.train import _prepare_X

CELL_LABELS = {
    (1, 2): "KR-대형",
    (1, 1): "KR-중형",
    (1, 0): "KR-소형",
    (0, 2): "US-대형",
    (0, 1): "US-중형",
    (0, 0): "US-소형",
}


def _latest_model_version(checkpoints_dir: Path) -> int:
    versions = []
    for p in checkpoints_dir.glob("model_v*.json"):
        if p.name.endswith(".meta.json"):
            continue
        versions.append(int(p.stem.replace("model_v", "")))
    if not versions:
        raise RuntimeError("모델 체크포인트가 없음 — 먼저 src.models.train을 실행할 것")
    return max(versions)


def _load_model(checkpoints_dir: Path, version: int) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(str(checkpoints_dir / f"model_v{version}.json"))
    return model


def _cell_analysis(sub: pd.DataFrame, label: str) -> dict:
    n_total = len(sub)
    actual_up = (sub["realized_return"] > 0).astype(int)
    n_correct = int((sub["predicted_up"] == actual_up).sum())
    binom = binomial_test_direction(n_correct, n_total)

    daily = daily_portfolio_returns(sub)
    cum = cumulative_return(daily)
    total_return = float(cum.iloc[-1]) if len(cum) else float("nan")

    rw_totals = random_walk_total_returns(sub, n_sims=1000, seed=42)
    percentile = float((rw_totals < total_return).mean() * 100)

    boot = bootstrap_return_ci(daily.to_numpy())
    bh = buy_and_hold_return(sub)

    return {
        "cell": label,
        **binom,
        "strategy_total_return": total_return,
        "buy_and_hold_return": bh,
        "random_walk_percentile": percentile,
        "bootstrap_ci_low": boot["ci_low"],
        "bootstrap_ci_high": boot["ci_high"],
    }


def run(data_root: Path | None = None, model_version: int | None = None) -> pd.DataFrame:
    data_root = data_root or get_data_root()
    processed_dir = data_root / "processed"
    checkpoints_dir = data_root / "checkpoints"
    reports_dir = data_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    test = pd.read_parquet(processed_dir / "test.parquet")
    ohlcv = pd.read_parquet(processed_dir / "ohlcv_meta.parquet")[["date", "ticker", "close"]]

    if model_version is None:
        model_version = _latest_model_version(checkpoints_dir)
    model = _load_model(checkpoints_dir, model_version)

    X_test = _prepare_X(test)
    proba = model.predict_proba(X_test)[:, 1]
    test = test.copy()
    test["predicted_up"] = (proba >= 0.5).astype(int)

    df = test[["date", "ticker", "market_id", "size_id", "predicted_up"]].merge(
        ohlcv, on=["date", "ticker"], how="left"
    )
    df = add_realized_return(df)
    # 종목별 test 구간 마지막 날은 익일 종가가 없어 realized_return이 NaN -> 백테스트 대상에서 제외
    df = df.dropna(subset=["realized_return"]).reset_index(drop=True)
    df = add_net_return(df, df["predicted_up"].to_numpy())

    cell_results = []
    for (market_id, size_id), label in CELL_LABELS.items():
        sub = df[(df["market_id"] == market_id) & (df["size_id"] == size_id)]
        if sub.empty:
            continue
        cell_results.append(_cell_analysis(sub, label))

    overall = _cell_analysis(df, "전체")
    results_df = pd.DataFrame(cell_results)

    report_path = _write_report(results_df, overall, model_version, reports_dir)
    print(f"[run_backtest] 리포트 저장 -> {report_path}")
    print(results_df.to_string(index=False))

    return results_df


def _write_report(results_df: pd.DataFrame, overall: dict, model_version: int, reports_dir: Path) -> Path:
    lines = []
    lines.append(f"# Phase 1 백테스트 리포트 v{model_version}")
    lines.append("")
    lines.append(f"생성 시각: {date.today().isoformat()}")
    lines.append("")
    lines.append(
        "**비용 가정 (placeholder, `src/backtest/costs.py`에 파라미터로 노출됨 — 하드코딩 아님)**: "
        "국내 왕복 수수료 0.03% + 거래세 0.18% / 해외 커미션 0 + 슬리피지 5bp. "
        "국내 거래세율은 정책적으로 변경되어 온 값이라 실제 적용 전 재확인 필요 (CLAUDE.md 명시)."
    )
    lines.append("")
    lines.append(
        "**전략 가정**: 모델이 익일 상승을 예측한 날에만 매수 후 익일 매도(왕복비용 부과), "
        "그 외에는 현금 보유. 연속 상승 예측이 이어져도 매일 재진입으로 취급하는 보수적 근사."
    )
    lines.append("")

    lines.append("## 전체 결과")
    lines.append("")
    lines.append(f"- 방향성 적중률: {overall['accuracy']:.4f} ({overall['n_correct']}/{overall['n_total']})")
    lines.append(f"- 이항검정 p-value (H0: 적중률=50%, 단측): {overall['p_value']:.4f}")
    lines.append(
        f"- Bonferroni 보정({N_CELLS}개 셀, α={BONFERRONI_ALPHA:.4f}) 후 유의: "
        f"{'유의함' if overall['significant_bonferroni'] else '유의하지 않음'}"
    )
    lines.append(f"- 전략 누적수익률(비용 반영, 동일가중 포트폴리오): {overall['strategy_total_return']:.4%}")
    lines.append(f"- Buy & Hold 참고 벤치마크(비용 미반영): {overall['buy_and_hold_return']:.4%}")
    lines.append(f"- 랜덤워크(N=1000) 대비 백분위: 상위 {100 - overall['random_walk_percentile']:.1f}% (percentile={overall['random_walk_percentile']:.1f})")
    lines.append(
        f"- 부트스트랩 95% CI (누적수익률): [{overall['bootstrap_ci_low']:.4%}, {overall['bootstrap_ci_high']:.4%}]"
    )
    lines.append("")

    lines.append(f"## Market_Id x Size_Id 6개 셀 breakdown (Bonferroni 보정 α={BONFERRONI_ALPHA:.4f})")
    lines.append("")
    lines.append(
        "| 셀 | n | 적중률 | p-value | 유의(보정후) | 전략 누적수익률 | B&H | 랜덤워크 백분위 | 부트스트랩 95% CI |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, row in results_df.iterrows():
        lines.append(
            f"| {row['cell']} | {row['n_total']} | {row['accuracy']:.4f} | {row['p_value']:.4f} | "
            f"{'✓' if row['significant_bonferroni'] else '✗'} | {row['strategy_total_return']:.4%} | "
            f"{row['buy_and_hold_return']:.4%} | {row['random_walk_percentile']:.1f} | "
            f"[{row['bootstrap_ci_low']:.4%}, {row['bootstrap_ci_high']:.4%}] |"
        )
    lines.append("")

    lines.append("## 해석 시 유의사항")
    lines.append("")
    lines.append(
        "- CLAUDE.md 모델/평가 원칙: 방향성 적중률 50~55% 수준이 현실적 한계 — 이를 크게 넘는 셀은 "
        "과최적화/누수 의심 신호로 취급하고 leakage-guard 재점검이 필요함."
    )
    lines.append(
        "- 6개 셀을 동시에 검정하므로 다중비교 문제가 생겨 Bonferroni 보정을 적용했음. "
        "보정 전 p<0.05였다가 보정 후 유의하지 않게 되는 셀이 있을 수 있음."
    )
    lines.append(
        "- 전략 누적수익률은 '그 날 신호가 있는 종목들의 동일가중 포트폴리오' 가정이며, "
        "실제 자본배분/포지션사이징을 반영한 수치가 아니다."
    )
    lines.append("")

    report_path = reports_dir / f"backtest_report_v{model_version}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


if __name__ == "__main__":
    run()
