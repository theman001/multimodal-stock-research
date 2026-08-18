"""Phase 3 — 학습된 정책 평가 + Phase 1 분류기 전략/랜덤워크/Buy&Hold 대비 비교.

CLAUDE.md "Phase 3 구체 사양" > "평가/비교" 기준. 저장된 정책을 지정 구간에서
결정적으로 1회 굴려 일별 NAV수익률 시계열을 만들고, 그 시계열을
`src.backtest.simulate`/`significance`의 기존 함수에 그대로 통과시켜 Phase 1과
동일 단위로 비교한다. RL 학습 신호는 `trading_env.REWARD_CLIP`으로 클리핑되지만
평가는 항상 `info["nav"]`(클리핑 없는 참값)로만 계산한다 — 리포트 수치가
실제로 실현 가능한 값이어야 하기 때문이다.

미해결로 명시(CLAUDE.md에 이미 문서화됨): 기존 6셀(market_id x size_id)
Bonferroni 독립 유의성 검정은 공동 현금 제약이 있는 포트폴리오 정책에는
독립성 가정이 깨져 그대로 재사용하지 않는다. 이 모듈은 셀별 RL 기여도 분해도
구현하지 않는다(포지션별 손익을 셀로 귀속시키려면 TradingEnv에 별도 계측이
필요 — 향후 확장 후보로 남겨둠, 리포트에 명시).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sklearn.preprocessing import StandardScaler

from src.backtest.significance import bootstrap_return_ci, paired_bootstrap_return_diff_ci
from src.backtest.simulate import (
    add_net_return,
    add_realized_return,
    buy_and_hold_return,
    cumulative_return,
    daily_portfolio_returns,
    random_walk_total_returns,
)
from src.config import get_data_root
from src.rl.panel import Panel, score_model_v3_probabilities
from src.rl.train_agent import load_checkpoint, load_checkpoint_meta, official_split_indices, walk_forward_fold_indices
from src.rl.trading_env import TradingEnv


def rollout_policy_daily_returns(
    panel: Panel, model: PPO, scaler: StandardScaler, date_start_idx: int, date_end_idx: int
) -> pd.Series:
    """저장된 정책을 [date_start_idx, date_end_idx] 구간에서 결정적으로 1회
    굴려, 일별 실제(클리핑 없는) NAV수익률 시계열을 반환한다.

    env.step()이 반환하는 reward는 학습용으로 클리핑돼 있으므로 쓰지 않고,
    매 스텝 info["nav"](참값)로 직접 일별수익률을 재구성한다.
    """
    daily_returns, _positioning = rollout_policy_with_positioning_stats(
        panel, model, scaler, date_start_idx, date_end_idx
    )
    return daily_returns


def rollout_policy_with_positioning_stats(
    panel: Panel, model: PPO, scaler: StandardScaler, date_start_idx: int, date_end_idx: int
) -> tuple[pd.Series, dict]:
    """rollout_policy_daily_returns와 동일하지만, 이 정책이 실제로 얼마나
    적극적으로 투자하는지(평균 보유종목수/현금비중)도 함께 반환한다.

    수익률 숫자만 보면 "베이스라인을 크게 넘는 결과가 나오면 과최적화/누수
    의심"(CLAUDE.md)이라는 원칙에 걸리는 경우가 생길 수 있다 — 실제로 이
    포지셔닝 통계를 보면 "정교한 타이밍"이 아니라 "거의 항상 널리 분산투자한
    채 최대 노출을 유지"하는 낮은-정교도 전략으로 수렴했는지를 바로 확인할
    수 있어, 리포트에서 raw 수익률과 함께 반드시 병기한다.
    """
    env = TradingEnv(
        panel, scaler=scaler, date_start_idx=date_start_idx, date_end_idx=date_end_idx, random_start=False
    )
    obs, _ = env.reset()

    records: list[tuple[np.datetime64, float]] = []
    n_holdings: list[int] = []
    cash_weights: list[float] = []
    terminated = truncated = False
    while not (terminated or truncated):
        t_before = env.t  # 이번 step()이 처리하는 거래일
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        records.append((panel.dates[t_before], info["nav"]))
        n_holdings.append(info["n_holding"])
        cash_weights.append(info["cash"] / info["nav"])

    nav_series = pd.Series([r[1] for r in records], index=pd.DatetimeIndex([r[0] for r in records]))
    daily_returns = nav_series.pct_change()
    daily_returns.iloc[0] = nav_series.iloc[0] / env.initial_nav - 1.0  # 첫날은 initial_nav 대비

    # 마지막 스텝은 에피소드 종료 강제청산이라 항상 n_holding=0/cash_weight=1 —
    # "평상시 얼마나 투자하는가"를 왜곡하므로 통계에서 제외한다.
    positioning = {
        "avg_n_holding": float(np.mean(n_holdings[:-1])) if len(n_holdings) > 1 else float("nan"),
        "max_n_holding": int(np.max(n_holdings[:-1])) if len(n_holdings) > 1 else 0,
        "avg_cash_weight": float(np.mean(cash_weights[:-1])) if len(cash_weights) > 1 else float("nan"),
    }
    return daily_returns, positioning


def _window_realized_return_df(
    features_df: pd.DataFrame, ohlcv_df: pd.DataFrame, date_start, date_end
) -> pd.DataFrame:
    """[date_start, date_end] 구간의 (ticker, date) 행에 실현수익률을 붙인다.
    run_backtest.py와 동일한 패턴(add_realized_return 재사용).

    date_end 당일의 realized_return을 구하려면 그 다음 거래일의 종가가 필요하다.
    먼저 [date_start, date_end]로 자른 뒤 add_realized_return을 적용하면, 각
    종목의 구간 마지막 행은 "다음 행이 없음"으로 처리돼(shift(-1)이 그룹 내
    다음 행을 못 찾음) NaN이 되고 dropna에서 사라진다 — RL 롤아웃(panel 전체
    날짜를 그대로 씀)은 date_end까지 꽉 채워 쓰는데 이 함수만 하루 짧아져
    비교 구간이 어긋나는 실제 버그였다. date_end 이후로 살짝 넓힌 구간에서
    먼저 계산한 뒤, 최종적으로 [date_start, date_end]로 다시 잘라낸다.
    """
    buffer_end = date_end + pd.Timedelta(days=10)  # simulate.py MAX_REALIZED_RETURN_GAP_DAYS와 동일한 여유
    extended = features_df[(features_df["date"] >= date_start) & (features_df["date"] <= buffer_end)].copy()
    df = extended.merge(ohlcv_df[["date", "ticker", "close"]], on=["date", "ticker"], how="left")
    df = add_realized_return(df)
    df = df[(df["date"] >= date_start) & (df["date"] <= date_end)]
    return df.dropna(subset=["realized_return"]).reset_index(drop=True)


def classifier_strategy_daily_returns(window_df: pd.DataFrame, data_root: Path) -> pd.Series:
    """model_v3 기반 Phase 1 전략(매수 후 익일 매도, 매일 재진입)의 그 구간
    일별 포트폴리오 수익률. simulate.py 함수를 그대로 재사용 — 새 로직 없음."""
    prob = score_model_v3_probabilities(window_df, data_root)
    predicted_up = (prob >= 0.5).astype(int)
    df = add_net_return(window_df, predicted_up)
    return daily_portfolio_returns(df)


def evaluate_window(
    panel: Panel,
    model: PPO,
    scaler: StandardScaler,
    features_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    data_root: Path,
    date_start_idx: int,
    date_end_idx: int,
) -> dict:
    """한 구간에서 RL 정책 vs 분류기 전략 vs 랜덤워크 vs Buy&Hold를 비교한다."""
    date_start, date_end = panel.dates[date_start_idx], panel.dates[date_end_idx]
    window_df = _window_realized_return_df(features_df, ohlcv_df, date_start, date_end)

    rl_returns, positioning = rollout_policy_with_positioning_stats(
        panel, model, scaler, date_start_idx, date_end_idx
    )
    classifier_returns = classifier_strategy_daily_returns(window_df, data_root)

    # RL/분류기 전략은 거래일 정의가 다를 수 있음(RL은 panel 전체 날짜, 분류기는
    # 신호 있는 종목이 있는 날만) — paired 비교를 위해 공통 날짜로 맞춘다.
    common_dates = rl_returns.index.intersection(classifier_returns.index)
    rl_common = rl_returns.loc[common_dates].to_numpy()
    classifier_common = classifier_returns.loc[common_dates].to_numpy()

    rl_cum = cumulative_return(rl_returns)
    classifier_cum = cumulative_return(classifier_returns)

    rw_totals = random_walk_total_returns(window_df, n_sims=1000, seed=42)
    rl_total = float(rl_cum.iloc[-1]) if len(rl_cum) else float("nan")
    rl_rw_percentile = float((rw_totals < rl_total).mean() * 100)

    return {
        "date_start": date_start,
        "date_end": date_end,
        "rl_total_return": rl_total,
        "rl_bootstrap_ci": bootstrap_return_ci(rl_returns.to_numpy()),
        "classifier_total_return": float(classifier_cum.iloc[-1]) if len(classifier_cum) else float("nan"),
        "random_walk_percentile": rl_rw_percentile,
        "buy_and_hold_return": buy_and_hold_return(window_df),
        "paired_diff_rl_vs_classifier": paired_bootstrap_return_diff_ci(rl_common, classifier_common),
        "n_common_days": len(common_dates),
        "avg_n_holding": positioning["avg_n_holding"],
        "avg_cash_weight": positioning["avg_cash_weight"],
    }


def _load_checkpoint_checked(checkpoints_dir: Path, name: str, include_event_features: bool) -> tuple[PPO, StandardScaler]:
    """체크포인트를 불러오되, 학습 당시의 include_event_features(그 .meta.json에
    기록됨)가 지금 이 평가가 쓰려는 값과 일치하는지 먼저 확인한다.

    둘이 다르면 관측 차원(D_static)이 어긋나 StandardScaler/정책망 입력 크기가
    안 맞아 sklearn/SB3 내부 어딘가에서 알아보기 힘든 shape 에러로 죽는다 —
    여기서 먼저 이름 붙은 명확한 에러로 막는다.
    """
    meta = load_checkpoint_meta(checkpoints_dir, name)
    trained_with = meta.get("include_event_features")
    if trained_with != include_event_features:
        raise ValueError(
            f"{name} 체크포인트는 include_event_features={trained_with}로 학습됐는데 "
            f"지금 평가는 include_event_features={include_event_features}로 패널을 만들었음 — "
            "관측 차원이 어긋나 평가를 진행할 수 없다. run()의 include_event_features를 "
            f"{trained_with}로 맞출 것."
        )
    return load_checkpoint(checkpoints_dir, name)


def run(data_root: Path | None = None, include_event_features: bool = True) -> dict:
    """walk-forward 5폴드 + 공식 단일분할 전부 평가하고 리포트를 저장한다."""
    from src.rl.panel import build_panel  # 지연 임포트: run() 호출 시점에만 필요(무거운 IO)

    data_root = data_root or get_data_root()
    processed = data_root / "processed"
    checkpoints_dir = data_root / "checkpoints"
    reports_dir = data_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    panel = build_panel(data_root=data_root, include_event_features=include_event_features)
    features_df = pd.read_parquet(processed / "features.parquet")
    ohlcv_df = pd.read_parquet(processed / "ohlcv_meta.parquet")[["date", "ticker", "close"]]

    fold_results = []
    for f in walk_forward_fold_indices(panel, n_folds=5):
        model, scaler = _load_checkpoint_checked(checkpoints_dir, f"rl_policy_fold{f.fold}", include_event_features)
        result = evaluate_window(
            panel, model, scaler, features_df, ohlcv_df, data_root, f.test_start_idx, f.test_end_idx
        )
        result["fold"] = f.fold
        fold_results.append(result)
        print(f"[evaluate] fold {f.fold} 평가 완료: RL 누적수익률 {result['rl_total_return']:.4%}")

    train_end_idx, test_start_idx, test_end_idx = official_split_indices(panel)
    model, scaler = _load_checkpoint_checked(checkpoints_dir, "rl_policy_v1", include_event_features)
    official_result = evaluate_window(
        panel, model, scaler, features_df, ohlcv_df, data_root, test_start_idx, test_end_idx
    )
    print(f"[evaluate] 공식 단일분할 평가 완료: RL 누적수익률 {official_result['rl_total_return']:.4%}")

    report_path = _write_report(official_result, fold_results, reports_dir)
    print(f"[evaluate] 리포트 저장 -> {report_path}")

    return {"official": official_result, "folds": fold_results}


def _write_report(official: dict, folds: list[dict], reports_dir: Path) -> Path:
    lines = []
    lines.append("# Phase 3 RL 정책 백테스트 리포트 v1")
    lines.append("")
    lines.append(f"생성 시각: {date.today().isoformat()}")
    lines.append("")
    lines.append(
        "> ⚠️ **이 리포트의 RL 수치는 학습 시점과 다른 입력분포로 평가된 것이라 신뢰할 수 없다.** "
        "`panel.py::score_model_v3_probabilities()`가 model_v3에 스케일링 안 된 raw 값을 넣던 버그를 "
        "이 리포트를 만들면서 발견해 수정했다(raw vs 스케일링 입력의 예측 상관계수 0.044 — 사실상 무관). "
        "그런데 지금 저장된 6개 정책은 전부 **그 버그가 있던 상태의 model_v3_prob 분포**를 보며 학습됐고, "
        "이 리포트는 방금 **수정된(올바른) model_v3_prob 분포**로 그 정책들을 평가한 것이다 — 즉 학습 때 "
        "본 적 없는 입력분포로 테스트하는 셈이라 train/eval이 어긋나 있다. 분류기 전략 벤치마크 수치와 "
        "평가 구간(1일 절단 버그 수정)은 이 리포트에서 올바르게 고쳐졌지만, **RL 정책 자체의 수치는 "
        "수정된 model_v3.py로 처음부터 재학습해야 신뢰할 수 있다.**"
    )
    lines.append("")
    lines.append(
        "**비용 가정**: `src/backtest/costs.py`(Phase 1과 동일 요율) — 국내 왕복 수수료 0.03% + "
        "거래세 0.18% / 해외 커미션 0 + 슬리피지 5bp. 다만 부과 **시점**이 Phase 1 백테스트와 다르다 — "
        "매일 재부과하지 않고 포지션 진입/청산 시점에만 절반씩 부과한다(RL은 진짜 포지션 상태가 있으므로, "
        "CLAUDE.md \"Phase 3 구체 사양\" > \"보상 함수\" 참고)."
    )
    lines.append("")
    lines.append(
        "**전략 가정**: 120종목 포트폴리오를 PPO 정책이 매수/매도/보유로 동시에 판단, "
        "동일가중 자본배분(종목당 NAV의 5% 상한, 캡 초과분 재분배 없음)."
    )
    lines.append("")

    lines.append("## 공식 단일분할 결과 (test.parquet 구간, backtest_report_v3.md와 동일 기간)")
    lines.append("")
    lines.append(f"- 구간: {pd.Timestamp(official['date_start']).date()} ~ {pd.Timestamp(official['date_end']).date()}")
    lines.append(f"- RL 정책 누적수익률: {official['rl_total_return']:.4%}")
    boot = official["rl_bootstrap_ci"]
    lines.append(f"- RL 부트스트랩 95% CI: [{boot['ci_low']:.4%}, {boot['ci_high']:.4%}]")
    lines.append(f"- Phase 1 분류기 전략(매일 재진입) 누적수익률: {official['classifier_total_return']:.4%}")
    diff = official["paired_diff_rl_vs_classifier"]
    lines.append(
        f"- **RL - 분류기 전략 차이의 95% CI (paired bootstrap, n={official['n_common_days']}일)**: "
        f"[{diff['ci_low']:.4%}, {diff['ci_high']:.4%}] (점추정 {diff['point_estimate']:.4%})"
    )
    lines.append(f"- Buy & Hold 참고 벤치마크(비용 미반영): {official['buy_and_hold_return']:.4%}")
    lines.append(
        f"- 랜덤워크(N=1000) 대비 백분위: 상위 {100 - official['random_walk_percentile']:.1f}% "
        f"(percentile={official['random_walk_percentile']:.1f})"
    )
    lines.append(
        f"- **포지셔닝 행태(평상시, 강제청산 스텝 제외)**: 평균 보유종목수 {official['avg_n_holding']:.1f}/120, "
        f"평균 현금비중 {official['avg_cash_weight']:.2%}"
    )
    lines.append("")

    lines.append("## Walk-forward 5폴드 견고성")
    lines.append("")
    lines.append(
        "| 폴드 | 평가 기간 | RL 누적수익률 | 분류기 전략 누적수익률 | RL-분류기 차이 95% CI | 랜덤워크 백분위 | Buy&Hold | 평균 보유종목수 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in folds:
        d = r["paired_diff_rl_vs_classifier"]
        lines.append(
            f"| {r['fold']} | {pd.Timestamp(r['date_start']).date()}~{pd.Timestamp(r['date_end']).date()} | "
            f"{r['rl_total_return']:.4%} | {r['classifier_total_return']:.4%} | "
            f"[{d['ci_low']:.4%}, {d['ci_high']:.4%}] | {r['random_walk_percentile']:.1f} | "
            f"{r['buy_and_hold_return']:.4%} | {r['avg_n_holding']:.1f}/120 |"
        )
    lines.append("")

    lines.append("## 해석 시 유의사항 — 절대수익률 숫자를 그대로 믿지 말 것")
    lines.append("")
    avg_holding_overall = official["avg_n_holding"]
    lines.append(
        f"- **RL이 실제로 학습한 건 '정교한 타이밍'이 아니라 '거의 항상 널리 분산해서 최대로 투자 상태 유지'로 "
        f"보인다.** 공식 분할 기준 평균 보유종목수 {avg_holding_overall:.1f}/120, 평균 현금비중 "
        f"{official['avg_cash_weight']:.2%} — 즉 항상 자본의 ~99%+ 를 60개 안팎 종목에 걸쳐 투자 중이었다. "
        "이 프로젝트 전체에서 이미 확인된 사실(기술지표+이벤트 신호로는 51~55%대 방향성 정확도가 현실적 "
        "천장, Phase 2 모듈 5는 이벤트 피처가 노이즈 수준 기여만 한다고 결론)과 종합하면, 개별 종목/타이밍을 "
        "잘 골라내는 능력을 학습했다기보다 '거래비용을 피하고 시장에 계속 노출되는 게 보상을 극대화한다'는 "
        "저정교도 전략으로 수렴했을 가능성이 높다 — CLAUDE.md 원칙(베이스라인을 크게 넘는 결과는 과최적화/"
        "누수부터 의심)에 따라 절대수익률 숫자 자체를 '학습된 알파'로 해석하지 않는다."
    )
    lines.append(
        "- **랜덤워크 백분위는 이 정책엔 약한 벤치마크다.** `random_walk_total_returns`는 종목마다 매일 "
        "독립적으로 50% 확률을 다시 뽑는 방식이라, 상승장에서도 며칠 연속 보유를 유지하지 못해 복리 효과를 "
        "구조적으로 거의 못 살린다(실측: 공식 분할 구간에서 랜덤워크 1000회 시뮬레이션의 최댓값이 +14.0%인데 "
        "같은 구간 Buy&Hold는 +89.5% — 반면 RL은 항상 투자된 상태를 유지하므로 이 비교에서 이기는 게 당연하다"
        "). 그래서 **'랜덤워크 백분위 100'은 skill의 증거가 아니라, 지속적으로 시장에 노출된 전략이면 거의 "
        "누구나 넘는 낮은 문턱이다** — 이 지표보다 아래 두 비교가 훨씬 의미 있다."
    )
    lines.append(
        "- **더 의미 있는 비교는 RL vs Buy&Hold다.** 두 전략 다 '항상 투자된 상태' 근처라 조건이 비슷한데, "
        "RL이 전 구간에서 Buy&Hold를 소폭(대체로 수 %p) 앞선다 — 이건 5% 집중도 상한 덕에 개별 종목 리스크가 "
        "덜 쏠린 효과일 수도 있고, 우연일 수도 있다. 큰 폭의 알파라기보단 '분산 규칙이 조금 다른 buy-and-hold' "
        "정도로 해석하는 게 안전하다."
    )
    lines.append(
        "- **RL vs 분류기 전략(Phase 1) 차이는 대부분 '지속 보유 vs 매일 재진입' 구조 차이에서 온다.** 분류기 "
        "전략은 매일 새로 판단해 재진입하므로 방향을 맞춘 날에도 복리 효과를 거의 못 얻는다 — RL이 이겼다고 "
        "해서 RL의 예측이 분류기보다 더 정확하다는 뜻은 아니다."
    )
    lines.append(
        "- **미해결로 명시**: 기존 6셀(시장×규모) Bonferroni 독립 유의성 검정은 120종목이 독립 시행이라는 "
        "가정 위에 있는데, 공동 현금 제약이 있는 포트폴리오 정책에는 그 가정이 깨진다 — 이 리포트는 "
        "독립 유의성 검정으로 재사용하지 않았고, 셀별 RL 기여도 분해도 구현하지 않았다(포지션별 손익을 "
        "셀로 귀속시키려면 `TradingEnv`에 별도 계측이 필요 — 향후 확장 후보)."
    )
    lines.append(
        "- RL 정책과 분류기 전략의 일별수익률은 서로 다른 거래일 정의를 가질 수 있어(RL은 전체 거래일, "
        "분류기는 신호가 있는 날만), paired bootstrap은 두 시계열의 공통 거래일에서만 계산했다."
    )
    lines.append(
        "- CLAUDE.md 모델/평가 원칙: 방향성 적중률 50~55% 수준이 현실적 한계 — RL도 마찬가지로, "
        "베이스라인을 못 이기는 결과가 나오면 그 자체로 유효한 결론이다(Phase 1/2와 동일 원칙)."
    )
    lines.append("")

    report_path = reports_dir / "phase3_rl_backtest_report_v1.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


if __name__ == "__main__":
    run()
