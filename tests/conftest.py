import pytest


@pytest.fixture(autouse=True)
def _no_real_mattermost_webhook_by_default(monkeypatch):
    """decide_us.py/execute_us.py의 실패 경로 테스트 다수가 notify.py::send_notification을
    직접 monkeypatch하지 않고 그대로 통과시킨다(재구성 불일치/락 경합 등 예외
    자체를 검증하는 게 목적이라 알림 호출 여부는 무관심 대상이었음) — 그런데
    send_notification()은 MATTERMOST_WEBHOOK_URL이 설정돼 있으면 실제로
    requests.post()를 호출한다. 이 프로젝트의 테스트 원칙("실제 네트워크 호출
    없음", CLAUDE.md 세션 지침)을 지키려면, 개발자 셸이나 CI 환경에 그 값이
    우연히 설정돼 있어도(예: 다른 프로젝트용 공용 webhook) 테스트가 실제
    Mattermost 채널로 알림을 보내면 안 된다 — 그래서 모든 테스트에서 기본값을
    강제로 비운다(review-loop 5단계 1차 재검토로 발견). 개별 테스트(test_live_notify.py)가
    monkeypatch.setenv()로 명시적으로 값을 세팅하는 건 이 fixture 이후에
    실행되므로 그대로 유효하다."""
    monkeypatch.delenv("MATTERMOST_WEBHOOK_URL", raising=False)
