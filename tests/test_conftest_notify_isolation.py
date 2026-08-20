"""conftest.py::_no_real_mattermost_webhook_by_default 회귀 테스트.

decide_us.py/execute_us.py의 여러 예외 경로 테스트(재구성 불일치, 락 경합
등)는 notify.py::send_notification을 monkeypatch하지 않고 그대로 통과시킨다
— 그 함수가 MATTERMOST_WEBHOOK_URL이 설정된 경우 실제로 requests.post()를
호출하기 때문에, 개발자 셸/CI가 그 값을 이미(이 테스트 스위트와 무관한
이유로) 물려주고 있으면 테스트 실행 중 실제 Mattermost 채널로 알림이
새나갈 수 있었다(review-loop 5단계 1차 재검토로 발견). 이 파일은 "pytest
시작 전부터 이미 셸에 설정돼 있던 상황"을 세션 스코프 fixture로 흉내내
conftest.py의 함수 스코프 autouse fixture가 매 테스트 시작 시 그 값을
강제로 지우는지 검증한다."""
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _simulate_shell_exported_webhook_url():
    """pytest 프로세스가 시작되기 전부터 셸에 export돼 있던 환경변수를
    흉내낸다 — monkeypatch가 아니라 os.environ을 직접 건드려서 세션 전체에
    걸쳐 남아있게 한다(정리하지 않음, 실제 셸 export와 동일한 지속성)."""
    os.environ["MATTERMOST_WEBHOOK_URL"] = "https://mattermost.example.com/hooks/leaked-from-shell"
    yield
    os.environ.pop("MATTERMOST_WEBHOOK_URL", None)


def test_webhook_url_absent_despite_session_level_shell_leak_1():
    assert os.environ.get("MATTERMOST_WEBHOOK_URL") is None


def test_webhook_url_absent_despite_session_level_shell_leak_2():
    """두 번째 테스트에서도 여전히 지워져 있어야 한다 — 함수 스코프 autouse
    fixture가 매 테스트마다 다시 지우는지(한 번만 지우고 마는 게 아닌지) 확인."""
    assert os.environ.get("MATTERMOST_WEBHOOK_URL") is None
