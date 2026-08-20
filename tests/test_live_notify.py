"""notify.py 유닛테스트 — 실제 Mattermost webhook 호출 없이 requests.post를 모킹한다."""
import requests

from src.live.notify import send_notification


class _FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def test_send_notification_returns_false_when_webhook_url_not_configured(monkeypatch):
    monkeypatch.delenv("MATTERMOST_WEBHOOK_URL", raising=False)
    called = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: called.append((a, k)) or _FakeResponse())

    result = send_notification("테스트 메시지")

    assert result is False
    assert called == []  # webhook URL이 없으면 네트워크 호출 자체를 시도하면 안 됨


def test_send_notification_posts_to_webhook_url_with_text_payload(monkeypatch):
    monkeypatch.setenv("MATTERMOST_WEBHOOK_URL", "https://mattermost.example.com/hooks/abc123")
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "post", fake_post)

    result = send_notification("결정 완료: 매수 3건")

    assert result is True
    assert captured["url"] == "https://mattermost.example.com/hooks/abc123"
    assert "결정 완료: 매수 3건" in captured["json"]["text"]
    assert captured["timeout"] is not None


def test_send_notification_prefixes_message_by_level(monkeypatch):
    monkeypatch.setenv("MATTERMOST_WEBHOOK_URL", "https://mattermost.example.com/hooks/abc123")
    captured = {}
    monkeypatch.setattr(
        requests, "post", lambda url, json=None, timeout=None: captured.update(json=json) or _FakeResponse(200)
    )

    send_notification("킬스위치 발동", level="warning")
    warning_text = captured["json"]["text"]

    captured.clear()
    send_notification("정상 완료", level="info")
    info_text = captured["json"]["text"]

    assert warning_text != info_text  # 레벨별로 프리픽스가 달라야 구분 가능


def test_send_notification_returns_false_on_http_error_without_raising(monkeypatch):
    monkeypatch.setenv("MATTERMOST_WEBHOOK_URL", "https://mattermost.example.com/hooks/abc123")
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(500))

    result = send_notification("실패해야 하는 메시지")

    assert result is False  # 예외를 던지면 안 됨 — 호출자(decide/execute)를 막으면 안 되므로


def test_send_notification_returns_false_on_network_exception_without_raising(monkeypatch):
    monkeypatch.setenv("MATTERMOST_WEBHOOK_URL", "https://mattermost.example.com/hooks/abc123")

    def raise_connection_error(*args, **kwargs):
        raise requests.ConnectionError("network unreachable")

    monkeypatch.setattr(requests, "post", raise_connection_error)

    result = send_notification("네트워크 끊긴 상황")

    assert result is False
