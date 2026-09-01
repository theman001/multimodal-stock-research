"""self_update.py — 06:00 KST 코드 자동동기화 + Mattermost 알림 규칙 검증."""
import subprocess

import pytest

import src.live.self_update as su


class _Git:
    """git rev-parse 가 before -> after 순으로 다른 SHA를 뱉도록 하는 페이크."""

    def __init__(self, shas, fail_on=None, pip_fails=False):
        self._shas = list(shas)
        self._fail_on = fail_on  # 이 부분문자열이 argv에 있으면 CalledProcessError
        self._pip_fails = pip_fails

    def __call__(self, args, **kwargs):
        joined = " ".join(args)
        if self._fail_on and self._fail_on in joined:
            raise subprocess.CalledProcessError(1, args, output="", stderr="boom")
        if "pip" in joined:
            if self._pip_fails:
                raise subprocess.CalledProcessError(1, args, output="", stderr="pip boom")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if "rev-parse" in joined:
            return subprocess.CompletedProcess(args, 0, stdout=self._shas.pop(0) + "\n", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")


@pytest.fixture
def notes(monkeypatch):
    captured = []
    monkeypatch.setattr(su, "send_notification", lambda text, level="info": captured.append((text, level)))
    return captured


def test_no_change_is_silent(monkeypatch, notes):
    monkeypatch.setattr(subprocess, "run", _Git(["abc1234", "abc1234"]))
    su.run()
    assert notes == []


def test_new_commit_pulls_and_notifies(monkeypatch, notes):
    monkeypatch.setattr(subprocess, "run", _Git(["aaaaaaaa", "bbbbbbbb"]))
    su.run()
    assert len(notes) == 1
    text, level = notes[0]
    assert level == "info" and "aaaaaaaa" in text and "bbbbbbbb" in text


def test_git_failure_notifies_error_and_raises(monkeypatch, notes):
    monkeypatch.setattr(subprocess, "run", _Git(["aaaaaaaa"], fail_on="fetch"))
    with pytest.raises(subprocess.CalledProcessError):
        su.run()
    assert notes and notes[0][1] == "error" and "실패" in notes[0][0]


def test_pip_failure_after_git_notifies_error_and_raises(monkeypatch, notes):
    monkeypatch.setattr(subprocess, "run", _Git(["aaaaaaaa", "bbbbbbbb"], pip_fails=True))
    with pytest.raises(subprocess.CalledProcessError):
        su.run()
    assert notes and notes[-1][1] == "error" and "pip" in notes[-1][0]
