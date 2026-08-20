"""Phase 4 공유 — Mattermost webhook 알림.

n8n 같은 중간 홉을 거치지 않고 이미 구성된 Mattermost incoming webhook URL을
`.env`의 `MATTERMOST_WEBHOOK_URL`로 직접 호출한다(plan/10 §B-7, 핵심 결정
6번 — "n8n 경유는 부가 홉이라 정작 필요할 때 침묵 실패 지점이 될 수 있어
배제"). `requests`로 POST만 하면 되므로 신규 의존성이 필요 없다.

**알림 실패는 절대 예외를 던지지 않는다.** decide_us.py/execute_us.py 같은
호출자 입장에서, "알림 전송"은 그날의 실제 매매 판단/집행보다 부차적인
관심사다 — Mattermost 서버가 응답이 없거나 webhook URL이 아직 설정 안 됐다는
이유로(Alpaca 키와 마찬가지로 알려진 블로커) 무인 매매 사이클 전체가 죽으면
본말이 전도된다. 실패하면 stderr에 사유를 남기고 `False`를 반환할 뿐이다 —
성공 여부를 굳이 확인하고 싶은 호출자를 위해 반환값은 남겨두되, 대부분의
호출자는 반환값을 무시해도 안전하다.
"""
from __future__ import annotations

import os
import sys
from typing import Literal

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TIMEOUT_SECONDS = 10  # Mattermost가 응답 없어도 decide/execute 사이클이 오래 막히면 안 됨

_LEVEL_PREFIX = {
    "info": "ℹ️",
    "warning": "⚠️",
    "error": "🚨",
}


def send_notification(text: str, level: Literal["info", "warning", "error"] = "info") -> bool:
    """Mattermost webhook으로 `text`를 보낸다. 성공하면 True, 실패해도(webhook
    URL 미설정/네트워크 오류/Mattermost 측 에러 응답 등) 예외 없이 False를
    반환한다(모듈 docstring 참고)."""
    webhook_url = os.environ.get("MATTERMOST_WEBHOOK_URL")
    if not webhook_url:
        print(f"[notify] MATTERMOST_WEBHOOK_URL 미설정 — 알림 전송 건너뜀: {text}", file=sys.stderr)
        return False

    prefix = _LEVEL_PREFIX.get(level, "")
    payload = {"text": f"{prefix} {text}".strip()}
    try:
        resp = requests.post(webhook_url, json=payload, timeout=DEFAULT_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"[notify] Mattermost 알림 전송 실패({e!r}) — 원문: {text}", file=sys.stderr)
        return False
