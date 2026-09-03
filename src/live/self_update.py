"""06:00 KST cron — origin/main 최신 코드로 동기화하고 결과를 Mattermost 로 알린다.

OMV(GUI 전용) 배포에서 터미널로 `git pull` 을 칠 수 없으므로, 컨테이너가
스스로 매일 코드를 당겨온다(deploy/docker-compose.yml). `/app` 의 tracked
파일은 파이프라인이 절대 안 건드리므로(모든 쓰기는 `DATA_ROOT` 볼륨)
`reset --hard` 가 항상 안전하다.

알림 규칙:
- 실패(git/pip 오류) → error 알림
- 실제로 새 커밋을 당겨옴 → info 알림 (before/after SHA)
- 변경 없음 → 무음 (매일 오는 노이즈 방지)
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

from src.live.notify import send_notification


def _run(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True, cwd=None).stdout.strip()


def run() -> None:
    # git 이 중간에 죽어 남긴 stale lock 을 먼저 치운다 — 단일 컨테이너라
    # 동시 git 실행이 없으므로 남아 있는 lock 은 항상 stale 이다(컨테이너
    # recreate 가 git 을 끊었을 때 발생, 2026-09-03 실제로 shallow.lock 로 막힘).
    for lock in glob.glob(".git/*.lock"):
        try:
            os.remove(lock)
        except OSError:
            pass
    try:
        before = _run("git", "rev-parse", "HEAD")
        _run("git", "fetch", "-q", "--depth", "1", "origin", "main")
        _run("git", "reset", "-q", "--hard", "FETCH_HEAD")
        after = _run("git", "rev-parse", "HEAD")
    except subprocess.CalledProcessError as e:
        send_notification(
            f"[stockbot] 코드 자동동기화 실패(git): {(e.stderr or e.stdout or '').strip()[:300]}", level="error"
        )
        raise

    if before == after:
        return  # 변경 없음 — 무음

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "-r", "requirements.txt"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        send_notification(
            f"[stockbot] 코드는 {before[:8]}->{after[:8]} 로 갱신됐으나 pip install 실패: "
            f"{(e.stderr or e.stdout or '').strip()[:300]}",
            level="error",
        )
        raise

    send_notification(f"[stockbot] 코드 업데이트 {before[:8]} -> {after[:8]}", level="info")


if __name__ == "__main__":
    run()
