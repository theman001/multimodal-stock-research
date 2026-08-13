"""Atomic read/write for ${DATA_ROOT}/progress_log.json.

Schema and protocol defined in CLAUDE.md "진행 로그 프로토콜".
"""
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _default_log() -> dict:
    return {
        "phase": "1",
        "last_completed_step": None,
        "steps": [],
        "next_action": "Phase 1 데이터 수집 모듈 시작",
        "blockers": [],
    }


def load(data_root: Path) -> dict:
    path = data_root / "progress_log.json"
    if not path.exists():
        return _default_log()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(data_root: Path, log: dict) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    path = data_root / "progress_log.json"
    fd, tmp_path = tempfile.mkstemp(dir=data_root, prefix=".progress_log_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def record_step(log: dict, step: str, status: str, output: str | None = None) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    for entry in log["steps"]:
        if entry["step"] == step:
            entry.update(status=status, timestamp=timestamp, output=output)
            break
    else:
        log["steps"].append(
            {"step": step, "status": status, "timestamp": timestamp, "output": output}
        )
    if status == "done":
        log["last_completed_step"] = step
    return log
