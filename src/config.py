"""DATA_ROOT resolution shared by all src/ modules.

Precedence: DATA_ROOT env var > Colab Drive mount > local ./data.
See CLAUDE.md "경로 규칙 (DATA_ROOT)".
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/주식프로젝트/multimodal-stock-research")


def get_data_root() -> Path:
    env_value = os.environ.get("DATA_ROOT")
    if env_value:
        return Path(env_value)
    if Path("/content/drive").exists():
        return COLAB_DRIVE_ROOT
    return REPO_ROOT / "data"
