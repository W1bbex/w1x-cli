import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Any

from .prompts_text import (
    COMMAND_HINTS as COMMAND_HINTS,
)
from .prompts_text import (
    DEFAULT_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
)
from .prompts_text import (
    FORMAT_INSTRUCTIONS as FORMAT_INSTRUCTIONS,
)
from .prompts_text import (
    TOOL_INSTRUCTIONS as TOOL_INSTRUCTIONS,
)
from .types import UserConfig

logger = logging.getLogger("w1x")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

APP_VERSION = "3.5.1"
EXIT_COMMANDS = {"q", "/q", "quit", "/quit", "exit", "/exit"}
SPINNER = ["|", "/", "-", "\\"]
SPINNER_DELAY = 0.16

DATA_DIR = Path.home() / ".w1x"
CHAT_DIR = DATA_DIR / "chats"
CURRENT_CHAT_PATH = DATA_DIR / "current-chat.json"
FACTS_PATH = DATA_DIR / "memory-facts.json"
CONFIG_FILE_PATH = DATA_DIR / "config.json"

def ensure_data_dirs() -> None:
    """Создать пользовательские директории данных при необходимости."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CHAT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

DEFAULT_CONFIG: UserConfig = {
    "default_model": "qwen3.5:4b",
    "keep_alive_minutes": 30,
    "context_tokens": 8192,
    "stats_enabled": False,
}


def is_first_run() -> bool:
    """Проверяет, настроен ли W1x (создан ли файл config.json)."""
    return not CONFIG_FILE_PATH.exists()


def load_user_config() -> UserConfig:
    if not CONFIG_FILE_PATH.exists():
        return dict(DEFAULT_CONFIG)  # type: ignore[return-value]
    try:
        data = json.loads(CONFIG_FILE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            cfg: dict[str, Any] = dict(DEFAULT_CONFIG)
            cfg.update(data)
            return cfg  # type: ignore[return-value]
        logger.warning("Corrupted config file %s (expected JSON object); using defaults", CONFIG_FILE_PATH)
    except Exception as exc:
        logger.warning("Failed to load config file %s: %s; using defaults", CONFIG_FILE_PATH, exc)
    return dict(DEFAULT_CONFIG)  # type: ignore[return-value]


def save_user_config(cfg: UserConfig | dict[str, Any]) -> bool:
    tmp_path = CONFIG_FILE_PATH.with_suffix(".tmp")
    try:
        CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(CONFIG_FILE_PATH)
        return True
    except Exception as exc:
        logger.warning("Failed to save config file %s: %s", CONFIG_FILE_PATH, exc)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return False


def get_keep_alive() -> str:
    try:
        mins = int(load_user_config().get("keep_alive_minutes", 30))
    except Exception:
        mins = 30
    if mins <= 0:
        return "0"
    return f"{mins}m"


def get_context_tokens() -> int:
    val = load_user_config().get("context_tokens", 8192)
    if isinstance(val, bool):
        return 8192
    try:
        tokens = int(val)
        return max(512, tokens)
    except Exception:
        return 8192


def get_max_history_chars() -> int:
    return int(get_context_tokens() * 3.5)


MAX_SAVED_MESSAGES = 100
MAX_TOOL_ROUNDS = 3
MAX_RESPONSE_TOKENS = 2048
AGY_TIMEOUT_SECONDS = 240
AGY_MAX_PROMPT_CHARS = 5500
AGY_MAX_OUTPUT_CHARS = 30000

ACTIVE_FILE_ROOT: Path | None = None
ACTIVE_AGY_PROCESS: subprocess.Popen[str] | None = None
ACTIVE_AGY_LOCK = threading.Lock()

ACTIVE_SESSION_DATA: dict[str, Any] = {
    "messages": [],
    "system_prompt": "",
}

