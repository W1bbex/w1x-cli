"""Хранение и восстановление истории диалога W1x."""
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from . import config
from .types import ChatMessage


def new_context(system_prompt: str) -> list[ChatMessage]:
    """Создать чистый контекст диалога с системной инструкцией."""
    return [{"role": "system", "content": system_prompt}]


def save_chat(messages: Sequence[ChatMessage], requested_name: str = "") -> Path:
    """Экспортировать историю диалога в Markdown-файл."""
    config.CHAT_DIR.mkdir(parents=True, exist_ok=True)
    if requested_name:
        safe_name = Path(requested_name).name.strip()
        if safe_name.strip(" .") == "":
            safe_name = datetime.now().strftime("chat-%Y-%m-%d-%H%M%S.md")
        elif not safe_name.lower().endswith(".md"):
            safe_name += ".md"
    else:
        safe_name = datetime.now().strftime("chat-%Y-%m-%d-%H%M%S.md")
    path = config.CHAT_DIR / safe_name

    lines = ["# Ollama chat", ""]
    for message in messages:
        if message["role"] == "system":
            continue
        title = "Вы" if message["role"] == "user" else "Ассистент"
        lines.extend([f"## {title}", "", message["content"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_current_chat(system_prompt: str, messages: Sequence[ChatMessage]) -> bool:
    """Атомарно сохранить текущий диалог на диск."""
    saved_messages = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in messages
        if message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
    ][-config.MAX_SAVED_MESSAGES:]

    state = {
        "version": 1,
        "system_prompt": system_prompt,
        "custom_system_prompt": system_prompt != config.DEFAULT_SYSTEM_PROMPT,
        "messages": saved_messages,
    }

    target_path = config.CURRENT_CHAT_PATH
    temporary_path = target_path.with_suffix(".tmp")
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(target_path)
        return True
    except OSError as error:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass
        config.logger.warning("Failed to save chat history %s: %s", target_path, error)
        return False


def load_current_chat() -> tuple[str, list[ChatMessage]]:
    """Восстановить сохранённый диалог или создать новый контекст."""
    target_path = config.CURRENT_CHAT_PATH
    if not target_path.exists():
        return config.DEFAULT_SYSTEM_PROMPT, new_context(config.DEFAULT_SYSTEM_PROMPT)

    try:
        state = json.loads(target_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            return config.DEFAULT_SYSTEM_PROMPT, new_context(config.DEFAULT_SYSTEM_PROMPT)
        saved_system_prompt = state.get("system_prompt") or ""
        has_custom_prompt = state.get("custom_system_prompt") is True
        system_prompt = (
            str(saved_system_prompt)
            if has_custom_prompt and saved_system_prompt
            else config.DEFAULT_SYSTEM_PROMPT
        )
        saved_messages: list[ChatMessage] = []

        discard_legacy_style = (
            "custom_system_prompt" not in state
            and saved_system_prompt
            and saved_system_prompt != config.DEFAULT_SYSTEM_PROMPT
        )

        raw_messages = state.get("messages")
        message_list = raw_messages if isinstance(raw_messages, list) else []

        for message in ([] if discard_legacy_style else message_list):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                saved_messages.append({"role": role, "content": content})

        if discard_legacy_style:
            config.logger.info("Discarded legacy experimental system prompt without custom flag")

        return system_prompt, [
            {"role": "system", "content": system_prompt},
            *saved_messages[-config.MAX_SAVED_MESSAGES:],
        ]
    except (OSError, ValueError, TypeError, AttributeError) as error:
        config.logger.warning("Failed to restore chat history %s: %s", target_path, error)
        return config.DEFAULT_SYSTEM_PROMPT, new_context(config.DEFAULT_SYSTEM_PROMPT)
