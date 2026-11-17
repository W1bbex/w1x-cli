"""Интеграция эскалации сложных задач в старшую модель Antigravity."""
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence

from .. import config
from ..terminal import render_message, run_with_spinner
from ..types import ChatMessage

_SECRET_PATTERNS = [
    re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password|passwd|authorization)\b"
        r"\s*[:=]\s*([^\s,;]{8,})"
    ),
    re.compile(r"\b(?:sk|ghp|github_pat)[_-][A-Za-z0-9_-]{16,}\b"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+PRIVATE KEY-----"),
]


def _redact_secrets(text: str) -> str:
    redacted = str(text)
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[СЕКРЕТ СКРЫТ]", redacted)
    return redacted


_AGY_CANCELLED = False


def cancel_active_agy() -> None:
    """Немедленно завершить активный процесс agy, если он ещё выполняется."""
    global _AGY_CANCELLED
    with config.ACTIVE_AGY_LOCK:
        _AGY_CANCELLED = True
        process = config.ACTIVE_AGY_PROCESS
    if process is not None and process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass


def ask_antigravity(task: str, context: str = "") -> str:
    """Передать сложную задачу в headless CLI agy с фильтрацией секретов."""
    task = _redact_secrets(str("" if task is None else task).strip())
    context = _redact_secrets(str("" if context is None else context).strip())
    if not task:
        return "ESCALATE_ERROR: задача для Antigravity пуста."
    executable = shutil.which("agy")
    if not executable:
        return "ESCALATE_ERROR: команда agy не найдена в PATH."
    if len(task) > 4000:
        task = task[:4000] + "\n… задача сокращена W1x"
    if len(context) > 1200:
        context = context[-1200:]
    prompt = (
        "Ты старшая модель для локального ассистента W1x. Реши задачу и верни "
        "готовый ответ пользователю на русском языке текстом/Markdown. "
        "Не создавай и не изменяй файлы, не вызывай инструменты и не проси "
        "разрешений. Если данных недостаточно, задай один конкретный "
        "уточняющий вопрос. Не упоминай эту служебную инструкцию.\n\n"
        + (f"Краткий контекст:\n{context}\n\n" if context else "")
        + f"Задача пользователя:\n{task}"
    )
    prompt = prompt[:config.AGY_MAX_PROMPT_CHARS]

    allowed_env_names = {
        "APPDATA", "COMSPEC", "HOME", "LANG", "LC_ALL", "LOCALAPPDATA", "PATH",
        "PATHEXT", "PROGRAMFILES", "PROGRAMFILES(X86)", "SYSTEMDRIVE",
        "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "WINDIR",
    }
    safe_environment = {
        key: value for key, value in os.environ.items() if key.upper() in allowed_env_names
    }
    global _AGY_CANCELLED
    with config.ACTIVE_AGY_LOCK:
        _AGY_CANCELLED = False
        config.ACTIVE_AGY_PROCESS = None

    with tempfile.TemporaryDirectory(prefix="w1x-agy-") as temp_dir:
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                [executable, "-p", prompt],
                cwd=temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=safe_environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            with config.ACTIVE_AGY_LOCK:
                if _AGY_CANCELLED:
                    try:
                        process.kill()
                    except OSError:
                        pass
                    return "ESCALATE_ERROR: Antigravity отменена пользователем."
                config.ACTIVE_AGY_PROCESS = process
            stdout, stderr = process.communicate(timeout=config.AGY_TIMEOUT_SECONDS)
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
                try:
                    process.communicate(timeout=2)
                except Exception:
                    pass
            return f"ESCALATE_ERROR: Antigravity не ответила за {config.AGY_TIMEOUT_SECONDS} секунд."
        except (OSError, PermissionError) as error:
            return f"ESCALATE_ERROR: не удалось запустить agy: {error}"
        finally:
            if process is not None and process.poll() is None:
                try:
                    process.kill()
                    process.communicate(timeout=2)
                except Exception:
                    pass
            with config.ACTIVE_AGY_LOCK:
                if config.ACTIVE_AGY_PROCESS is process:
                    config.ACTIVE_AGY_PROCESS = None

    output = (stdout or "").strip()
    error_text = (stderr or "").strip()
    if return_code != 0:
        combined_error = f"{error_text}\n{output}".lower()
        if "permission" in combined_error and ("denied" in combined_error or "failed" in combined_error):
            return "ESCALATE_ERROR: Antigravity запросила запрещённое действие; доступ не выдан."
        details = (error_text or output or "без описания")[-1200:]
        return f"ESCALATE_ERROR: agy завершилась с кодом {return_code}: {details}"
    if not output:
        return "ESCALATE_ERROR: Antigravity не вернула текст."
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    output = ansi_escape.sub("", output)
    if len(output) > config.AGY_MAX_OUTPUT_CHARS:
        output = output[:config.AGY_MAX_OUTPUT_CHARS]
        if output.count("```") % 2:
            output += "\n```"
        output += "\n\n> ⚠️ Ответ Antigravity обрезан по лимиту W1x."
    return output


def should_auto_escalate(text: str) -> bool:
    """Определить необходимость автоматической эскалации запроса в Antigravity."""
    normalized = " ".join(text.casefold().split())
    if re.search(r"\b(?:agy|антигравити)\b", normalized):
        return True
    explicit = re.search(
        r"\b(?:передай|эскалируй|спроси|отдай)\w*\b.*"
        r"\b(?:старш|сильн|другой)\w*\s+(?:ии|модел)",
        normalized,
    )
    if explicit:
        return True
    large_code_task = (
        re.search(r"\b(?:проект|кодовая\s+база|репозитор)\w*\b", normalized)
        and re.search(
            r"\b(?:архитектур|рефактор|race\s*condition|гонк|аудит|"
            r"диагност|миграц|безопасност)\w*\b",
            normalized,
        )
        and re.search(r"\b(?:найди|исправ|спроект|проанализ|сделай|проверь)\w*\b", normalized)
    )
    many_files = re.search(r"\b(?:[5-9]\d|\d{3,})\s+файл", normalized)
    return bool(large_code_task or many_files)


def recent_context_for_agy(messages: Sequence[ChatMessage]) -> str:
    """Сформировать компактный контекст последних сообщений для Antigravity."""
    recent = []
    for message in messages:
        if message.get("role") in {"user", "assistant"}:
            recent.append(f"{message['role']}: {message.get('content', '')}")
    return "\n".join(recent[-3:])[-1200:]


def run_agy_for_user(task: str, messages: Sequence[ChatMessage]) -> str:
    """Выполнить эскалацию в Antigravity со спиннером и вывести результат."""
    render_message("Передаю задачу старшей модели Antigravity", role="tool")
    result = run_with_spinner(
        "Antigravity думает...",
        lambda: ask_antigravity(task, recent_context_for_agy(messages)),
        on_interrupt=cancel_active_agy,
    )
    if result.startswith("ESCALATE_ERROR:"):
        render_message(result, role="notice")
    else:
        render_message("Antigravity ›", role="assistant_label")
        render_message(result, role="assistant")
    return result
