"""Управление моделями Ollama, фоновой предзагрузкой и выгрузкой из VRAM."""
import os
import threading
from collections.abc import Sequence
from typing import Any

from ollama import generate, ps
from ollama import list as ollama_list
from rich.markup import escape as escape_markup
from rich.panel import Panel

from .config import (
    get_context_tokens,
    get_keep_alive,
    load_user_config,
    save_user_config,
)
from .terminal import console, run_with_spinner

LAST_LIST_ERROR: Exception | None = None
_LIST_LOCK = threading.Lock()
_CONSOLE_CTRL_HANDLER_REF: Any = None


def model_name(item: Any) -> str | None:
    """Извлечь строковое имя модели из структуры ответа Ollama."""
    raw = getattr(item, "model", None) or (
        item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
    )
    return str(raw) if raw else None


def installed_models(*, show_error: bool = True) -> list[str]:
    """Вернуть список установленных в Ollama моделей или [] при ошибке."""
    global LAST_LIST_ERROR
    with _LIST_LOCK:
        try:
            names = [model_name(item) for item in ollama_list().models]
            clean_names = [name for name in names if name]
            LAST_LIST_ERROR = None
            return clean_names
        except Exception as error:
            LAST_LIST_ERROR = error
            if show_error:
                err_str = str(error).lower()
                if any(w in err_str for w in ("connection", "connect", "refused", "111", "10061")):
                    console.print(
                        Panel(
                            "[bold red]❌ Не удалось подключиться к серверу Ollama.[/bold red]\n\n"
                            "• Убедитесь, что [bold green]Ollama[/bold green] запущена в Windows (значок в трее рядом с часами).\n"
                            "• Если Ollama ещё не установлена — скачайте её с официального сайта: [bold cyan]https://ollama.com[/bold cyan]",
                            title="[bold]Сервер Ollama недоступен[/bold]",
                            border_style="red",
                            padding=(1, 2),
                        )
                    )
                else:
                    console.print(f"[red]Не удалось получить список моделей: {escape_markup(str(error))}[/red]")
            return []


def get_installed_models_with_status(
    *, show_error: bool = True
) -> tuple[list[str], Exception | None]:
    """Получить список установленных в Ollama моделей и объект ошибки (если возникла)."""
    names = installed_models(show_error=show_error)
    return names, LAST_LIST_ERROR


def running_models() -> list[str]:
    """Вернуть список моделей, загруженных сейчас в память VRAM."""
    try:
        return [
            name
            for item in ps().models
            if (name := model_name(item))
        ]
    except Exception:
        return []


def unload_all_models() -> list[str]:
    """Выгрузить все активные модели из VRAM."""
    unloaded: list[str] = []
    for name in running_models():
        try:
            generate(model=name, prompt="", keep_alive=0)
            unloaded.append(name)
        except Exception:
            pass
    return unloaded


def register_console_ctrl_handler() -> None:
    """Зарегистрировать системный обработчик закрытия консоли Windows для выгрузки VRAM."""
    global _CONSOLE_CTRL_HANDLER_REF
    if os.name != "nt" or _CONSOLE_CTRL_HANDLER_REF is not None:
        return
    try:
        import ctypes
        from ctypes import wintypes

        ctypes_mod: Any = ctypes
        _PHANDLER_ROUTINE = ctypes_mod.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def _console_ctrl_handler(ctrl_type: int) -> bool:
            if ctrl_type in {2, 5, 6}:
                try:
                    unload_all_models()
                except Exception:
                    pass
            return False

        handler = _PHANDLER_ROUTINE(_console_ctrl_handler)
        if ctypes_mod.windll.kernel32.SetConsoleCtrlHandler(handler, True):
            _CONSOLE_CTRL_HANDLER_REF = handler
    except Exception:
        pass


def reset_all_models(show_message: bool = False) -> list[str]:
    """Выгрузить модели из памяти и при необходимости вывести результат пользователю."""
    global _CURRENT_PRELOAD
    with _PRELOAD_LOCK:
        if _CURRENT_PRELOAD is not None:
            _CURRENT_PRELOAD.cancelled = True
            _CURRENT_PRELOAD = None
    unloaded = unload_all_models()
    if show_message:
        if unloaded:
            console.print(f"[green]✓ Модели выгружены из VRAM:[/green] {', '.join(unloaded)}")
        else:
            console.print("[dim]Активных моделей в VRAM не найдено.[/dim]")
    return unloaded


class PreloadTask:
    """Состояние асинхронного прогрева модели в памяти."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.event = threading.Event()
        self.error: Exception | None = None
        self.thread: threading.Thread | None = None
        self.cancelled: bool = False


_CURRENT_PRELOAD: PreloadTask | None = None
_PRELOAD_LOCK = threading.Lock()


def _background_preload_worker(task: PreloadTask) -> None:
    try:
        if task.cancelled:
            return
        generate(
            model=task.name,
            prompt="",
            keep_alive=get_keep_alive(),
            options={"num_ctx": get_context_tokens()},
        )
    except Exception as e:
        if not task.cancelled:
            task.error = e
    finally:
        task.event.set()


def start_background_preload(name: str) -> None:
    """Запустить фоновый прогрев указанной модели в отдельном потоке."""
    global _CURRENT_PRELOAD
    with _PRELOAD_LOCK:
        if _CURRENT_PRELOAD is not None and _CURRENT_PRELOAD.name == name and not _CURRENT_PRELOAD.event.is_set():
            return
        if _CURRENT_PRELOAD is not None and not _CURRENT_PRELOAD.event.is_set():
            _CURRENT_PRELOAD.cancelled = True
        task = PreloadTask(name)
        worker = threading.Thread(
            target=_background_preload_worker,
            args=(task,),
            daemon=True,
            name=f"Preload-{name}",
        )
        task.thread = worker
        _CURRENT_PRELOAD = task
        worker.start()


def wait_for_model_ready(name: str | None = None) -> None:
    """Подождать готовность модели перед отправкой первого запроса."""
    global _CURRENT_PRELOAD
    with _PRELOAD_LOCK:
        task = _CURRENT_PRELOAD
    if task is None:
        return
    if name and task.name != name:
        return

    if task.event.is_set():
        if task.error:
            with _PRELOAD_LOCK:
                if _CURRENT_PRELOAD is task:
                    _CURRENT_PRELOAD = None
            raise task.error
        return

    run_with_spinner(
        f"Догружаю {name or task.name or 'модель'} в память...",
        task.event.wait,
    )
    if task.error:
        with _PRELOAD_LOCK:
            if _CURRENT_PRELOAD is task:
                _CURRENT_PRELOAD = None
        raise task.error


def _normalize_model_tag(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def get_last_model(models: Sequence[str]) -> str | None:
    """Получить уже загруженную в VRAM модель, либо модель из настроек."""
    if not models:
        return None

    try:
        active = running_models()
        if active:
            for act in active:
                for m in models:
                    if act == m:
                        return m
            for act in active:
                act_norm = _normalize_model_tag(act)
                for m in models:
                    if _normalize_model_tag(m) == act_norm:
                        return m
            for act in active:
                base = act.split(":")[0]
                for m in models:
                    if m.split(":")[0] == base:
                        return m
    except Exception:
        pass

    cfg = load_user_config()
    saved = cfg.get("default_model")
    if isinstance(saved, str) and saved in models:
        return saved

    for pref in ("qwen3.5:4b", "qwen2.5:7b", "qwen2.5:3b", "llama3.2:3b"):
        for m in models:
            if pref in m:
                return m
    return models[0]


def save_last_model(name: str) -> bool:
    """Сохранить выбранную модель в пользовательский конфиг."""
    cfg = dict(load_user_config())
    cfg["default_model"] = name
    return save_user_config(cfg)
