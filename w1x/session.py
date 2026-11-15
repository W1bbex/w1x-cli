"""Управление состоянием интерактивной сессии W1x и заголовками интерфейса."""
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rich.markup import escape as escape_markup
from rich.panel import Panel

from . import config
from .chat_store import new_context
from .config import APP_VERSION
from .stats import add_to_totals, empty_totals
from .terminal import console as _default_console
from .types import ChatMessage, SessionTotals, StatsRecord


def _get_console() -> Any:
    app_mod = sys.modules.get("w1x.app")
    if app_mod is not None and hasattr(app_mod, "console"):
        return app_mod.console
    return _default_console


@dataclass
class SessionState:
    """Мутабельное состояние активной интерактивной сессии W1x."""

    model: str
    system_prompt: str
    messages: list[ChatMessage]
    think_mode: bool | str = False
    stats_enabled: bool = False
    last_stats: StatsRecord | None = None
    totals: SessionTotals = field(default_factory=empty_totals)
    last_response: str = ""
    cancelled_prompt: str = ""

    def __post_init__(self) -> None:
        self.sync_active_data()

    def sync_active_data(self) -> None:
        """Синхронизировать состояние с глобальным фасадом для обратной совместимости."""
        config.ACTIVE_SESSION_DATA["session_state"] = self
        config.ACTIVE_SESSION_DATA["messages"] = self.messages
        config.ACTIVE_SESSION_DATA["system_prompt"] = self.system_prompt
        if self.cancelled_prompt:
            config.ACTIVE_SESSION_DATA["cancelled_prompt"] = self.cancelled_prompt
        else:
            config.ACTIVE_SESSION_DATA.pop("cancelled_prompt", None)

    def set_cancelled_prompt(self, prompt: str) -> None:
        self.cancelled_prompt = prompt
        config.ACTIVE_SESSION_DATA["cancelled_prompt"] = prompt

    def pop_cancelled_prompt(self) -> str:
        res = self.cancelled_prompt or str(config.ACTIVE_SESSION_DATA.pop("cancelled_prompt", "") or "")
        self.cancelled_prompt = ""
        config.ACTIVE_SESSION_DATA.pop("cancelled_prompt", None)
        return res

    def append_user_message(self, prompt: str) -> None:
        self.pop_cancelled_prompt()
        if self.messages and self.messages[-1].get("role") == "user":
            self.messages.pop()
        self.messages.append({"role": "user", "content": prompt})
        self.sync_active_data()

    def append_assistant_reply(self, reply: str, stats: StatsRecord | None) -> None:
        self.pop_cancelled_prompt()
        self.messages.append({"role": "assistant", "content": reply})
        self.last_response = reply
        self.last_stats = stats
        if stats:
            add_to_totals(self.totals, stats)
        self.sync_active_data()

    def reset_context(self) -> None:
        self.pop_cancelled_prompt()
        self.messages = new_context(self.system_prompt)
        self.last_stats = None
        self.last_response = ""
        self.totals = empty_totals()
        self.sync_active_data()

    def undo_turn(self) -> bool:
        self.pop_cancelled_prompt()
        removed = False
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages.pop()
            removed = True
        if self.messages and self.messages[-1].get("role") == "user":
            self.messages.pop()
            removed = True
        if removed:
            self.last_response = next(
                (item["content"] for item in reversed(self.messages) if item["role"] == "assistant"),
                "",
            )
            self.sync_active_data()
        return removed

    def rollback_last_turn(self) -> None:
        if self.messages and self.messages[-1].get("role") == "user":
            self.messages.pop()
        self.sync_active_data()


# Для обратной совместимости со старыми вызовами _SessionState
_SessionState = SessionState


def print_status(
    model: str,
    think_mode: bool | str,
    stats_enabled: bool,
    messages: Sequence[ChatMessage],
    totals: SessionTotals | None = None,
) -> None:
    """Вывести панель текущего состояния сессии и модели."""
    think_text = "выключен" if not think_mode else str(think_mode)
    turns = sum(message["role"] == "user" for message in messages)
    lines = [
        f"Модель: [bold]{model}[/bold]",
        "ИИ: [bold]W1x[/bold]  •  разработчик: [bold]W1bbex[/bold]",
        f"Сборка: [bold]{APP_VERSION}[/bold]",
        f"Thinking: [bold]{think_text}[/bold]",
        "Инструменты: [bold]поиск, время, калькулятор, Python, память, "
        "файлы, Antigravity[/bold]",
        f"Папка проекта: [bold]{escape_markup(str(config.ACTIVE_FILE_ROOT)) if config.ACTIVE_FILE_ROOT else 'не выбрана'}[/bold]",
        f"Статистика: [bold]{'включена' if stats_enabled else 'выключена'}[/bold]",
        f"Сообщений пользователя: [bold]{turns}[/bold]",
    ]
    if totals and totals.get("requests", 0) > 0:
        reqs = totals.get("requests", 0)
        tot_tok = totals.get("total_tokens", 0)
        prompt_tok = totals.get("prompt_tokens", 0)
        out_tok = totals.get("output_tokens", 0)
        lines.append(
            f"Токенов за сессию: [bold]{tot_tok:,}[/bold] ([dim]вход: {prompt_tok:,}, выход: {out_tok:,}[/dim]) "
            f"в [bold]{reqs}[/bold] ответах"
        )
    _get_console().print(
        Panel(
            "\n".join(lines),
            title="Состояние",
            border_style="cyan",
        )
    )


def print_header(model: str) -> None:
    """Вывести стартовый баннер с активной моделью."""
    _get_console().print(
        Panel(
            f"[bold green]{model}[/bold green]\n"
            f"[dim]W1x CLI  •  v{APP_VERSION}[/dim]\n"
            "[dim]Enter — отправить  •  Shift+Enter — новая строка  •  "
            "/help — команды[/dim]",
            title="W1x",
            border_style="green",
        )
    )


def print_help() -> None:
    """Вывести справку по горячим клавишам и слэш-командам."""
    _get_console().print(
        """
[bold]Ввод[/bold]
Enter               отправить сообщение
Shift+Enter          новая строка
Ctrl+Enter           новая строка
Alt+Enter            запасной вариант новой строки
Ctrl+S               вернуть прошлый вопрос в строку ввода и стереть из памяти
Ctrl+Q               выгрузить все модели и выйти
Ctrl+Backspace       удалить слово слева
Ctrl+Delete          удалить слово справа
Ctrl+A               выделить весь текст; Delete/Backspace — очистить
Ctrl+X               вырезать выделенный текст (или весь ввод) в буфер
Ctrl+C               копировать выделение / остановить генерацию
Ctrl+V               вставить текст из буфера обмена
/ком...             серое продолжение и список; Tab/Enter — подставить

[bold]Команды[/bold]
[cyan]/model[/cyan]             сменить модель (стрелки ↑/↓ или Enter)
[cyan]/pull [модели...][/cyan]  скачать модели из библиотеки Ollama
[cyan]/unload[/cyan], [cyan]/free[/cyan]    выгрузить активные модели из VRAM
[cyan]/config[/cyan]            настройки контекста, VRAM и модели
[cyan]/models[/cyan]            список установленных моделей
[cyan]/new[/cyan], [cyan]/clear[/cyan]       начать новый диалог
[cyan]/undo[/cyan]              удалить последний вопрос и ответ
[cyan]/retry[/cyan]             заново ответить на последний вопрос
[cyan]/think[/cyan]             включить/выключить размышления
[cyan]/think low|medium|high[/cyan]  выбрать уровень размышлений
[cyan]/tools[/cyan]             доступные инструменты
[cyan]/agy задача[/cyan]        передать задачу Antigravity напрямую
[cyan]/workspace путь[/cyan]    выбрать папку проекта для безопасного чтения
[cyan]/stats[/cyan]             статистика прошлого и следующих ответов
[cyan]/status[/cyan]            текущие настройки
[cyan]/system[/cyan]            показать системную инструкцию
[cyan]/system текст[/cyan]      заменить системную инструкцию
[cyan]/copy[/cyan]              скопировать последний ответ
[cyan]/testpy[/cyan]            протестировать последний Python-код в отдельном окне
[cyan]/save[/cyan] [имя]        сохранить диалог в Markdown
[cyan]/memory[/cyan]            показать состояние постоянной памяти
[cyan]/forget текст[/cyan]      удалить подходящий факт из памяти
[cyan]/cls[/cyan]               очистить экран
[cyan]/help[/cyan]              эта справка
[cyan]q[/cyan], [cyan]/exit[/cyan]           выгрузить модели и выйти
"""
    )
