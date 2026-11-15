"""Интерфейс терминала, темы Rich, Markdown-рендеринг и спиннеры ожидания."""
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

from .config import SPINNER, SPINNER_DELAY
from .stats import format_duration
from .types import StatsRecord

_T = TypeVar("_T")


def configure_utf8_console() -> None:
    """Включить UTF-8 и Virtual Terminal Processing в консоли Windows."""
    if os.name == "nt":
        try:
            import ctypes

            ctypes_mod: Any = ctypes
            kernel32 = ctypes_mod.windll.kernel32
            kernel32.SetConsoleCP(65001)
            kernel32.SetConsoleOutputCP(65001)
            for handle_id in (-11, -12):
                handle = kernel32.GetStdHandle(handle_id)
                mode = ctypes.c_ulong()
                if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                    mode.value |= 0x0004 | 0x0001
                    kernel32.SetConsoleMode(handle, mode)
        except Exception:
            pass

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


CHAT_THEME = Theme(
    {
        "markdown.bold": "bold bright_white",
        "markdown.code": "bold bright_cyan on grey11",
        "markdown.item.bullet": "bright_cyan",
        "markdown.link": "underline bright_blue",
        "markdown.h1": "bold bright_cyan",
        "markdown.h2": "bold cyan",
        "markdown.h3": "bold cyan",
    }
)
console = Console(theme=CHAT_THEME)


class CopyableCodeBlock(CodeBlock):
    """Markdown-код без декоративного отступа, портящего копирование."""

    def __rich_console__(
        self,
        console_: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        code = str(self.text).rstrip()
        yield Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            word_wrap=True,
            padding=0,
        )


class ChatMarkdown(Markdown):
    elements = Markdown.elements.copy()
    elements["fence"] = CopyableCodeBlock
    elements["code_block"] = CopyableCodeBlock


def _stream_preview(text: str | None, role: str) -> Text:
    flattened = " ".join((text or "").split())
    preview = flattened[-48:] if flattened else "…"
    label = "Размышляет › " if role == "thinking" else "Печатает › "
    style = "dim" if role == "thinking" else "cyan"
    return Text(f"{label}{preview}", style=style, overflow="ellipsis", no_wrap=True)


def render_message(
    text: object,
    role: str = "assistant",
    *,
    streaming: bool = False,
    live: Live | None = None,
) -> None:
    """Единая точка вывода текста модели, thinking и инструментов."""
    text = "" if text is None else str(text)

    if streaming:
        preview_renderable = _stream_preview(text, role)
        if live is not None:
            live.update(preview_renderable, refresh=True)
        else:
            console.print(preview_renderable)
        return

    renderable: RenderableType
    if role == "assistant":
        renderable = ChatMarkdown(text)
    elif role == "thinking":
        renderable = Panel(
            ChatMarkdown(text),
            title="Размышление",
            border_style="bright_black",
            padding=(0, 1),
        )
    elif role == "tool":
        tool_text = Text()
        tool_text.append("Инструмент › ", style="bold magenta")
        tool_text.append(text)
        renderable = tool_text
    elif role == "assistant_label":
        renderable = Text(text, style="bold cyan")
    elif role == "notice":
        renderable = Text(text, style="yellow")
    else:
        renderable = Text(text)

    console.print(renderable)


def uses_modern_windows_terminal() -> bool:
    if os.name != "nt":
        return True
    return bool(
        os.environ.get("WT_SESSION")
        or os.environ.get("TERM_PROGRAM")
        or os.environ.get("ConEmuANSI") == "ON"
        or os.environ.get("ANSICON")
    )


def show_terminal_emoji_hint() -> None:
    if not uses_modern_windows_terminal():
        render_message(
            "Для корректного отображения emoji рекомендуется Windows Terminal.",
            role="notice",
        )


def run_with_spinner(
    text: str,
    function: Callable[[], _T],
    on_interrupt: Callable[[], Any] | None = None,
) -> _T:
    finished = threading.Event()
    result: dict[str, Any] = {"value": None, "error": None}
    terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    max_text_width = max(12, terminal_width - 4)
    display_text = (
        text
        if len(text) <= max_text_width
        else text[: max_text_width - 1] + "…"
    )

    def worker() -> None:
        try:
            result["value"] = function()
        except BaseException as error:
            result["error"] = error
        finally:
            finished.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    frame = 0

    try:
        while not finished.is_set():
            symbol = SPINNER[frame % len(SPINNER)]
            sys.stdout.write(f"\r{symbol} {display_text}")
            sys.stdout.flush()
            frame += 1
            time.sleep(0)
            if finished.wait(SPINNER_DELAY):
                break
    except KeyboardInterrupt:
        if callable(on_interrupt):
            try:
                on_interrupt()
            except Exception:
                pass
        raise
    finally:
        sys.stdout.write("\r" + " " * (len(display_text) + 4) + "\r")
        sys.stdout.flush()
        thread.join(timeout=2.0)

    if result["error"] is not None:
        raise result["error"]
    return cast(_T, result["value"])


def print_stats(stats: StatsRecord | None) -> None:
    """Вывести в консоль статистику токенов и времени последнего ответа."""
    if stats is None:
        return

    if stats.get("source") == "direct_tool":
        duration = format_duration(stats.get("total_duration"))
        tool_label = stats.get("tool_name", "инструмент")
        console.print(f"[dim]Ответ: {tool_label}  •  время: {duration}[/dim]")
        return

    duration = format_duration(stats.get("total_duration"))
    eval_time = format_duration(stats.get("eval_duration"))
    prompt_tokens = stats.get("prompt_tokens", 0)
    output_tokens = stats.get("output_tokens", 0)
    speed = stats.get("speed", 0)
    console.print(
        f"[dim]Токены: {prompt_tokens} ввод + {output_tokens} ответ  •  "
        f"скорость: {speed:.1f} т/с  •  время: {duration} (генерация {eval_time})[/dim]"
    )
