"""Интерфейс терминала, темы Rich, буфер обмена Windows и горячие клавиши prompt_toolkit."""
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any, TypeVar, cast

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.selection import SelectionType
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

from .config import (
    ACTIVE_SESSION_DATA,
    COMMAND_HINTS,
    SPINNER,
    SPINNER_DELAY,
)
from .stats import format_duration
from .types import ChatMessage, StatsRecord

_T = TypeVar("_T")
_undo_persist_callback: Callable[[str, list[ChatMessage]], Any] | None = None


def set_undo_persist_callback(
    callback: Callable[[str, list[ChatMessage]], Any] | None,
) -> None:
    """Зарегистрировать функцию сохранения чата для горячей клавиши Ctrl+S."""
    global _undo_persist_callback
    _undo_persist_callback = callback


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


ANSI_SEQUENCES["\x1b[13;2u"] = Keys.ControlJ
ANSI_SEQUENCES["\x1b[13;5u"] = Keys.ControlJ
ANSI_SEQUENCES["\x1b[27;2;13~"] = Keys.ControlJ
ANSI_SEQUENCES["\x1b[27;5;13~"] = Keys.ControlJ
ANSI_SEQUENCES["\x1b[3;5~"] = Keys.ControlDelete

if os.name == "nt":
    import prompt_toolkit.input.win32 as win32_input

    win32_mod: Any = win32_input
    ConsoleInputReader = win32_mod.ConsoleInputReader
    _original_event_to_key_presses = ConsoleInputReader._event_to_key_presses

    def _event_to_key_presses_with_ctrl_backspace(
        self: Any, event: Any
    ) -> list[KeyPress]:
        if not getattr(event, "KeyDown", getattr(event, "bKeyDown", True)):
            return []
        ctrl_pressed = event.ControlKeyState & (
            self.LEFT_CTRL_PRESSED | self.RIGHT_CTRL_PRESSED
        )
        shift_pressed = bool(event.ControlKeyState & self.SHIFT_PRESSED)
        if event.VirtualKeyCode == 8 and ctrl_pressed:
            return [KeyPress(Keys.ControlW, "")]
        if event.VirtualKeyCode == 13 and shift_pressed:
            return [KeyPress(Keys.ControlJ, "\n")]
        return cast(list[KeyPress], _original_event_to_key_presses(self, event))

    cast(Any, ConsoleInputReader)._event_to_key_presses = (
        _event_to_key_presses_with_ctrl_backspace
    )


def _windows_shift_is_pressed() -> bool:
    """Проверить зажатие Shift напрямую через Win32 API."""
    if os.name != "nt":
        return False
    try:
        import ctypes

        ctypes_mod: Any = ctypes
        return bool(ctypes_mod.windll.user32.GetAsyncKeyState(0x10) & 0x8000)
    except Exception:
        return False


def _windows_ctrl_is_pressed() -> bool:
    """Проверить Ctrl напрямую через Win32 API."""
    if os.name != "nt":
        return False
    try:
        import ctypes

        ctypes_mod: Any = ctypes
        return bool(ctypes_mod.windll.user32.GetAsyncKeyState(0x11) & 0x8000)
    except Exception:
        return False


def _read_clipboard_text() -> str:
    """Получить текст из буфера обмена с поддержкой Windows, macOS, Linux и tkinter fallback."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            ctypes_mod: Any = ctypes
            user32 = ctypes_mod.windll.user32
            kernel32 = ctypes_mod.windll.kernel32

            user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
            user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
            user32.OpenClipboard.argtypes = [wintypes.HWND]
            user32.OpenClipboard.restype = wintypes.BOOL
            user32.CloseClipboard.argtypes = []
            user32.CloseClipboard.restype = wintypes.BOOL
            user32.GetClipboardData.argtypes = [wintypes.UINT]
            user32.GetClipboardData.restype = wintypes.HANDLE
            kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
            kernel32.GlobalLock.restype = ctypes.c_wchar_p
            kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
            kernel32.GlobalUnlock.restype = wintypes.BOOL

            CF_UNICODETEXT = 13
            if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
                return ""
            if user32.OpenClipboard(None):
                try:
                    handle = user32.GetClipboardData(CF_UNICODETEXT)
                    if handle:
                        text_ptr = kernel32.GlobalLock(handle)
                        if text_ptr is not None:
                            try:
                                return str(text_ptr)
                            finally:
                                kernel32.GlobalUnlock(handle)
                    return ""
                finally:
                    user32.CloseClipboard()
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            proc = subprocess.run(["pbpaste"], capture_output=True, text=True, check=True)
            return proc.stdout
        except Exception:
            pass
    else:
        for cmd in (
            ["wl-paste", "--no-newline"],
            ["xclip", "-o", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--output"],
        ):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return proc.stdout
            except Exception:
                continue

    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    try:
        root.update()
        return root.clipboard_get()
    finally:
        root.destroy()


def copy_to_clipboard(text: str) -> None:
    """Скопировать текст в буфер обмена кроссплатформенно (Windows, macOS, Linux)."""
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", "Set-Clipboard"],
                input=text,
                text=True,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return
        except Exception:
            pass
    elif sys.platform == "darwin":  # macOS
        try:
            subprocess.run(["pbcopy"], input=text, text=True, check=True)
            return
        except Exception:
            pass
    else:  # Linux (Wayland / X11)
        for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
            try:
                subprocess.run(cmd, input=text, text=True, check=True)
                return
            except Exception:
                continue

    # Универсальный fallback через tkinter
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        try:
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()
        finally:
            root.destroy()
        return
    except Exception as exc:
        raise RuntimeError("Буфер обмена недоступен в текущем окружении.") from exc


bindings = KeyBindings()
_last_paste_burst_time = 0.0


def _is_multiline_paste_event(event: Any) -> bool:
    global _last_paste_burst_time
    now = time.monotonic()

    key_processor = getattr(event.app, "key_processor", None)
    if key_processor and getattr(key_processor, "input_queue", None):
        if any(
            (isinstance(k.key, str) and ("\n" in k.key or "\r" in k.key))
            or k.key in {Keys.ControlM, Keys.ControlJ, Keys.BracketedPaste}
            for k in key_processor.input_queue
        ):
            _last_paste_burst_time = now
            return True

    if (now - _last_paste_burst_time) < 0.08:
        return True

    if os.name == "nt":
        try:
            input_obj = getattr(event.app, "input", None)
            reader = getattr(input_obj, "console_input_reader", None)
            handle = getattr(reader, "handle", None)
            if handle:
                import ctypes
                from ctypes import wintypes

                class _KEY_EVENT_RECORD(ctypes.Structure):
                    class _UCHAR(ctypes.Union):
                        _fields_ = [
                            ("UnicodeChar", wintypes.WCHAR),
                            ("AsciiChar", ctypes.c_char),
                        ]

                    _fields_ = [
                        ("bKeyDown", wintypes.BOOL),
                        ("wRepeatCount", wintypes.WORD),
                        ("wVirtualKeyCode", wintypes.WORD),
                        ("wVirtualScanCode", wintypes.WORD),
                        ("uChar", _UCHAR),
                        ("dwControlKeyState", wintypes.DWORD),
                    ]

                class _INPUT_RECORD(ctypes.Structure):
                    class _EVENT(ctypes.Union):
                        _fields_ = [
                            ("KeyEvent", _KEY_EVENT_RECORD),
                            ("_pad", wintypes.DWORD * 4),
                        ]

                    _fields_ = [
                        ("EventType", wintypes.WORD),
                        ("Event", _EVENT),
                    ]

                ctypes_mod: Any = ctypes
                buffer_size = 16
                records = (_INPUT_RECORD * buffer_size)()
                read_count = wintypes.DWORD(0)
                if ctypes_mod.windll.kernel32.PeekConsoleInputW(
                    handle, records, buffer_size, ctypes.byref(read_count)
                ):
                    for idx in range(read_count.value):
                        rec = records[idx]
                        if rec.EventType == 1 and rec.Event.KeyEvent.bKeyDown:
                            _last_paste_burst_time = now
                            return True
        except Exception:
            pass

    return False


@bindings.add("enter")
def submit_message(event: Any) -> None:
    """Enter выбирает подсказку, переносит строку при Shift/вставке либо отправляет."""
    buffer = event.current_buffer
    completion_state = buffer.complete_state
    if completion_state is not None and completion_state.current_completion:
        buffer.apply_completion(completion_state.current_completion)
        return
    if buffer.suggestion is not None and buffer.suggestion.text:
        buffer.insert_text(buffer.suggestion.text)
        return

    if _windows_shift_is_pressed():
        buffer.insert_text("\n")
        return

    if _is_multiline_paste_event(event):
        buffer.insert_text("\n")
        return

    buffer.validate_and_handle()


@bindings.add(Keys.BracketedPaste)
def bracketed_paste_handler(event: Any) -> None:
    text = event.data.replace("\r\n", "\n").replace("\r", "\n")
    event.current_buffer.insert_text(text)


@bindings.add("c-j")
def insert_line_ctrl_j(event: Any) -> None:
    event.current_buffer.insert_text("\n")


@bindings.add("escape", "c-j")
def insert_line_ctrl_enter(event: Any) -> None:
    event.current_buffer.insert_text("\n")


@bindings.add("escape", "enter")
def insert_line_alt_enter(event: Any) -> None:
    event.current_buffer.insert_text("\n")


@bindings.add("c-q")
def exit_app_ctrl_q(event: Any) -> None:
    event.app.exit(result="/exit")


@bindings.add("c-s")
def undo_and_restore_prompt(event: Any) -> None:
    cancelled_prompt = str(ACTIVE_SESSION_DATA.pop("cancelled_prompt", "") or "")
    if cancelled_prompt:
        buffer = event.app.current_buffer
        buffer.text = cancelled_prompt
        buffer.cursor_position = len(buffer.text)
        return

    messages: list[ChatMessage] = ACTIVE_SESSION_DATA.get("messages", [])
    system_prompt = str(ACTIVE_SESSION_DATA.get("system_prompt", "") or "")
    last_user_prompt = ""

    if messages and messages[-1].get("role") == "user":
        last_user_prompt = messages.pop().get("content", "")
    elif (
        len(messages) >= 2
        and messages[-1].get("role") == "assistant"
        and messages[-2].get("role") == "user"
    ):
        messages.pop()
        last_user_prompt = messages.pop().get("content", "")

    if last_user_prompt:
        if callable(_undo_persist_callback):
            _undo_persist_callback(system_prompt, messages)
        buffer = event.app.current_buffer
        buffer.text = last_user_prompt
        buffer.cursor_position = len(buffer.text)


@bindings.add("c-delete")
def delete_word_right(event: Any) -> None:
    buffer = event.app.current_buffer
    if buffer.selection_state is not None:
        buffer.cut_selection()
        return

    position = buffer.document.find_next_word_ending(
        include_current_position=True
    )
    count = (
        len(buffer.document.text_after_cursor)
        if position is None
        else position
    )
    buffer.delete(count=count)


@bindings.add("c-a")
def select_all_input(event: Any) -> None:
    buffer = event.app.current_buffer
    if not buffer.text:
        return

    buffer.cursor_position = 0
    buffer.start_selection(selection_type=SelectionType.CHARACTERS)
    buffer.cursor_position = len(buffer.text)


@bindings.add("c-x")
def cut_selection_to_clipboard(event: Any) -> None:
    """Ctrl+X вырезает выделенный текст (или весь ввод) и копирует в буфер обмена."""
    buffer = event.app.current_buffer
    if buffer.selection_state is not None:
        data = buffer.cut_selection()
        if data and data.text:
            try:
                copy_to_clipboard(data.text)
            except Exception:
                event.app.output.bell()
    elif buffer.text:
        text_to_cut = buffer.text
        try:
            copy_to_clipboard(text_to_cut)
        except Exception:
            event.app.output.bell()
            return
        buffer.text = ""
        buffer.cursor_position = 0


@bindings.add("c-c")
def copy_selection_or_abort(event: Any) -> None:
    """Ctrl+C при наличии выделения копирует текст в буфер, иначе прерывает."""
    buffer = event.app.current_buffer
    if buffer.selection_state is not None:
        data = buffer.copy_selection()
        if data and data.text:
            try:
                copy_to_clipboard(data.text)
            except Exception:
                event.app.output.bell()
        buffer.exit_selection()
    else:
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")


@bindings.add("c-v", eager=True)
def paste_clipboard(event: Any) -> None:
    try:
        text = _read_clipboard_text()
    except Exception:
        event.app.output.bell()
        return

    event.app.current_buffer.insert_text(
        text.replace("\r\n", "\n").replace("\r", "\n")
    )


@bindings.add("delete")
def delete_or_clear_selection(event: Any) -> None:
    buffer = event.app.current_buffer
    if buffer.selection_state is not None:
        buffer.cut_selection()
    else:
        buffer.delete(count=1)


def _delete_word_left(buffer: Any) -> None:
    position = buffer.document.find_start_of_previous_word()
    count = (
        len(buffer.document.text_before_cursor)
        if position is None
        else -position
    )
    if count > 0:
        buffer.delete_before_cursor(count=count)


@bindings.add("c-h")
def backspace_or_clear_selection(event: Any) -> None:
    buffer = event.app.current_buffer
    if buffer.selection_state is not None:
        buffer.cut_selection()
    elif _windows_ctrl_is_pressed():
        _delete_word_left(buffer)
    else:
        buffer.delete_before_cursor(count=1)


@bindings.add("c-w")
def delete_word_left(event: Any) -> None:
    buffer = event.app.current_buffer
    if buffer.selection_state is not None:
        buffer.cut_selection()
        return
    _delete_word_left(buffer)


class SlashCommandCompleter(Completer):
    def get_completions(
        self, document: Any, complete_event: Any
    ) -> Iterable[Completion]:
        typed = document.text_before_cursor
        if not typed.startswith("/") or any(char.isspace() for char in typed):
            return

        normalized = typed.casefold()
        for command, description in COMMAND_HINTS.items():
            if command.casefold().startswith(normalized):
                yield Completion(
                    command,
                    start_position=-len(typed),
                    display=command,
                    display_meta=description,
                )


class SlashCommandAutoSuggest(AutoSuggest):
    def get_suggestion(self, buffer: Any, document: Any) -> Suggestion | None:
        typed = document.text_before_cursor
        if not typed.startswith("/") or any(char.isspace() for char in typed):
            return None

        normalized = typed.casefold()
        matches = [
            command
            for command in COMMAND_HINTS
            if command.casefold().startswith(normalized)
            and len(command) > len(typed)
        ]
        if not matches:
            return None

        command = matches[0]
        return Suggestion(command[len(typed):])


_session: PromptSession[str] | None = None
_secondary_session: PromptSession[str] | None = None


def _create_prompt_session(*, with_repl_bindings: bool) -> PromptSession[str] | None:
    kwargs: dict[str, Any] = (
        {
            "key_bindings": bindings,
            "history": InMemoryHistory(),
            "completer": SlashCommandCompleter(),
            "complete_while_typing": True,
            "auto_suggest": SlashCommandAutoSuggest(),
        }
        if with_repl_bindings
        else {"history": None}
    )
    try:
        return PromptSession(**kwargs)
    except Exception:
        try:
            from prompt_toolkit.output import DummyOutput

            return PromptSession(output=DummyOutput(), **kwargs)
        except Exception:
            return None


def get_session() -> PromptSession[str] | None:
    """Ленивая инициализация PromptSession с защитой от запуска в неинтерактивных пайпах."""
    global _session
    if _session is None:
        _session = _create_prompt_session(with_repl_bindings=True)
    return _session


def get_secondary_session() -> PromptSession[str] | None:
    """Изолированный PromptSession для меню/подтверждений без Ctrl+S/Ctrl+Q и автодополнения чата."""
    global _secondary_session
    if _secondary_session is None:
        _secondary_session = _create_prompt_session(with_repl_bindings=False)
    return _secondary_session


class _SessionProxy:
    def __init__(self, factory: Callable[[], PromptSession[str] | None]) -> None:
        self._factory = factory

    def __getattr__(self, name: str) -> Any:
        s = self._factory()
        if s is None:
            raise RuntimeError("PromptSession недоступен в этом окружении.")
        return getattr(s, name)

    def prompt(self, *args: Any, **kwargs: Any) -> str:
        s = self._factory()
        if s is None:
            return input(str(args[0]) if args else "")
        return cast(str, s.prompt(*args, **kwargs))


session = _SessionProxy(get_session)
secondary_session = _SessionProxy(get_secondary_session)


def read_input() -> str:
    return session.prompt(
        HTML("<b><ansigreen>Ты › </ansigreen></b>"),
        multiline=True,
        prompt_continuation=lambda width, line_number, wrap_count: HTML(
            "<ansibrightblack>... │ </ansibrightblack>"
        ),
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
            time.sleep(SPINNER_DELAY)
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


def _poll_key_windows() -> str | None:
    import msvcrt

    msvcrt_mod: Any = msvcrt
    if not msvcrt_mod.kbhit():
        return None
    key = msvcrt_mod.getch()
    if key in (b"\x00", b"\xe0"):
        ext = msvcrt_mod.getch()
        if ext == b"H":
            return "UP"
        if ext == b"P":
            return "DOWN"
        if ext == b"I":
            return "PGUP"
        if ext == b"Q":
            return "PGDN"
        if ext == b"G":
            return "HOME"
        if ext == b"O":
            return "END"
        return None
    if key in (b"\r", b"\n"):
        return "ENTER"
    if key in (b"q", b"Q", b"\x1b", b"\x03", b"\x11"):
        return "EXIT"
    if key.isdigit():
        return str(key.decode("ascii", errors="ignore"))
    return None


def _read_unix_byte(fd: int, select_mod: Any, timeout: float = 0.02) -> bytes:
    rlist, _, _ = select_mod.select([fd], [], [], timeout)
    if not rlist:
        return b""
    return os.read(fd, 1)


def _poll_key_unix() -> str | None:
    import select
    import termios
    import tty

    termios_mod: Any = termios
    tty_mod: Any = tty

    if not sys.stdin.isatty():
        return None
    try:
        fd = sys.stdin.fileno()
        old_settings = termios_mod.tcgetattr(fd)
    except Exception:
        return None
    try:
        tty_mod.setraw(fd)
        raw = _read_unix_byte(fd, select, 0.04)
        if not raw:
            return None
        ch = raw.decode("latin1")
        if ch == "\x1b":
            b2 = _read_unix_byte(fd, select, 0.02)
            if not b2:
                return "EXIT"
            ch2 = b2.decode("latin1")
            if ch2 == "[":
                b3 = _read_unix_byte(fd, select, 0.02)
                if not b3:
                    return None
                ch3 = b3.decode("latin1")
                if ch3 == "A":
                    return "UP"
                if ch3 == "B":
                    return "DOWN"
                if ch3 == "5":
                    _read_unix_byte(fd, select, 0.02)
                    return "PGUP"
                if ch3 == "6":
                    _read_unix_byte(fd, select, 0.02)
                    return "PGDN"
                if ch3 == "H":
                    return "HOME"
                if ch3 == "F":
                    return "END"
            elif ch2 == "O":
                b3 = _read_unix_byte(fd, select, 0.02)
                if not b3:
                    return None
                ch3 = b3.decode("latin1")
                if ch3 == "A":
                    return "UP"
                if ch3 == "B":
                    return "DOWN"
            return None
        if ch in ("\r", "\n"):
            return "ENTER"
        if ch in ("q", "Q", "\x03", "\x11"):
            return "EXIT"
        if ch.isascii() and ch.isdigit():
            return ch
        return None
    finally:
        termios_mod.tcsetattr(fd, termios_mod.TCSADRAIN, old_settings)


def poll_key_crossplatform() -> str | None:
    """Неблокирующее чтение навигационных клавиш на Windows и Unix-терминалах."""
    if os.name == "nt":
        return _poll_key_windows()
    return _poll_key_unix()
