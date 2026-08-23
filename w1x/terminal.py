# -*- coding: utf-8 -*-
"""Интерфейс терминала, темы Rich, буфер обмена Windows и горячие клавиши prompt_toolkit."""
import os
import sys
import time
import shutil
import threading
import subprocess

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

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

from .config import (
    COMMAND_HINTS,
    SPINNER,
    SPINNER_DELAY,
    ACTIVE_SESSION_DATA,
)


def configure_utf8_console():
    """Включить UTF-8 и Virtual Terminal Processing в консоли Windows."""
    if os.name == "nt":
        subprocess.run(
            ["cmd.exe", "/d", "/c", "chcp", "65001"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
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
            reconfigure(encoding="utf-8", errors="replace")


configure_utf8_console()

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


def _stream_preview(text, role):
    flattened = " ".join((text or "").split())
    preview = flattened[-48:] if flattened else "…"
    label = "Размышляет › " if role == "thinking" else "Печатает › "
    style = "dim" if role == "thinking" else "cyan"
    return Text(f"{label}{preview}", style=style, overflow="ellipsis", no_wrap=True)


def render_message(text, role="assistant", *, streaming=False, live=None):
    """Единая точка вывода текста модели, thinking и инструментов."""
    text = "" if text is None else str(text)

    if streaming:
        renderable = _stream_preview(text, role)
        if live is not None:
            live.update(renderable, refresh=True)
        else:
            console.print(renderable)
        return

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
        renderable = Text()
        renderable.append("Инструмент › ", style="bold magenta")
        renderable.append(text)
    elif role == "assistant_label":
        renderable = Text(text, style="bold cyan")
    elif role == "notice":
        renderable = Text(text, style="yellow")
    else:
        renderable = Text(text)

    console.print(renderable)


def uses_modern_windows_terminal():
    if os.name != "nt":
        return True
    return bool(
        os.environ.get("WT_SESSION")
        or os.environ.get("TERM_PROGRAM")
        or os.environ.get("ConEmuANSI") == "ON"
        or os.environ.get("ANSICON")
    )


def show_terminal_emoji_hint():
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
    from prompt_toolkit.input.win32 import ConsoleInputReader

    _original_event_to_key_presses = ConsoleInputReader._event_to_key_presses

    def _event_to_key_presses_with_ctrl_backspace(self, event):
        ctrl_pressed = event.ControlKeyState & (
            self.LEFT_CTRL_PRESSED | self.RIGHT_CTRL_PRESSED
        )
        shift_pressed = bool(event.ControlKeyState & self.SHIFT_PRESSED)
        if event.VirtualKeyCode == 8 and ctrl_pressed:
            return [KeyPress(Keys.ControlW, "")]
        if event.VirtualKeyCode == 13 and shift_pressed:
            return [KeyPress(Keys.ControlJ, "\n")]
        return _original_event_to_key_presses(self, event)

    ConsoleInputReader._event_to_key_presses = (
        _event_to_key_presses_with_ctrl_backspace
    )


def _windows_shift_is_pressed():
    """Проверить зажатие Shift напрямую через Win32 API."""
    if os.name != "nt":
        return False
    try:
        import ctypes

        return bool(ctypes.windll.user32.GetAsyncKeyState(0x10) & 0x8000)
    except Exception:
        return False


def _windows_ctrl_is_pressed():
    """Проверить Ctrl напрямую через Win32 API."""
    if os.name != "nt":
        return False
    try:
        import ctypes

        return bool(ctypes.windll.user32.GetAsyncKeyState(0x11) & 0x8000)
    except Exception:
        return False


def _read_clipboard_text():
    """Получить Unicode-текст из буфера Windows без порчи кириллицы/emoji."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

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
            if user32.OpenClipboard(None):
                try:
                    handle = user32.GetClipboardData(CF_UNICODETEXT)
                    if handle:
                        text_ptr = kernel32.GlobalLock(handle)
                        if text_ptr:
                            result = str(text_ptr)
                            kernel32.GlobalUnlock(handle)
                            return result
                finally:
                    user32.CloseClipboard()
        except Exception:
            pass

    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    try:
        root.update()
        return root.clipboard_get()
    finally:
        root.destroy()


def copy_to_clipboard(text):
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
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
    except Exception:
        pass


bindings = KeyBindings()
_last_paste_burst_time = 0.0


def _is_multiline_paste_event(event) -> bool:
    global _last_paste_burst_time
    now = time.monotonic()

    key_processor = getattr(event.app, "key_processor", None)
    if key_processor and getattr(key_processor, "input_queue", None):
        if any(
            not isinstance(k.key, Keys)
            or k.key in {Keys.ControlM, Keys.ControlJ, Keys.BracketedPaste, Keys.Any}
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

                count = wintypes.DWORD(0)
                if ctypes.windll.kernel32.GetNumberOfConsoleInputEvents(
                    handle, ctypes.byref(count)
                ):
                    if count.value > 0:
                        _last_paste_burst_time = now
                        return True
        except Exception:
            pass

    return False


@bindings.add("enter")
def submit_message(event):
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
def bracketed_paste_handler(event):
    text = event.data.replace("\r\n", "\n").replace("\r", "\n")
    event.current_buffer.insert_text(text)


@bindings.add("c-j")
def insert_line_ctrl_j(event):
    event.current_buffer.insert_text("\n")


@bindings.add("escape", "c-j")
def insert_line_ctrl_enter(event):
    event.current_buffer.insert_text("\n")


@bindings.add("escape", "enter")
def insert_line_alt_enter(event):
    event.current_buffer.insert_text("\n")


@bindings.add("c-q")
def exit_app_ctrl_q(event):
    event.app.exit(result="/exit")


@bindings.add("c-s")
def undo_and_restore_prompt(event):
    from .app import save_current_chat
    messages = ACTIVE_SESSION_DATA.get("messages", [])
    system_prompt = ACTIVE_SESSION_DATA.get("system_prompt", "")
    last_user_prompt = ""

    if messages and messages[-1].get("role") == "assistant":
        messages.pop()
    if messages and messages[-1].get("role") == "user":
        last_user_prompt = messages.pop().get("content", "")

    if last_user_prompt:
        save_current_chat(system_prompt, messages)
        buffer = event.app.current_buffer
        buffer.text = last_user_prompt
        buffer.cursor_position = len(buffer.text)


@bindings.add("c-delete")
def delete_word_right(event):
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
def select_all_input(event):
    buffer = event.app.current_buffer
    if not buffer.text:
        return

    buffer.cursor_position = 0
    buffer.start_selection(selection_type=SelectionType.CHARACTERS)
    buffer.cursor_position = len(buffer.text)


@bindings.add("c-x")
def cut_selection_to_clipboard(event):
    """Ctrl+X вырезает выделенный текст (или весь ввод) и копирует в буфер обмена."""
    buffer = event.app.current_buffer
    if buffer.selection_state is not None:
        data = buffer.cut_selection()
        if data and data.text:
            copy_to_clipboard(data.text)
    elif buffer.text:
        text_to_cut = buffer.text
        buffer.text = ""
        buffer.cursor_position = 0
        copy_to_clipboard(text_to_cut)


@bindings.add("c-c")
def copy_selection_or_abort(event):
    """Ctrl+C при наличии выделения копирует текст в буфер, иначе прерывает."""
    buffer = event.app.current_buffer
    if buffer.selection_state is not None:
        data = buffer.copy_selection()
        if data and data.text:
            copy_to_clipboard(data.text)
        buffer.exit_selection()
    else:
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")


@bindings.add("c-v", eager=True)
def paste_clipboard(event):
    try:
        text = _read_clipboard_text()
    except Exception:
        event.app.output.bell()
        return

    event.app.current_buffer.insert_text(
        text.replace("\r\n", "\n").replace("\r", "\n")
    )


@bindings.add("delete")
def delete_or_clear_selection(event):
    buffer = event.app.current_buffer
    if buffer.selection_state is not None:
        buffer.cut_selection()
    else:
        buffer.delete(count=1)


def _delete_word_left(buffer):
    position = buffer.document.find_start_of_previous_word()
    count = (
        len(buffer.document.text_before_cursor)
        if position is None
        else -position
    )
    if count > 0:
        buffer.delete_before_cursor(count=count)


@bindings.add("c-h")
def backspace_or_clear_selection(event):
    buffer = event.app.current_buffer
    if buffer.selection_state is not None:
        buffer.cut_selection()
    elif _windows_ctrl_is_pressed():
        _delete_word_left(buffer)
    else:
        buffer.delete_before_cursor(count=1)


@bindings.add("c-w")
def delete_word_left(event):
    buffer = event.app.current_buffer
    if buffer.selection_state is not None:
        buffer.cut_selection()
        return
    _delete_word_left(buffer)


class SlashCommandCompleter(Completer):
    def get_completions(self, document, complete_event):
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
    def get_suggestion(self, buffer, document):
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


_session = None


def get_session():
    """Ленивая инициализация PromptSession с защитой от запуска в неинтерактивных пайпах."""
    global _session
    if _session is None:
        try:
            _session = PromptSession(
                key_bindings=bindings,
                history=InMemoryHistory(),
                completer=SlashCommandCompleter(),
                complete_while_typing=True,
                auto_suggest=SlashCommandAutoSuggest(),
            )
        except Exception:
            try:
                from prompt_toolkit.output import DummyOutput
                _session = PromptSession(
                    key_bindings=bindings,
                    history=InMemoryHistory(),
                    completer=SlashCommandCompleter(),
                    complete_while_typing=True,
                    auto_suggest=SlashCommandAutoSuggest(),
                    output=DummyOutput(),
                )
            except Exception:
                _session = None
    return _session


class _SessionProxy:
    def __getattr__(self, name):
        s = get_session()
        if s is None:
            raise RuntimeError("PromptSession недоступен в этом окружении.")
        return getattr(s, name)

    def prompt(self, *args, **kwargs):
        s = get_session()
        if s is None:
            return input(str(args[0]) if args else "")
        return s.prompt(*args, **kwargs)


session = _SessionProxy()


def read_input():
    return session.prompt(
        HTML("<b><ansigreen>Ты › </ansigreen></b>"),
        multiline=True,
        prompt_continuation=lambda width, line_number, wrap_count: HTML(
            "<ansibrightblack>... │ </ansibrightblack>"
        ),
    )


def run_with_spinner(text, function):
    finished = threading.Event()
    result = {"value": None, "error": None}
    terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    max_text_width = max(12, terminal_width - 4)
    display_text = (
        text
        if len(text) <= max_text_width
        else text[: max_text_width - 1] + "…"
    )

    def worker():
        try:
            result["value"] = function()
        except Exception as error:
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
        from .tools import cancel_active_agy
        cancel_active_agy()
        raise
    finally:
        sys.stdout.write("\r" + " " * (len(display_text) + 4) + "\r")
        sys.stdout.flush()

    thread.join()
    if result["error"] is not None:
        raise result["error"]
    return result["value"]
