"""Кроссплатформенный неблокирующий опрос клавиш и управление режимами терминала."""
import os
import sys
from contextlib import contextmanager
from typing import Any

_UNIX_RAW_DEPTH = 0
_UNIX_SAVED_ATTRS: Any = None


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


@contextmanager
def raw_terminal_mode() -> Any:
    """Контекстный менеджер для удержания raw-режима терминала на Unix без переключения на каждом тике."""
    global _UNIX_RAW_DEPTH, _UNIX_SAVED_ATTRS
    if os.name == "nt" or not sys.stdin.isatty():
        yield
        return

    import termios
    import tty

    termios_mod: Any = termios
    tty_mod: Any = tty
    try:
        fd = sys.stdin.fileno()
    except Exception:
        yield
        return

    should_restore = False
    try:
        if _UNIX_RAW_DEPTH == 0:
            _UNIX_SAVED_ATTRS = termios_mod.tcgetattr(fd)
            (getattr(tty_mod, "setcbreak", None) or tty_mod.setraw)(fd)
            should_restore = True
        _UNIX_RAW_DEPTH += 1
        yield
    finally:
        _UNIX_RAW_DEPTH = max(0, _UNIX_RAW_DEPTH - 1)
        if _UNIX_RAW_DEPTH == 0 and should_restore and _UNIX_SAVED_ATTRS is not None:
            try:
                termios_mod.tcsetattr(fd, termios_mod.TCSADRAIN, _UNIX_SAVED_ATTRS)
            except Exception:
                pass
            _UNIX_SAVED_ATTRS = None


@contextmanager
def cooked_terminal_mode() -> Any:
    """Временный возврат канонического режима терминала (для prompt-сессий внутри raw-циклов)."""
    global _UNIX_RAW_DEPTH, _UNIX_SAVED_ATTRS
    if (
        os.name == "nt"
        or not sys.stdin.isatty()
        or _UNIX_RAW_DEPTH == 0
        or _UNIX_SAVED_ATTRS is None
    ):
        yield
        return

    import termios
    import tty

    termios_mod: Any = termios
    tty_mod: Any = tty
    try:
        fd = sys.stdin.fileno()
        current_raw_attrs = termios_mod.tcgetattr(fd)
        termios_mod.tcsetattr(fd, termios_mod.TCSADRAIN, _UNIX_SAVED_ATTRS)
    except Exception:
        yield
        return

    try:
        yield
    finally:
        try:
            termios_mod.tcsetattr(fd, termios_mod.TCSADRAIN, current_raw_attrs)
        except Exception:
            try:
                (getattr(tty_mod, "setcbreak", None) or tty_mod.setraw)(fd)
            except Exception:
                pass


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
    except Exception:
        return None

    in_managed_raw = _UNIX_RAW_DEPTH > 0
    old_settings = None
    if not in_managed_raw:
        try:
            old_settings = termios_mod.tcgetattr(fd)
            (getattr(tty_mod, "setcbreak", None) or tty_mod.setraw)(fd)
        except Exception:
            return None

    try:
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
                if ch3 in ("1", "7"):
                    _read_unix_byte(fd, select, 0.02)
                    return "HOME"
                if ch3 in ("4", "8"):
                    _read_unix_byte(fd, select, 0.02)
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
                if ch3 == "H":
                    return "HOME"
                if ch3 == "F":
                    return "END"
            return None
        if ch in ("\r", "\n"):
            return "ENTER"
        if ch in ("q", "Q", "\x03", "\x11"):
            return "EXIT"
        if ch.isascii() and ch.isdigit():
            return ch
        return None
    finally:
        if not in_managed_raw and old_settings is not None:
            try:
                termios_mod.tcsetattr(fd, termios_mod.TCSADRAIN, old_settings)
            except Exception:
                pass


def poll_key_crossplatform() -> str | None:
    """Неблокирующее чтение навигационных клавиш на Windows и Unix-терминалах."""
    if not sys.stdin.isatty():
        return "EXIT"
    if os.name == "nt":
        return _poll_key_windows()
    return _poll_key_unix()
