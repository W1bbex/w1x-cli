"""Кроссплатформенная работа с буфером обмена (Windows, macOS, Linux)."""
import os
import subprocess
import sys
from typing import Any


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
            proc = subprocess.run(
                ["pbpaste"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2.0,
            )
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
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=2.0,
                )
                return proc.stdout
            except Exception:
                continue

    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        try:
            root.update()
            return root.clipboard_get()
        finally:
            root.destroy()
    except Exception:
        return ""


def _win32_copy_to_clipboard(text: str) -> bool:
    """Прямое копирование текста в буфер обмена Windows через Win32 Unicode API."""
    try:
        import ctypes
        from ctypes import wintypes

        ctypes_mod: Any = ctypes
        user32 = ctypes_mod.windll.user32
        kernel32 = ctypes_mod.windll.kernel32

        CF_UNICODETEXT = 13
        GMEM_MOVEABLE = 0x0002

        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalUnlock.restype = wintypes.BOOL

        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        user32.EmptyClipboard.argtypes = []
        user32.EmptyClipboard.restype = wintypes.BOOL
        user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        user32.SetClipboardData.restype = wintypes.HANDLE
        user32.CloseClipboard.argtypes = []
        user32.CloseClipboard.restype = wintypes.BOOL

        if not user32.OpenClipboard(None):
            return False
        try:
            user32.EmptyClipboard()
            encoded = (text + "\0").encode("utf-16le")
            handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(encoded))
            if not handle:
                return False
            ptr = kernel32.GlobalLock(handle)
            if not ptr:
                return False
            ctypes_mod.memmove(ptr, encoded, len(encoded))
            kernel32.GlobalUnlock(handle)
            return bool(user32.SetClipboardData(CF_UNICODETEXT, handle))
        finally:
            user32.CloseClipboard()
    except Exception:
        return False


def copy_to_clipboard(text: str) -> None:
    """Скопировать текст в буфер обмена кроссплатформенно (Windows, macOS, Linux)."""
    if sys.platform == "win32":
        if _win32_copy_to_clipboard(text):
            return
        try:
            subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    "[Console]::InputEncoding = [System.Text.Encoding]::UTF8; Set-Clipboard",
                ],
                input=text.encode("utf-8"),
                check=True,
                timeout=5.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            subprocess.run(
                ["pbcopy"],
                input=text,
                text=True,
                check=True,
                timeout=2.0,
            )
            return
        except Exception:
            pass
    else:
        for cmd in (
            ["wl-copy"],
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        ):
            try:
                subprocess.run(
                    cmd,
                    input=text,
                    text=True,
                    check=True,
                    timeout=2.0,
                )
                return
            except Exception:
                continue

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
