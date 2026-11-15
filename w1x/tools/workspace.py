"""Безопасное чтение файлов в рамках выбранной папки проекта."""
import os
from pathlib import Path

from .. import config

_EXCLUDED_WORKSPACE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
    ".tox",
}


def set_file_workspace(path_text: str) -> Path:
    """Установить корневую директорию workspace в config."""
    path = Path(path_text).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError("Указанный путь не является папкой.")
    config.ACTIVE_FILE_ROOT = path
    return path


def _safe_workspace_path(relative_path: str) -> Path:
    root = config.ACTIVE_FILE_ROOT
    if root is None:
        raise ValueError("Папка проекта не выбрана. Используй /workspace ПУТЬ.")
    raw_path = Path(str(relative_path or "."))
    candidate = (
        raw_path.resolve(strict=False)
        if raw_path.is_absolute()
        else (root / raw_path).resolve(strict=False)
    )
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("Путь выходит за пределы выбранной папки проекта.") from error
    return candidate


def list_project_files(relative_path: str = ".") -> str:
    """Вывести список файлов внутри выбранного workspace (до 200 записей)."""
    root = config.ACTIVE_FILE_ROOT
    try:
        directory = _safe_workspace_path(relative_path)
    except Exception as error:
        return f"Ошибка доступа: {error}"
    if root is None:
        return "Ошибка доступа: Папка проекта не выбрана. Используй /workspace ПУТЬ."
    if not directory.exists() or not directory.is_dir():
        return "Ошибка: папка не существует."
    files = []
    for root_dir, dirnames, filenames in os.walk(str(directory)):
        dirnames[:] = [
            d for d in sorted(dirnames)
            if d not in _EXCLUDED_WORKSPACE_DIRS and not d.startswith(".")
        ]
        dir_path = Path(root_dir)
        for fname in sorted(filenames):
            file_path = dir_path / fname
            try:
                resolved = file_path.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                files.append(str(resolved.relative_to(root)))
            if len(files) >= 200:
                break
        if len(files) >= 200:
            break
    if not files:
        return "В выбранной папке файлов не найдено."
    suffix = "\n… список ограничен 200 файлами" if len(files) == 200 else ""
    return "Файлы проекта:\n" + "\n".join(f"- {item}" for item in files) + suffix


def read_project_file(relative_path: str) -> str:
    """Прочитать текстовый файл из workspace (до 2 МБ и до 16000 символов)."""
    root = config.ACTIVE_FILE_ROOT
    try:
        path = _safe_workspace_path(relative_path)
    except Exception as error:
        return f"Ошибка доступа: {error}"
    if root is None:
        return "Ошибка доступа: Папка проекта не выбрана. Используй /workspace ПУТЬ."
    if not path.exists() or not path.is_file():
        return "Ошибка: файл не существует."
    try:
        if path.stat().st_size > 2_000_000:
            return "Ошибка: файл больше 2 МБ и не будет прочитан."
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return f"Ошибка чтения: {error}"
    if "\x00" in text[:2000]:
        return "Ошибка: файл похож на бинарный."
    if len(text) > 16000:
        text = text[:16000] + "\n\n… файл обрезан до 16000 символов"
    return f"Файл: {path.relative_to(root)}\n\n{text}"


def project_files(action: str, path: str = ".") -> str:
    """Просмотреть файлы проекта: действие list или read."""
    action_clean = str(action or "").strip().casefold()
    if action_clean == "list":
        return list_project_files(path)
    if action_clean == "read":
        return read_project_file(path)
    return "Ошибка: action должен быть list или read."
