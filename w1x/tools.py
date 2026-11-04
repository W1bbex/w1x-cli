# -*- coding: utf-8 -*-
"""Система встроенных инструментов W1x (поиск, время, арифметика, файлы, Antigravity, Python)."""
import ast
import json
import os
import operator
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

from .config import (
    ACTIVE_FILE_ROOT,
    ACTIVE_AGY_PROCESS,
    ACTIVE_AGY_LOCK,
    AGY_TIMEOUT_SECONDS,
    AGY_MAX_PROMPT_CHARS,
    AGY_MAX_OUTPUT_CHARS,
)
from .terminal import (
    console,
    session,
    render_message,
    run_with_spinner,
    HTML,
)
from .memory import memory


def search_web(query: str) -> str:
    """Найти свежую информацию в интернете.

    Args:
        query: Короткий и точный поисковый запрос.

    Returns:
        Заголовки, фрагменты текста и адреса найденных страниц.
    """
    query = query.strip()
    if not query:
        return "Ошибка: пустой поисковый запрос."

    has_cyrillic = any("а" <= char.lower() <= "я" or char.lower() == "ё" for char in query)
    regions = ["ru-ru", "us-en"] if has_cyrillic else ["us-en", "ru-ru"]
    results = []
    seen_urls = set()
    search = DDGS(timeout=12)
    for region in regions:
        try:
            region_results = search.text(
                query,
                region=region,
                safesearch="off",
                max_results=5,
            )
        except Exception:
            continue
        for item in region_results:
            url = str(item.get("href") or item.get("url") or "").strip()
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append(item)
            if len(results) >= 3:
                break
        if len(results) >= 3:
            break
    if not results:
        return "Поиск не вернул результатов."

    # Выводим красивое дерево найденных сайтов и сниппетов в консоль пользователя
    try:
        tree = Tree(f"[bold cyan]🔍 Поиск в интернете:[/bold cyan] [white]{query}[/white]")
        for item in results:
            url = str(item.get("href") or item.get("url") or "").strip()
            domain = urlparse(url).netloc.replace("www.", "") or url
            title = str(item.get("title") or domain).strip()
            snippet = " ".join(str(item.get("body") or "").split()).strip()
            if len(snippet) > 130:
                snippet = snippet[:130] + "…"

            branch = tree.add(f"[bold green]🌐 {domain}[/bold green] — [dim]{title}[/dim]")
            if snippet:
                branch.add(f"[dim italic]«{snippet}»[/dim italic]")

        console.print()
        console.print(tree)
        console.print()
    except Exception:
        pass

    lines = []
    for index, item in enumerate(results, 1):
        title = str(item.get("title") or "Без названия").strip()
        url = str(item.get("href") or item.get("url") or "").strip()
        snippet = " ".join(str(item.get("body") or "").split())[:300]
        lines.append(f"{index}. {title}\nURL: {url}\n{snippet}")
    return "\n\n".join(lines)


WEEKDAYS_RU = [
    "понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье",
]
MONTHS_RU = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]


def _plural_ru(value, one, few, many):
    value = abs(int(value))
    if value % 10 == 1 and value % 100 != 11:
        return one
    if value % 10 in {2, 3, 4} and value % 100 not in {12, 13, 14}:
        return few
    return many


def _format_datetime_ru(value):
    return (
        f"{WEEKDAYS_RU[value.weekday()]}, {value.day} "
        f"{MONTHS_RU[value.month - 1]} {value.year} года, "
        f"{value:%H:%M:%S}"
    )


def _format_offset(days, hours, minutes, seconds):
    parts = []
    if days:
        parts.append(f"{abs(days)} {_plural_ru(days, 'день', 'дня', 'дней')}")
    if hours:
        parts.append(f"{abs(hours)} {_plural_ru(hours, 'час', 'часа', 'часов')}")
    if minutes:
        parts.append(f"{abs(minutes)} {_plural_ru(minutes, 'минута', 'минуты', 'минут')}")
    if seconds:
        parts.append(f"{abs(seconds)} {_plural_ru(seconds, 'секунда', 'секунды', 'секунд')}")
    return " ".join(parts) or "0 минут"


def calculate_datetime(
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    base_datetime: str = "",
) -> str:
    """Точно получить текущую дату/время или вычислить смещение.

    Args:
        days: Количество дней вперёд (+) или назад (-).
        hours: Количество часов вперёд (+) или назад (-).
        minutes: Количество минут вперёд (+) или назад (-).
        seconds: Количество секунд вперёд (+) или назад (-).
        base_datetime: Необязательная исходная дата ISO 8601. Если пусто — сейчас.

    Returns:
        JSON с готовым полем answer, которое нельзя пересчитывать или изменять.
    """
    days = int(days)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    if (
        abs(days) > 365000
        or abs(hours) > 1_000_000
        or abs(minutes) > 10_000_000
        or abs(seconds) > 600_000_000
    ):
        return json.dumps(
            {"error": "Слишком большое смещение даты или времени."},
            ensure_ascii=False,
        )

    system_now = datetime.now().astimezone().replace(microsecond=0)
    if base_datetime:
        try:
            now = datetime.fromisoformat(str(base_datetime).strip())
            if now.tzinfo is None:
                now = now.replace(tzinfo=system_now.tzinfo)
            now = now.replace(microsecond=0)
        except ValueError:
            return json.dumps(
                {"error": "Неправильная исходная дата. Используй ISO 8601."},
                ensure_ascii=False,
            )
    else:
        now = system_now
    delta = timedelta(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
    )
    result = now + delta
    offset = now.strftime("%z")
    offset = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset

    current_text = _format_datetime_ru(now)
    base_label = "Исходный момент" if base_datetime else "Сейчас"
    if delta == timedelta(0):
        answer = f"{base_label}: {current_text}."
    else:
        direction = "назад" if delta.total_seconds() < 0 else "вперёд"
        offset_text = _format_offset(days, hours, minutes, seconds)
        answer = (
            f"{base_label}: {current_text}.\n"
            f"{offset_text.capitalize()} {direction}: "
            f"{_format_datetime_ru(result)}."
        )

    return json.dumps(
        {
            "current": now.isoformat(),
            "offset": {
                "days": days,
                "hours": hours,
                "minutes": minutes,
                "seconds": seconds,
            },
            "result": result.isoformat(),
            "timezone": "локальный часовой пояс системы",
            "utc_offset": offset,
            "answer": answer,
            "instruction": "Вернуть поле answer дословно без пересчёта.",
        },
        ensure_ascii=False,
    )


_TIME_VALUE_RE = re.compile(
    r"(?P<value>[+-]?\d+)\s*"
    r"(?P<unit>"
    r"д(?:ень|ня|ней)|дн(?:я|ей)?|"
    r"час(?:а|ов)?|ч|"
    r"минут(?:а|ы|у)?|мин|м|"
    r"секунд(?:а|ы|у)?|сек"
    r")(?=\s|$|[,.!?])",
    re.IGNORECASE,
)
_MONTH_NUMBER_RU = {name: index + 1 for index, name in enumerate(MONTHS_RU)}


def _extract_base_datetime(text):
    normalized = text.lower().replace("ё", "е")
    time_match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b", normalized)
    month_pattern = "|".join(_MONTH_NUMBER_RU)
    date_match = re.search(
        rf"\b([0-3]?\d)\s+({month_pattern})(?:\s+(20\d{{2}}))?\b",
        normalized,
    )
    if not time_match and not date_match:
        return ""
    if not date_match and not re.search(r"\b(?:если\s+)?сейчас\b", normalized):
        return ""
    current = datetime.now().astimezone().replace(microsecond=0)
    day = int(date_match.group(1)) if date_match else current.day
    month = _MONTH_NUMBER_RU[date_match.group(2)] if date_match else current.month
    year = int(date_match.group(3)) if date_match and date_match.group(3) else current.year
    hour = int(time_match.group(1)) if time_match else 0
    minute = int(time_match.group(2)) if time_match else 0
    second = int(time_match.group(3) or 0) if time_match else 0
    try:
        base = datetime(year, month, day, hour, minute, second, tzinfo=current.tzinfo)
    except ValueError:
        return ""
    return base.isoformat()


_DIRECT_TIME_WORDS = {
    *_MONTH_NUMBER_RU.keys(),
    "а", "будет", "будут", "будущий", "бы", "был", "была", "было", "были",
    "в", "вперед", "времени", "время", "вычти", "дата", "дату", "день",
    "дней", "дня", "добавь", "если", "и", "какая", "какой", "какое", "какие",
    "который", "год", "месяц", "м", "мин", "минус", "минута", "минут",
    "минуты", "минуту", "на", "назад", "недели", "неделя", "от", "отмотай",
    "отмотать", "плюс", "пожалуйста", "покажи", "посчитай", "прибавь",
    "привет", "скажи", "сколько", "сейчас", "сек", "секунда", "секунд",
    "секунды", "секунду", "сегодня", "тому", "теперь", "точно", "точное",
    "точный", "ровно", "ответь", "назови", "ч", "час", "часа", "часов",
    "через", "число", "что",
}


def _is_pure_datetime_request(normalized):
    words = re.findall(r"[a-zа-я]+", normalized)
    return bool(words) and all(word in _DIRECT_TIME_WORDS for word in words)


def parse_datetime_request(text, require_pure=True):
    normalized = " ".join(text.lower().replace("ё", "е").split())
    if len(normalized) > 240:
        return None
    if require_pure and not _is_pure_datetime_request(normalized):
        return None

    matches = list(_TIME_VALUE_RE.finditer(normalized))
    forward = bool(re.search(r"\b(вперед|через|прибав|добав|плюс)\w*", normalized))
    backward = bool(re.search(r"\b(назад|отмот|вычт|минус)\w*", normalized))

    if matches and (forward or backward or any(
        match.group("value").startswith(("+", "-")) for match in matches
    )):
        direction = 1 if forward else -1 if backward else 1
        values = {"days": 0, "hours": 0, "minutes": 0, "seconds": 0}
        for match in matches:
            raw_value = int(match.group("value"))
            value = abs(raw_value) * direction if (forward or backward) else raw_value
            unit = match.group("unit")
            if unit.startswith("д"):
                values["days"] += value
            elif unit.startswith("ч") or unit.startswith("час"):
                values["hours"] += value
            elif unit.startswith("м"):
                values["minutes"] += value
            else:
                values["seconds"] += value
        base_datetime = _extract_base_datetime(text)
        if base_datetime:
            values["base_datetime"] = base_datetime
        return values

    bare_offset = re.fullmatch(
        r"(?:а\s+)?(?:если\s+)?(\d+)\s*"
        r"(назад|вперед)[?!.]*",
        normalized,
    )
    if bare_offset:
        hours = int(bare_offset.group(1))
        if bare_offset.group(2) == "назад":
            hours = -hours
        result = {"days": 0, "hours": hours, "minutes": 0, "seconds": 0}
        base_datetime = _extract_base_datetime(text)
        if base_datetime:
            result["base_datetime"] = base_datetime
        return result

    current_time_patterns = [
        r"\bкоторый\s+час\b",
        r"\bсколько\s+(?:сейчас\s+)?времени\b",
        r"\bсколько\s+сейчас\b",
        r"\bкакая\s+(?:сейчас|сегодня)\s+дата\b",
        r"\bкакой\s+сегодня\s+день\b",
        r"\bскажи\s+(?:сегодняшнюю\s+)?дату\b",
        r"\bсегодняшняя\s+дата\b",
    ]
    if any(re.search(pattern, normalized) for pattern in current_time_patterns):
        return {"days": 0, "hours": 0, "minutes": 0, "seconds": 0}
    return None


def split_datetime_request(text):
    arguments = parse_datetime_request(text, require_pure=False)
    if arguments is None:
        return None, text
    if parse_datetime_request(text, require_pure=True) is not None:
        return arguments, ""

    remaining = []
    removed_time_clause = False
    previous_was_time = False
    for piece in re.findall(r"[^?!]+[?!]*", text):
        stripped = piece.strip()
        if not stripped:
            continue
        piece_arguments = parse_datetime_request(stripped, require_pure=True)
        normalized_piece = " ".join(
            stripped.lower().replace("ё", "е").split()
        )
        is_time_format_tail = (
            previous_was_time
            and _is_pure_datetime_request(normalized_piece)
        )
        if piece_arguments is not None or is_time_format_tail:
            removed_time_clause = True
            previous_was_time = True
            continue
        remaining.append(stripped)
        previous_was_time = False

    if not removed_time_clause:
        cleanup_patterns = [
            r"\bкакая\s+сегодня\s+дата\b",
            r"\bкакое\s+сейчас\s+время\b",
            r"\bсколько\s+сейчас\s+времени\b",
            r"\bскажи\s+(?:сегодняшнюю\s+)?дату\b",
        ]
        cleaned = text
        for pattern in cleanup_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*[,;:-]?\s*(?:и\s+)?", "", cleaned).strip()
        if cleaned != text.strip():
            return arguments, cleaned
        return None, text
    return arguments, " ".join(remaining).strip()


def datetime_answer(arguments):
    raw_result = calculate_datetime(**arguments)
    try:
        payload = json.loads(raw_result)
    except json.JSONDecodeError:
        return raw_result
    return payload.get("answer") or payload.get("error") or raw_result


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_math_node(node):
    if isinstance(node, ast.Expression):
        return _evaluate_math_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_math_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_math_node(node.left)
        right = _evaluate_math_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Слишком большая степень")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    raise ValueError("Разрешены только числа, скобки и операции + - * / // % **")


def calculate_expression(expression: str) -> str:
    """Точно вычислить арифметическое выражение без приблизительного счёта.

    Args:
        expression: Выражение с числами, скобками и операциями + - * / // % **.

    Returns:
        Точный результат вычисления или описание ошибки.
    """
    expression = expression.strip()
    if not expression or len(expression) > 300:
        return "Ошибка: выражение пустое или слишком длинное."
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_evaluate_math_node(tree))
    except Exception as error:
        return f"Ошибка вычисления: {type(error).__name__}: {error}"


_MATH_ALLOWED_WORDS = {
    "а", "будет", "вычисли", "вычислить", "калькулятором", "математикой",
    "пожалуйста", "посчитай", "проверь", "результат", "сколько", "точно",
    "чему", "равно", "равен", "теперь",
}


def parse_math_request(text):
    normalized = text.lower().replace("×", "*").replace("÷", "/")
    if re.search(r"не\s+(?:используй|вызывай).*калькулятор", normalized):
        return None
    candidates = re.findall(r"(?<![\w.])[\d\s().,+\-*/%]{3,}(?![\w.])", normalized)
    candidates = [candidate.strip(" ,.") for candidate in candidates]
    candidates = [
        candidate
        for candidate in candidates
        if re.search(r"[+\-*/%]", candidate) and re.search(r"\d", candidate)
    ]
    if not candidates:
        return None
    expression = max(candidates, key=len).replace(",", ".").strip()
    remaining = normalized.replace(max(candidates, key=len), " ")
    words = set(re.findall(r"[a-zа-яё]+", remaining))
    if words and not words <= _MATH_ALLOWED_WORDS:
        return None
    return expression


def previous_math_expression(messages):
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        expression = parse_math_request(message.get("content", ""))
        if expression:
            return expression
    return None


def needs_clarification(text):
    normalized = " ".join(text.casefold().strip(" .!?,-").split())
    vague_patterns = [
        r"(?:сделай|повтори|реши|запусти|удали)(?:\s+(?:это|там|так))?(?:\s+как\s+(?:вчера|раньше))?",
        r"(?:я\s+же\s+)?(?:сказал|писал)\s+уже",
        r"сделай\s+как\s+(?:вчера|раньше)",
    ]
    return len(normalized) < 80 and any(
        re.fullmatch(pattern, normalized) for pattern in vague_patterns
    )


def run_python_code(code: str, purpose: str, inputs: str = "") -> str:
    """Проверить небольшой фрагмент Python-кода на компьютере пользователя.

    Args:
        code: Полный Python-код для проверки.
        purpose: Коротко: что проверяет код и зачем это нужно для ответа.
        inputs: Необязательно: текстовые строки для input(), через \n (например: "2\n*\n5\n").

    Returns:
        Код завершения, stdout и stderr программы.
    """
    if not code.strip():
        return "Ошибка: модель передала пустой код."
    if len(code) > 8000:
        return "Ошибка: код длиннее 8000 символов и не будет запущен."

    purpose = " ".join(str(purpose or "").split())
    if not purpose:
        return "Ошибка: модель не объяснила, зачем ей запускать Python-код."

    console.print()
    render_message("Зачем нужен Python ›", role="assistant_label")
    render_message(purpose[:600], role="assistant")
    if inputs:
        formatted_inputs = ", ".join(repr(line) for line in str(inputs).splitlines() if line)
        render_message(f"Входные данные для input() › {formatted_inputs}", role="assistant_label")
    console.print()
    console.print(
        Panel(
            Syntax(code, "python", theme="monokai", word_wrap=True, padding=0),
            title="Python просит разрешение на запуск",
            border_style="yellow",
        )
    )
    console.print(
        "[green]Нажми [bold]Enter[/bold] для запуска. Чтобы отклонить — введи [bold]n[/bold] или [bold]нет[/bold].[/green]"
    )
    try:
        confirmation = session.prompt(
            HTML("<b><ansigreen>Разрешить запуск? [Y/n] › </ansigreen></b>"),
            multiline=False,
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        confirmation = "n"
    if confirmation in {"n", "no", "н", "нет", "cancel", "отмена"}:
        return "Пользователь отказался запускать код."

    try:
        with tempfile.TemporaryDirectory(prefix="ollama-python-") as temp_dir:
            input_bytes = (str(inputs) + "\n").encode("utf-8") if inputs else None
            completed = subprocess.run(
                [sys.executable, "-I", "-X", "utf8", "-c", code],
                input=input_bytes,
                cwd=temp_dir,
                capture_output=True,
                timeout=10,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            stdout = completed.stdout.decode("utf-8", errors="replace")[-12000:].rstrip()
            stderr = completed.stderr.decode("utf-8", errors="replace")[-12000:].rstrip()
            return_code = completed.returncode
    except subprocess.TimeoutExpired as error:
        stdout_err = (
            (error.stdout.decode("utf-8", errors="replace") if error.stdout else "")
            + (error.stderr.decode("utf-8", errors="replace") if error.stderr else "")
        )[-12000:]
        return f"Выполнение остановлено через 10 секунд.\n{stdout_err}"
    except Exception as error:
        return f"Ошибка запуска: {type(error).__name__}: {error}"

    parts = [f"Код завершения: {return_code}"]
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    if not stdout and not stderr:
        if return_code == 0:
            parts.append(
                "Программа завершилась успешно (код 0), но в stdout ничего не выведено "
                "(код выполнился без ошибок, но без явного вызова print())."
            )
        else:
            parts.append("Программа завершилась с ошибкой (код != 0) без вывода в stdout.")
    return "\n\n".join(parts)


def set_file_workspace(path_text):
    global ACTIVE_FILE_ROOT
    from . import config
    path = Path(path_text).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError("Указанный путь не является папкой.")
    config.ACTIVE_FILE_ROOT = path
    return path


def _safe_workspace_path(relative_path):
    from .config import ACTIVE_FILE_ROOT
    if ACTIVE_FILE_ROOT is None:
        raise ValueError("Папка проекта не выбрана. Используй /workspace ПУТЬ.")
    raw_path = Path(str(relative_path or "."))
    candidate = (
        raw_path.resolve(strict=False)
        if raw_path.is_absolute()
        else (ACTIVE_FILE_ROOT / raw_path).resolve(strict=False)
    )
    try:
        candidate.relative_to(ACTIVE_FILE_ROOT)
    except ValueError as error:
        raise ValueError("Путь выходит за пределы выбранной папки проекта.") from error
    return candidate


def list_project_files(relative_path: str = ".") -> str:
    from .config import ACTIVE_FILE_ROOT
    try:
        directory = _safe_workspace_path(relative_path)
    except Exception as error:
        return f"Ошибка доступа: {error}"
    if not directory.exists() or not directory.is_dir():
        return "Ошибка: папка не существует."
    files = []
    for path in directory.rglob("*"):
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(ACTIVE_FILE_ROOT)
        except (OSError, ValueError):
            continue
        if path.is_file():
            files.append(str(path.relative_to(ACTIVE_FILE_ROOT)))
        if len(files) >= 200:
            break
    if not files:
        return "В выбранной папке файлов не найдено."
    suffix = "\n… список ограничен 200 файлами" if len(files) == 200 else ""
    return "Файлы проекта:\n" + "\n".join(f"- {item}" for item in files) + suffix


def read_project_file(relative_path: str) -> str:
    from .config import ACTIVE_FILE_ROOT
    try:
        path = _safe_workspace_path(relative_path)
    except Exception as error:
        return f"Ошибка доступа: {error}"
    if not path.exists() or not path.is_file():
        return "Ошибка: файл не существует."
    if path.stat().st_size > 2_000_000:
        return "Ошибка: файл больше 2 МБ и не будет прочитан."
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return f"Ошибка чтения: {error}"
    if "\x00" in text[:2000]:
        return "Ошибка: файл похож на бинарный."
    if len(text) > 16000:
        text = text[:16000] + "\n\n… файл обрезан до 16000 символов"
    return f"Файл: {path.relative_to(ACTIVE_FILE_ROOT)}\n\n{text}"


def project_files(action: str, path: str = ".") -> str:
    action = action.strip().casefold()
    if action == "list":
        return list_project_files(path)
    if action == "read":
        return read_project_file(path)
    return "Ошибка: action должен быть list или read."


_SECRET_PATTERNS = [
    re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password|passwd|authorization)\b"
        r"\s*[:=]\s*([^\s,;]{8,})"
    ),
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+PRIVATE KEY-----"),
]


def _redact_secrets(text):
    redacted = str(text)
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[СЕКРЕТ СКРЫТ]", redacted)
    return redacted


def cancel_active_agy():
    from . import config
    with config.ACTIVE_AGY_LOCK:
        process = config.ACTIVE_AGY_PROCESS
    if process is not None and process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass


def ask_antigravity(task: str, context: str = "") -> str:
    from . import config
    executable = shutil.which("agy")
    if not executable:
        return "ESCALATE_ERROR: команда agy не найдена в PATH."
    task = _redact_secrets(task.strip())
    context = _redact_secrets(context.strip())
    if not task:
        return "ESCALATE_ERROR: задача для Antigravity пуста."
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
    prompt = prompt[:AGY_MAX_PROMPT_CHARS]

    process = None
    allowed_env_names = {
        "APPDATA", "COMSPEC", "LANG", "LC_ALL", "LOCALAPPDATA", "PATH",
        "PATHEXT", "PROGRAMFILES", "PROGRAMFILES(X86)", "SYSTEMDRIVE",
        "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "WINDIR",
    }
    safe_environment = {
        key: value for key, value in os.environ.items() if key.upper() in allowed_env_names
    }
    try:
        with tempfile.TemporaryDirectory(prefix="w1x-agy-") as temp_dir:
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
                config.ACTIVE_AGY_PROCESS = process
            stdout, stderr = process.communicate(timeout=AGY_TIMEOUT_SECONDS)
            return_code = process.returncode
    except subprocess.TimeoutExpired:
        if process is not None:
            process.kill()
            stdout, stderr = process.communicate()
        return f"ESCALATE_ERROR: Antigravity не ответила за {AGY_TIMEOUT_SECONDS} секунд."
    except (OSError, PermissionError) as error:
        return f"ESCALATE_ERROR: не удалось запустить agy: {error}"
    finally:
        with config.ACTIVE_AGY_LOCK:
            if config.ACTIVE_AGY_PROCESS is process:
                config.ACTIVE_AGY_PROCESS = None

    output = (stdout or "").strip()
    error_text = (stderr or "").strip()
    combined_error = f"{output}\n{error_text}".lower()
    if "permission" in combined_error and ("denied" in combined_error or "failed" in combined_error):
        return "ESCALATE_ERROR: Antigravity запросила запрещённое действие; доступ не выдан."
    if return_code != 0:
        details = (error_text or output or "без описания")[-1200:]
        return f"ESCALATE_ERROR: agy завершилась с кодом {return_code}: {details}"
    if not output:
        return "ESCALATE_ERROR: Antigravity не вернула текст."
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    output = ansi_escape.sub("", output)
    if len(output) > AGY_MAX_OUTPUT_CHARS:
        output = output[:AGY_MAX_OUTPUT_CHARS]
        if output.count("```") % 2:
            output += "\n```"
        output += "\n\n> ⚠️ Ответ Antigravity обрезан по лимиту W1x."
    return output


def should_auto_escalate(text):
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


def recent_context_for_agy(messages):
    recent = []
    for message in messages:
        if message.get("role") in {"user", "assistant"}:
            recent.append(f"{message['role']}: {message.get('content', '')}")
    return "\n".join(recent[-3:])[-1200:]


def run_agy_for_user(task, messages):
    render_message("Передаю задачу старшей модели Antigravity", role="tool")
    result = run_with_spinner(
        "Antigravity думает...",
        lambda: ask_antigravity(task, recent_context_for_agy(messages)),
    )
    render_message("Antigravity ›", role="assistant_label")
    render_message(result, role="assistant")
    return result


AVAILABLE_TOOLS = {
    "search_web": search_web,
    "calculate_datetime": calculate_datetime,
    "calculate_expression": calculate_expression,
    "run_python_code": run_python_code,
    "memory": memory,
    "project_files": project_files,
    "ask_antigravity": ask_antigravity,
}
TOOL_FUNCTIONS = list(AVAILABLE_TOOLS.values())

DIRECT_PASSTHROUGH_TOOLS = {
    "calculate_expression",
    "project_files",
    "ask_antigravity",
}


def format_direct_tool_result(name, result):
    if name == "calculate_datetime":
        try:
            payload = json.loads(result)
            return payload.get("answer") or result
        except json.JSONDecodeError:
            return result
    return result


def execute_tool_call(call):
    name = call.function.name
    arguments = call.function.arguments or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return f"Инструмент {name} получил неправильные аргументы: {arguments}"
    if not isinstance(arguments, dict):
        return f"Инструмент {name} получил аргументы неправильного типа."
    function = AVAILABLE_TOOLS.get(name)
    if function is None:
        return f"Неизвестный инструмент: {name}"

    labels = {
        "search_web": f"Ищу в интернете: {arguments.get('query', '')}",
        "calculate_datetime": "Вычисляю точные дату и время",
        "calculate_expression": f"Считаю: {arguments.get('expression', '')}",
        "run_python_code": "Модель хочет проверить Python-код",
        "memory": f"Постоянная память: {arguments.get('action', '')}",
        "project_files": f"Файлы проекта, только чтение: {arguments.get('action', '')}",
        "ask_antigravity": "Передаю сложную задачу Antigravity",
    }
    console.print()
    render_message(labels.get(name, name), role="tool")
    try:
        if name == "ask_antigravity":
            return str(
                run_with_spinner(
                    "Antigravity думает...",
                    lambda: function(**arguments),
                )
            )
        return str(function(**arguments))
    except KeyboardInterrupt:
        return "Вызов инструмента остановлен пользователем."
    except Exception as error:
        return f"Ошибка инструмента {name}: {type(error).__name__}: {error}"
