"""Инструмент и парсер запросов даты и времени."""
import json
import re
from datetime import datetime, timedelta
from typing import Any

WEEKDAYS_RU = [
    "понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье",
]
MONTHS_RU = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]


def _plural_ru(value: int, one: str, few: str, many: str) -> str:
    value = abs(int(value))
    if value % 10 == 1 and value % 100 != 11:
        return one
    if value % 10 in {2, 3, 4} and value % 100 not in {12, 13, 14}:
        return few
    return many


def _format_datetime_ru(value: datetime) -> str:
    return (
        f"{WEEKDAYS_RU[value.weekday()]}, {value.day} "
        f"{MONTHS_RU[value.month - 1]} {value.year} года, "
        f"{value:%H:%M:%S}"
    )


def _format_offset(
    delta: timedelta | None = None,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
) -> str:
    if delta is None:
        delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    total_seconds = int(abs(delta.total_seconds()))
    d = total_seconds // 86400
    rem = total_seconds % 86400
    h = rem // 3600
    rem = rem % 3600
    m = rem // 60
    s = rem % 60
    parts = []
    if d:
        parts.append(f"{d} {_plural_ru(d, 'день', 'дня', 'дней')}")
    if h:
        parts.append(f"{h} {_plural_ru(h, 'час', 'часа', 'часов')}")
    if m:
        parts.append(f"{m} {_plural_ru(m, 'минута', 'минуты', 'минут')}")
    if s:
        parts.append(f"{s} {_plural_ru(s, 'секунда', 'секунды', 'секунд')}")
    return " ".join(parts) or "0 секунд"


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
    try:
        days = int(float(days or 0))
        hours = int(float(hours or 0))
        minutes = int(float(minutes or 0))
        seconds = int(float(seconds or 0))
    except (ValueError, TypeError, OverflowError):
        return json.dumps(
            {"error": "Некорректные параметры смещения времени."},
            ensure_ascii=False,
        )
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
            raw_iso = str(base_datetime).strip()
            if raw_iso.endswith(("Z", "z")):
                raw_iso = raw_iso[:-1] + "+00:00"
            now = datetime.fromisoformat(raw_iso)
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
    try:
        result = now + delta
    except OverflowError:
        return json.dumps({"error": "Смещение даты слишком велико."}, ensure_ascii=False)
    offset = now.strftime("%z")
    offset = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset

    current_text = _format_datetime_ru(now)
    base_label = "Исходный момент" if base_datetime else "Сейчас"
    if delta == timedelta(0):
        answer = f"{base_label}: {current_text}."
    else:
        direction = "назад" if delta.total_seconds() < 0 else "вперёд"
        offset_text = _format_offset(delta)
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
    r"(?<![\d:\w])(?P<value>[+-]?\d+)\s*"
    r"(?P<unit>"
    r"д(?:ень|ня|ней)|дн(?:я|ей)?|"
    r"час(?:а|ов)?|ч|"
    r"минут(?:а|ы|у)?|мин|м|"
    r"секунд(?:а|ы|у)?|сек"
    r")(?=\s|$|[,.!?])",
    re.IGNORECASE,
)
_MONTH_NUMBER_RU = {name: index + 1 for index, name in enumerate(MONTHS_RU)}
_INVALID_BASE_DATETIME = "__INVALID_BASE_DATETIME__"


def _extract_base_datetime(text: str) -> str:
    normalized = text.lower().replace("ё", "е")
    time_match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b", normalized)
    month_pattern = "|".join(_MONTH_NUMBER_RU)
    date_match = re.search(
        rf"\b([0-3]?\d)\s+({month_pattern})(?:\s+(\d{{4}}))?\b",
        normalized,
    )
    numeric_date_match = re.search(
        r"\b([0-3]?\d)[./]([01]?\d)[./](\d{4}|\d{2})\b",
        normalized,
    )
    iso_date_match = re.search(
        r"\b(\d{4})-([01]\d)-([0-3]\d)\b",
        normalized,
    )
    has_any_date = bool(date_match or numeric_date_match or iso_date_match)
    if not time_match and not has_any_date:
        return ""
    current = datetime.now().astimezone().replace(microsecond=0)
    if date_match:
        day = int(date_match.group(1))
        month = _MONTH_NUMBER_RU[date_match.group(2)]
        year = int(date_match.group(3)) if date_match.group(3) else current.year
    elif numeric_date_match:
        day = int(numeric_date_match.group(1))
        month = int(numeric_date_match.group(2))
        raw_year = int(numeric_date_match.group(3))
        year = raw_year + 2000 if raw_year < 100 else raw_year
    elif iso_date_match:
        year = int(iso_date_match.group(1))
        month = int(iso_date_match.group(2))
        day = int(iso_date_match.group(3))
    else:
        day = current.day
        month = current.month
        year = current.year
    hour = int(time_match.group(1)) if time_match else 0
    minute = int(time_match.group(2)) if time_match else 0
    second = int(time_match.group(3) or 0) if time_match else 0
    try:
        base = datetime(year, month, day, hour, minute, second, tzinfo=current.tzinfo)
    except ValueError:
        return _INVALID_BASE_DATETIME
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


def _is_pure_datetime_request(normalized: str) -> bool:
    words = re.findall(r"[a-zа-я]+", normalized)
    return bool(words) and all(word in _DIRECT_TIME_WORDS for word in words)


def parse_datetime_request(text: str, require_pure: bool = True) -> dict[str, Any] | None:
    """Распознать запрос даты/времени или смещения из текста пользователя."""
    normalized = " ".join(text.lower().replace("ё", "е").split())
    if len(normalized) > 240:
        return None
    if require_pure and not _is_pure_datetime_request(normalized):
        return None

    matches = list(_TIME_VALUE_RE.finditer(normalized))
    forward_re = re.compile(r"\b(вперед|через|прибав|добав|плюс)\w*")
    backward_re = re.compile(r"\b(назад|отмот|вычт|минус)\w*")
    forward = bool(forward_re.search(normalized))
    backward = bool(backward_re.search(normalized))

    if matches and (forward or backward or any(
        match.group("value").startswith(("+", "-")) for match in matches
    )):
        default_direction = 1 if forward else -1 if backward else 1
        values: dict[str, Any] = {"days": 0, "hours": 0, "minutes": 0, "seconds": 0}
        for match in matches:
            raw_str = match.group("value")
            raw_value = int(raw_str)
            if raw_str.startswith(("+", "-")):
                value = raw_value
            else:
                prefix = normalized[:match.start()]
                fwd_before = list(forward_re.finditer(prefix))
                bwd_before = list(backward_re.finditer(prefix))
                closest_before: tuple[int, int] | None = None
                if fwd_before or bwd_before:
                    last_fwd_end = fwd_before[-1].end() if fwd_before else -1
                    last_bwd_end = bwd_before[-1].end() if bwd_before else -1
                    if last_fwd_end > last_bwd_end:
                        closest_before = (match.start() - last_fwd_end, 1)
                    else:
                        closest_before = (match.start() - last_bwd_end, -1)

                suffix = normalized[match.end():]
                fwd_after = forward_re.search(suffix)
                bwd_after = backward_re.search(suffix)
                closest_after: tuple[int, int] | None = None
                if fwd_after and bwd_after:
                    if fwd_after.start() < bwd_after.start():
                        closest_after = (fwd_after.start(), 1)
                    else:
                        closest_after = (bwd_after.start(), -1)
                elif fwd_after:
                    closest_after = (fwd_after.start(), 1)
                elif bwd_after:
                    closest_after = (bwd_after.start(), -1)

                if closest_before is not None and closest_after is not None:
                    comp_dir = closest_after[1] if closest_after[0] < closest_before[0] else closest_before[1]
                elif closest_after is not None:
                    comp_dir = closest_after[1]
                elif closest_before is not None:
                    comp_dir = closest_before[1]
                else:
                    comp_dir = default_direction
                value = abs(raw_value) * comp_dir
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
        if base_datetime == _INVALID_BASE_DATETIME:
            return {"error": "Ошибка: указана несуществующая календарная дата."}
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
        result: dict[str, Any] = {"days": 0, "hours": hours, "minutes": 0, "seconds": 0}
        base_datetime = _extract_base_datetime(text)
        if base_datetime == _INVALID_BASE_DATETIME:
            return {"error": "Ошибка: указана несуществующая календарная дата."}
        if base_datetime:
            result["base_datetime"] = base_datetime
        return result

    has_timezone_or_city = bool(
        re.search(r"\b(?:времени|время|час|дата|день)\s+(?:в|во|по|для|на)\s+[a-zа-яё-]{2,}", normalized)
        or re.search(r"\bпо\s+(?:utc|gmt|мск|гринвич\w*)\b", normalized)
    )
    current_time_patterns = [
        r"\bкоторый\s+час\b",
        r"\bсколько\s+(?:сейчас\s+)?времени\b",
        r"\bсколько\s+сейчас\b",
        r"\bкакая\s+(?:сейчас|сегодня)\s+дата\b",
        r"\bкакой\s+сегодня\s+день\b",
        r"\bскажи\s+(?:сегодняшнюю\s+)?дату\b",
        r"\bсегодняшняя\s+дата\b",
    ]
    if not has_timezone_or_city and any(re.search(pattern, normalized) for pattern in current_time_patterns):
        return {"days": 0, "hours": 0, "minutes": 0, "seconds": 0}
    return None


def split_datetime_request(text: str) -> tuple[dict[str, Any] | None, str]:
    """Отделить временной подзапрос от основного вопроса в комбинированном промпте."""
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


def datetime_answer(arguments: dict[str, Any]) -> str:
    """Преобразовать разобранные аргументы времени в готовый текстовый ответ."""
    if "error" in arguments:
        return str(arguments["error"])
    raw_result = calculate_datetime(**arguments)
    try:
        payload = json.loads(raw_result)
    except json.JSONDecodeError:
        return raw_result
    return str(payload.get("answer") or payload.get("error") or raw_result)
