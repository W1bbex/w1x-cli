"""Долговременная память фактов W1x."""
import json
import re
import secrets
import shutil
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from . import config
from .types import ChatMessage, MemoryFact


def _get_facts_path() -> Path | None:
    return getattr(config, "FACTS_PATH", None)


_MEMORY_STOPWORDS = {
    "а", "в", "вы", "где", "для", "его", "ее", "и", "как", "какая",
    "какой", "кто", "мне", "мой", "моя", "мое", "моё", "на", "называется",
    "о", "он", "она", "по", "про", "с", "скажи", "ты", "у", "что", "это", "я",
    "ли", "же", "бы", "хоть", "был", "была", "было", "звали", "зовут",
    "пожалуйста", "пж", "плиз", "удали", "забудь", "стереть",
}

_GENERAL_MEMORY_PATTERNS = (
    r"обо?\s+мне",
    r"что\s+(?:ты\s+)?(?:знаешь|помнишь)",
    r"покажи\s+память",
    r"все\s+факты",
    r"список\s+фактов",
    r"что\s+в\s+памяти",
)


def load_facts_with_status() -> tuple[list[MemoryFact], bool]:
    """Загрузить отдельную постоянную память фактов с признаком успешности чтения."""
    facts_p = _get_facts_path()
    if not facts_p:
        return [], True
    try:
        if not facts_p.exists():
            return [], True
        payload = json.loads(facts_p.read_text(encoding="utf-8"))
    except (ValueError, TypeError):
        try:
            corrupt_p = facts_p.with_name(f"{facts_p.stem}.corrupted.{int(time.time())}.json")
            shutil.move(facts_p, corrupt_p)
        except Exception:
            try:
                facts_p.unlink(missing_ok=True)
            except Exception:
                pass
        return [], True
    except OSError:
        return [], False
    if not isinstance(payload, list):
        try:
            corrupt_p = facts_p.with_name(f"{facts_p.stem}.corrupted.{int(time.time())}.json")
            shutil.move(facts_p, corrupt_p)
        except Exception:
            try:
                facts_p.unlink(missing_ok=True)
            except Exception:
                pass
        return [], True
    return [
        item  # type: ignore[misc]
        for item in payload
        if isinstance(item, dict)
        and isinstance(item.get("fact"), str)
        and item["fact"].strip()
    ], True


def load_facts() -> list[MemoryFact]:
    """Загрузить отдельную постоянную память фактов."""
    facts, _ = load_facts_with_status()
    return facts


def save_facts(facts: Sequence[MemoryFact]) -> bool:
    facts_p = _get_facts_path()
    if not facts_p:
        return False
    temporary_path = facts_p.with_suffix(".tmp")
    try:
        facts_p.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(facts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(facts_p)
        return True
    except OSError:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass
        return False


def remember_fact(fact: str) -> str:
    """Запомнить пользовательский факт между диалогами.

    Args:
        fact: Короткий факт, который пользователь явно попросил запомнить.

    Returns:
        Подтверждение сохранения факта.
    """
    fact = " ".join(str("" if fact is None else fact).split()).strip(" .")
    if not fact:
        return "Ошибка: нечего запоминать."
    if len(fact) > 500:
        return "Ошибка: один факт не должен быть длиннее 500 символов."

    facts, ok = load_facts_with_status()
    if not ok:
        return "Ошибка: не удалось прочитать существующую память (файл заблокирован)."
    normalized = fact.casefold()
    for item in facts:
        if item["fact"].casefold() == normalized:
            return f"Уже помню: {item['fact']}."
    facts.append(
        {
            "id": secrets.token_hex(4),
            "fact": fact,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    if not save_facts(facts[-200:]):
        return "Ошибка: не удалось сохранить память."
    return f"Запомнил: {fact}."


def _stem_word(word: str) -> str:
    """Удалить падежные окончания для точного поиска в русском языке."""
    w = word.casefold()
    if len(w) <= 3:
        return w
    for suffix in (
        "ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими", "ах", "ях",
        "ов", "ев", "ей", "ом", "ем", "ам", "ям", "ой", "ий", "ый",
        "ая", "яя", "ое", "ее", "ые", "ие", "ут", "ют", "ат", "ят", "ил",
        "ила", "или", "ит", "ет", "у", "ю", "а", "я", "е", "ы", "и", "о",
    ):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


def _memory_words(text: str) -> set[str]:
    words = set()
    safe_text = "" if text is None else str(text)
    normalized = safe_text.replace("-", " ")
    for raw in re.findall(r"[a-zа-яё_]{2,}|\d+", normalized.casefold()):
        if raw not in _MEMORY_STOPWORDS:
            stem = _stem_word(raw)
            if stem:
                words.add(stem)
    return words


def _is_general_query(query: str) -> bool:
    q = ("" if query is None else str(query)).casefold().strip()
    if not q:
        return True
    return any(re.search(pat, q) for pat in _GENERAL_MEMORY_PATTERNS)


def _rank_facts(facts: Sequence[MemoryFact], safe_query: str) -> list[str]:
    """Ранжировать факты по релевантности запросу."""
    if not facts:
        return []
    if _is_general_query(safe_query):
        return [item["fact"] for item in facts[-20:]]

    query_words = _memory_words(safe_query)
    ranked = []
    for index, item in enumerate(facts):
        fact_words = _memory_words(item["fact"])
        overlap = len(query_words & fact_words)
        if not overlap:
            for qw in query_words:
                if len(qw) >= 3 and any(qw in fw or fw in qw for fw in fact_words):
                    overlap += 1
        if overlap:
            ranked.append((overlap, index, item["fact"]))

    if not ranked:
        if len(facts) <= 5:
            return [item["fact"] for item in facts]
        return []

    ranked.sort(reverse=True)
    return [fact for _, _, fact in ranked[:8]]


def recall_facts(query: str = "") -> str:
    """Найти факты в постоянной памяти W1x.

    Args:
        query: Что именно нужно вспомнить. Пустая строка показывает все сохранённые факты.

    Returns:
        Найденные факты или сообщение, что совпадений нет.
    """
    safe_query = "" if query is None else str(query)
    facts, ok = load_facts_with_status()
    if not ok:
        return "Ошибка: не удалось прочитать файл постоянной памяти с диска."
    if not facts:
        return "Постоянная память фактов пуста."

    matched = _rank_facts(facts, safe_query)
    if not matched:
        return "В постоянной памяти нет подходящего факта."

    return "Факты из постоянной памяти:\n" + "\n".join(
        f"- {fact}" for fact in matched
    )


def forget_facts(query: str) -> str:
    """Удалить подходящие факты из постоянной памяти.

    Args:
        query: Слова из факта, который пользователь попросил забыть.

    Returns:
        Количество удалённых фактов.
    """
    query_words = _memory_words(query)
    if not query_words:
        return "Ошибка: уточни, какой факт забыть."
    facts, ok = load_facts_with_status()
    if not ok:
        return "Ошибка: не удалось прочитать память (файл заблокирован)."
    kept = []
    removed = []
    for item in facts:
        fact_words = _memory_words(item["fact"])
        match = all(
            qw in fact_words or (len(qw) >= 3 and any(qw in fw or fw in qw for fw in fact_words))
            for qw in query_words
        )
        if match:
            removed.append(item["fact"])
        else:
            kept.append(item)
    if not removed:
        return "Подходящий факт в памяти не найден."
    if not save_facts(kept):
        return "Ошибка: не удалось обновить память."
    return "Забыл: " + "; ".join(removed) + "."


def memory(action: str, text: str = "") -> str:
    """Управление постоянной памятью фактов О ПОЛЬЗОВАТЕЛЕ (имя, питомцы, предпочтения).
    НЕ вызывай этот инструмент для общих вопросов о мире, географии, физике, истории или воде в море.

    Args:
        action: remember — сохранить факт о пользователе, recall — вспомнить факт о пользователе, forget — забыть факт.
        text: Личный факт о пользователе (для remember) или слова поиска (для recall/forget).

    Returns:
        Результат операции с постоянной памятью пользователя.
    """
    normalized_action = ("" if action is None else str(action)).strip().casefold()
    safe_text = "" if text is None else str(text)
    if normalized_action == "remember":
        return remember_fact(safe_text)
    if normalized_action == "recall":
        return recall_facts(safe_text)
    if normalized_action == "forget":
        return forget_facts(safe_text)
    return "Ошибка: action должен быть remember, recall или forget."


def relevant_memory_block(query: str = "") -> str:
    facts = load_facts()
    if not facts:
        return ""
    # Если сохранено до 30 фактов — подмешиваем их ВСЕ в контекст системного промпта!
    if len(facts) <= 30:
        return "Факты из постоянной памяти о пользователе:\n" + "\n".join(
            f"- {item['fact']}" for item in facts
        )
    safe_query = "" if query is None else str(query)
    matched = _rank_facts(facts, safe_query)
    if not matched:
        return ""
    return "Факты из постоянной памяти о пользователе:\n" + "\n".join(
        f"- {fact}" for fact in matched
    )


def extract_remember_request(text: str) -> str:
    match = re.match(
        r"^\s*(?:запомни|запиши\s+в\s+память|сохрани\s+в\s+память)"
        r"\s*[:,-]?\s*(.+?)\s*$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def extract_forget_request(text: str) -> str:
    match = re.match(
        r"^\s*(?:забудь|удали\s+из\s+памяти)\s*[:,-]?\s*(.+?)\s*$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def looks_like_memory_question(text: str) -> bool:
    normalized = text.casefold()
    return bool(
        re.search(r"\b(?:помнишь|вспомни|забыл\w*)\b", normalized)
        or re.search(r"\bкак\s+называется\s+(?:мой|моя|мое|моё)\b", normalized)
        or re.search(r"\bчто\s+ты\s+(?:знаешь|помнишь)\s+обо?\s+мне\b", normalized)
    )


def is_memory_migrated() -> bool:
    cfg = config.load_user_config()
    return bool(cfg.get("memory_migrated", False))


def set_memory_migrated() -> None:
    cfg = dict(config.load_user_config())
    cfg["memory_migrated"] = True
    config.save_user_config(cfg)


def migrate_explicit_memory(messages: Sequence[ChatMessage]) -> int:
    """Один раз подобрать старые фразы «запомни» из сохранённого диалога."""
    if is_memory_migrated():
        return 0
    facts, ok = load_facts_with_status()
    if not ok:
        return 0
    existing = {item["fact"].casefold() for item in facts}
    migrated = 0
    for message in messages:
        if message.get("role") != "user":
            continue
        fact = extract_remember_request(message.get("content", "") or "")
        cleaned = " ".join(fact.split()).strip(" .")
        if not cleaned or len(cleaned) > 500:
            continue
        normalized = cleaned.casefold()
        if normalized in existing:
            continue
        existing.add(normalized)
        facts.append(
            {
                "id": secrets.token_hex(4),
                "fact": cleaned,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )
        migrated += 1
    if migrated:
        if not save_facts(facts[-200:]):
            return 0
    set_memory_migrated()
    return migrated
