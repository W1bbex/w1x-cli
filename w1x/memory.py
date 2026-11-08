# -*- coding: utf-8 -*-
"""Долговременная память фактов W1x."""
import json
import re
import secrets
import shutil
import time
from datetime import datetime

from . import config


def _get_facts_path():
    return getattr(config, "FACTS_PATH", None)


def _get_chat_dir():
    return getattr(config, "CHAT_DIR", None)

_MEMORY_STOPWORDS = {
    "а", "в", "вы", "где", "для", "его", "ее", "и", "как", "какая",
    "какой", "кто", "мне", "мой", "моя", "мое", "моё", "на", "называется",
    "о", "он", "она", "по", "про", "с", "скажи", "ты", "у", "что", "это", "я",
    "ли", "же", "бы", "хоть", "был", "была", "было", "звали", "зовут",
}

_GENERAL_MEMORY_PATTERNS = (
    r"обо?\s+мне",
    r"что\s+(?:ты\s+)?(?:знаешь|помнишь)",
    r"покажи\s+память",
    r"все\s+факты",
    r"список\s+фактов",
    r"что\s+в\s+памяти",
)


def load_facts():
    """Загрузить отдельную постоянную память фактов."""
    facts_p = _get_facts_path()
    if not facts_p or not facts_p.exists():
        return []
    try:
        facts_p = _get_facts_path()
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
        return []
    except OSError:
        return []
    if not isinstance(payload, list):
        return []
    return [
        item
        for item in payload
        if isinstance(item, dict)
        and isinstance(item.get("fact"), str)
        and item["fact"].strip()
    ]


def save_facts(facts):
    _get_chat_dir().mkdir(parents=True, exist_ok=True)
    facts_p = _get_facts_path()
    temporary_path = facts_p.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(facts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(facts_p)


def remember_fact(fact: str) -> str:
    """Запомнить пользовательский факт между диалогами.

    Args:
        fact: Короткий факт, который пользователь явно попросил запомнить.

    Returns:
        Подтверждение сохранения факта.
    """
    fact = " ".join(str(fact).split()).strip(" .")
    if not fact:
        return "Ошибка: нечего запоминать."
    if len(fact) > 500:
        return "Ошибка: один факт не должен быть длиннее 500 символов."

    facts = load_facts()
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
    save_facts(facts[-200:])
    return f"Запомнил: {fact}."


def _stem_word(word: str) -> str:
    """Удалить падежные окончания для точного поиска в русском языке."""
    w = word.casefold()
    if len(w) <= 3:
        return w
    for suffix in (
        "ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими", "ах", "ях",
        "ов", "ев", "ей", "ом", "ем", "ам", "ям", "ой", "ей", "ий", "ый",
        "ая", "яя", "ое", "ее", "ые", "ие", "ут", "ют", "ат", "ят", "ил",
        "ила", "или", "ит", "ет", "у", "ю", "а", "я", "е", "ы", "и", "о",
    ):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


def _memory_words(text: str) -> set[str]:
    words = set()
    for raw in re.findall(r"[a-zа-яё0-9_-]{2,}", text.casefold()):
        if raw not in _MEMORY_STOPWORDS:
            words.add(raw)
            stem = _stem_word(raw)
            if stem:
                words.add(stem)
    return words


def _is_general_query(query: str) -> bool:
    q = query.casefold().strip()
    if not q:
        return True
    return any(re.search(pat, q) for pat in _GENERAL_MEMORY_PATTERNS)


def recall_facts(query: str = "") -> str:
    """Найти факты в постоянной памяти W1x.

    Args:
        query: Что именно нужно вспомнить. Пустая строка показывает все сохранённые факты.

    Returns:
        Найденные факты или сообщение, что совпадений нет.
    """
    facts = load_facts()
    if not facts:
        return "Постоянная память фактов пуста."

    if _is_general_query(query):
        return "Факты из постоянной памяти:\n" + "\n".join(
            f"- {item['fact']}" for item in facts[-20:]
        )

    query_words = _memory_words(query)
    ranked = []
    for index, item in enumerate(facts):
        fact_words = _memory_words(item["fact"])
        overlap = len(query_words & fact_words)
        # Также проверяем частичные подстроки
        if not overlap:
            for qw in query_words:
                if len(qw) >= 3 and any(qw in fw or fw in qw for fw in fact_words):
                    overlap += 1
        if overlap:
            ranked.append((overlap, index, item["fact"]))

    if not ranked:
        # Если точных совпадений нет, но фактов мало (до 5) — возвращаем их
        if len(facts) <= 5:
            return "Факты из постоянной памяти:\n" + "\n".join(
                f"- {item['fact']}" for item in facts
            )
        return "В постоянной памяти нет подходящего факта."

    ranked.sort(reverse=True)
    selected = [fact for _, _, fact in ranked[:8]]
    return "Факты из постоянной памяти:\n" + "\n".join(
        f"- {fact}" for fact in selected
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
    facts = load_facts()
    kept = []
    removed = []
    for item in facts:
        fact_words = _memory_words(item["fact"])
        match = all(qw in fact_words for qw in query_words)
        if match:
            removed.append(item["fact"])
        else:
            kept.append(item)
    if not removed:
        return "Подходящий факт в памяти не найден."
    save_facts(kept)
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
    action = action.strip().casefold()
    if action == "remember":
        return remember_fact(text)
    if action == "recall":
        return recall_facts(text)
    if action == "forget":
        return forget_facts(text)
    return "Ошибка: action должен быть remember, recall или forget."


def relevant_memory_block(query=""):
    facts = load_facts()
    if not facts:
        return ""
    # Если сохранено до 30 фактов — подмешиваем их ВСЕ в контекст системного промпта!
    if len(facts) <= 30:
        return "Факты из постоянной памяти о пользователе:\n" + "\n".join(
            f"- {item['fact']}" for item in facts
        )
    recalled = recall_facts(query)
    return "" if recalled.endswith("пуста.") or "нет подходящего" in recalled else recalled


def extract_remember_request(text):
    match = re.match(
        r"^\s*(?:запомни|запиши\s+в\s+память|сохрани\s+в\s+память)"
        r"\s*[:,-]?\s*(.+?)\s*$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def extract_forget_request(text):
    match = re.match(
        r"^\s*(?:забудь|удали\s+из\s+памяти)\s*[:,-]?\s*(.+?)\s*$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def looks_like_memory_question(text):
    normalized = text.casefold()
    return bool(
        re.search(r"\b(?:помнишь|вспомни|забыл\w*)\b", normalized)
        or re.search(r"\bкак\s+называется\s+(?:мой|моя|мое|моё)\b", normalized)
        or re.search(r"\bчто\s+ты\s+(?:знаешь|помнишь)\s+обо?\s+мне\b", normalized)
    )


def migrate_explicit_memory(messages):
    """Один раз подобрать старые фразы «запомни» из сохранённого диалога."""
    migrated = 0
    for message in messages:
        if message.get("role") != "user":
            continue
        fact = extract_remember_request(message.get("content", ""))
        if not fact:
            continue
        before = len(load_facts())
        remember_fact(fact)
        migrated += len(load_facts()) > before
    return migrated
