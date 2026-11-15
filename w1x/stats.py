"""Сбор и агрегация телеметрии генерации и инструментов."""
import time
from typing import Any

from .types import SessionTotals, StatsRecord


def format_duration(nanoseconds: int | float | None) -> str:
    """Форматировать длительность из наносекунд в читаемую строку."""
    if not nanoseconds:
        return "—"
    if nanoseconds < 1_000_000:
        return "<1 мс"
    seconds = nanoseconds / 1_000_000_000
    return f"{seconds * 1000:.0f} мс" if seconds < 1 else f"{seconds:.2f} с"


def collect_stats(response: Any) -> StatsRecord | None:
    """Извлечь метрики токенов и скорости из ответа Ollama."""
    if response is None:
        return None
    prompt_tokens = int(getattr(response, "prompt_eval_count", None) or 0)
    output_tokens = int(getattr(response, "eval_count", None) or 0)
    eval_duration = int(getattr(response, "eval_duration", None) or 0)
    speed = (
        output_tokens / (eval_duration / 1_000_000_000)
        if output_tokens and eval_duration
        else 0.0
    )
    return {
        "source": "model",
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "total_duration": getattr(response, "total_duration", None),
        "load_duration": getattr(response, "load_duration", None),
        "prompt_duration": getattr(response, "prompt_eval_duration", None),
        "eval_duration": eval_duration,
        "speed": speed,
    }


def direct_tool_stats(tool_name: str, started_ns: int) -> StatsRecord:
    """Сформировать статистику для прямого вызова инструмента без LLM."""
    duration_ns = time.perf_counter_ns() - started_ns
    return {
        "source": "direct_tool",
        "tool_name": tool_name,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_duration": duration_ns,
        "load_duration": 0,
        "prompt_duration": 0,
        "eval_duration": duration_ns,
        "speed": 0.0,
    }


def merge_stats(
    aggregate: StatsRecord | None,
    step_stats: StatsRecord | None,
) -> StatsRecord | None:
    """Объединить статистику нескольких раундов генерации/инструментов."""
    if step_stats is None:
        return aggregate
    if aggregate is None:
        return dict(step_stats)  # type: ignore[return-value]

    aggregate["prompt_tokens"] = max(
        aggregate.get("prompt_tokens", 0),
        step_stats.get("prompt_tokens", 0),
    )
    aggregate["output_tokens"] = (
        aggregate.get("output_tokens", 0)
        + step_stats.get("output_tokens", 0)
    )
    aggregate["total_tokens"] = (
        aggregate.get("prompt_tokens", 0)
        + aggregate.get("output_tokens", 0)
    )
    aggregate["eval_duration"] = (
        aggregate.get("eval_duration", 0)
        + (step_stats.get("eval_duration") or 0)
    )
    return aggregate


def finalize_stats(aggregate: StatsRecord | None, started_ns: int) -> StatsRecord | None:
    """Вычислить итоговое время ответа и среднюю скорость токенов."""
    if aggregate is None:
        return None
    total_duration = time.perf_counter_ns() - started_ns
    eval_duration = aggregate.get("eval_duration") or 0
    output_tokens = aggregate.get("output_tokens") or 0
    speed = (
        output_tokens / (eval_duration / 1_000_000_000)
        if output_tokens and eval_duration
        else 0.0
    )
    aggregate["total_duration"] = total_duration
    aggregate["speed"] = speed
    return aggregate


def empty_totals() -> SessionTotals:
    """Создать пустой накопитель сессионной статистики."""
    return {
        "requests": 0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_duration": 0,
    }


def add_to_totals(totals: SessionTotals, stats: StatsRecord | None) -> None:
    """Добавить статистику одного ответа к накопителю сессии."""
    if stats is None:
        return
    totals["requests"] += 1
    totals["prompt_tokens"] += stats.get("prompt_tokens") or 0
    totals["output_tokens"] += stats.get("output_tokens") or 0
    totals["total_tokens"] += stats.get("total_tokens") or 0
    totals["total_duration"] += stats.get("total_duration") or 0
