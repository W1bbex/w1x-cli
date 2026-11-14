"""Движок генерации ответов модели и оркестрация инструментов."""
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from ollama import chat
from rich.console import Group, RenderableType
from rich.live import Live
from rich.panel import Panel

from .config import (
    FORMAT_INSTRUCTIONS,
    MAX_RESPONSE_TOKENS,
    MAX_TOOL_ROUNDS,
    TOOL_INSTRUCTIONS,
    get_context_tokens,
    get_keep_alive,
    get_max_history_chars,
)
from .memory import relevant_memory_block
from .stats import collect_stats, direct_tool_stats, finalize_stats, merge_stats
from .terminal import ChatMarkdown, console, render_message, run_with_spinner
from .tools.antigravity import ask_antigravity, cancel_active_agy
from .tools.datetime_tool import (
    calculate_datetime,
    datetime_answer,
    split_datetime_request,
)
from .tools.math_tool import (
    calculate_expression,
    needs_clarification,
    parse_math_request,
    previous_math_expression,
)
from .tools.registry import (
    DIRECT_PASSTHROUGH_TOOLS,
    TOOL_FUNCTIONS,
    execute_tool_call,
    format_direct_tool_result,
)
from .types import ChatMessage, StatsRecord


def trim_messages_for_model(
    messages: Sequence[ChatMessage],
) -> list[dict[str, Any]]:
    """Собрать историю для контекста модели с сохранением последних сообщений."""
    system_messages = [item for item in messages if item.get("role") == "system"]
    conversation = [
        item for item in messages
        if item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ]

    total_chars = 0
    selected: list[dict[str, Any]] = []
    max_history = get_max_history_chars()
    # Всегда гарантированно сохраняем хотя бы последние 6 сообщений диалога
    for idx, item in enumerate(reversed(conversation)):
        content_len = len(item["content"])
        if idx >= 6 and (total_chars + content_len > max_history):
            break
        selected.append({"role": item["role"], "content": item["content"]})
        total_chars += content_len

    selected.reverse()
    system_payload = (
        system_messages[0]["content"] if system_messages else "Отвечай по-русски."
    )
    return [{"role": "system", "content": system_payload}, *selected]


def weak_tool_answer(text: str) -> bool:
    """Проверить, является ли ответ модели пустым, отказом или сигналом ESCALATE."""
    normalized = " ".join(text.lower().split())
    if not normalized or "escalate" in normalized:
        return True
    return bool(
        re.search(
            r"^(?:(?:к\s+сожалению|извините)[,\s]+)?(?:я\s+)?"
            r"(?:не\s+могу|не\s+удалось|не\s+получилось|"
            r"не\s+нашел|не\s+наш[её]л|не\s+вижу|нет\s+информации)",
            normalized,
        )
        or re.search(
            r"(?:поиск|инструмент).*(?:не\s+дал|не\s+вернул|не\s+помог|ошибк)",
            normalized,
        )
    )


def _precompute_math_block(last_user_text: str) -> str:
    """Извлечь и заранее вычислить точные математические подвыражения из вопроса."""
    precomputed_math = []
    has_explicit_math_intent = bool(
        re.search(
            r"(?:\b(?:посчитай|вычисли|сколько\s+будет|чему\s+равно|результат)\b|=)",
            last_user_text,
            re.IGNORECASE,
        )
    )
    normalized_text = last_user_text.replace("×", "*").replace("÷", "/")
    math_candidates = re.findall(r"[\d\s().,+\-*/%]{3,}", normalized_text)
    for cand in math_candidates:
        cand_clean = cand.strip(" ,.")
        if re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", cand_clean) or re.search(
            r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b", cand_clean
        ):
            continue
        if re.search(r"\b\d{1,3}(?:,\d{3})+\b", cand_clean):
            continue
        if re.search(r"^\+?\d[\d\s()-]{6,}\d$", cand_clean) and not re.search(
            r"[+*/%]|(?:\s-\s)", cand_clean.lstrip("+")
        ):
            continue
        only_hyphen_range = (
            "-" in cand_clean
            and not re.search(r"[+*/%]", cand_clean)
            and " - " not in cand_clean
        )
        if only_hyphen_range and not has_explicit_math_intent:
            continue
        normalized_expr = cand_clean.replace(",", ".")
        if (
            re.search(r"[+\-*/%]", normalized_expr)
            and re.search(r"\d", normalized_expr)
            and len(normalized_expr) >= 3
        ):
            val = calculate_expression(normalized_expr)
            if not val.startswith("Ошибка"):
                precomputed_math.append(f"{normalized_expr} = {val}")

    if not precomputed_math:
        return ""
    return (
        "Точные математические вычисления системы:\n"
        + "\n".join(f"- {item}" for item in precomputed_math)
        + "\nИспользуй эти точные результаты в ответе без повторного вызова калькулятора."
    )


def _build_runtime_system_prompt(
    model: str, base_prompt: str, last_user_text: str
) -> str:
    """Собрать системный промпт с текущей датой, правилами и памятью."""
    now_dt = datetime.now()
    date_info_block = (
        f"Текущая реальная дата и время: {now_dt.strftime('%d.%m.%Y (%A)')}, текущий {now_dt.year} год. "
        "Для актуальных вопросов (события, игры, патчи, новости 'сейчас/сегодня') всегда учитывай "
        f"{now_dt.year} год и составляй поисковые запросы с {now_dt.year} годом."
    )
    memory_block = relevant_memory_block(last_user_text)
    math_pre_block = _precompute_math_block(last_user_text)
    return (
        f"{base_prompt}\n\n"
        f"Сейчас внутри W1x работает базовая модель Ollama: {model}. "
        "Это не меняет твою личность W1x.\n\n"
        f"{date_info_block}\n\n"
        f"{TOOL_INSTRUCTIONS}\n\n{FORMAT_INSTRUCTIONS}"
        + (f"\n\n{memory_block}" if memory_block else "")
        + (f"\n\n{math_pre_block}" if math_pre_block else "")
    )


def _prepare_turn_or_shortcut(
    model: str,
    messages: Sequence[ChatMessage],
    last_user_text: str,
    request_started_ns: int,
) -> tuple[tuple[str, StatsRecord | None, bool] | None, list[dict[str, Any]] | None]:
    """Проверить мгновенные shortcut-ответы (уточнение, калькулятор, время) или подготовить историю."""
    if needs_clarification(last_user_text):
        answer = (
            "Уточни, пожалуйста, что именно нужно сделать: "
            "повторить прошлый ответ, изменить его или решить новую задачу?"
        )
        console.print()
        render_message("ИИ ›", role="assistant_label")
        render_message(answer, role="assistant")
        return (answer, direct_tool_stats("clarification", request_started_ns), False), None

    math_expression = parse_math_request(last_user_text)
    if math_expression is None and re.search(
        r"\bпроверь\w*\s+(?:это\s+)?калькулятор", last_user_text.casefold()
    ):
        math_expression = previous_math_expression(messages[:-1])
    if math_expression is not None:
        result = calculate_expression(math_expression)
        answer = format_direct_tool_result("calculate_expression", result)
        console.print()
        render_message("ИИ ›", role="assistant_label")
        console.print()
        render_message(f"Считаю: {math_expression}", role="tool")
        render_message("Ответ ›", role="assistant_label")
        render_message(answer, role="assistant")
        return (answer, direct_tool_stats("calculate_expression", request_started_ns), False), None

    datetime_arguments, remaining_request = split_datetime_request(last_user_text)
    if datetime_arguments is not None and not remaining_request:
        answer = datetime_answer(datetime_arguments)
        console.print()
        render_message("ИИ ›", role="assistant_label")
        console.print()
        render_message("Вычисляю точные дату и время", role="tool")
        render_message("Ответ ›", role="assistant_label")
        render_message(answer, role="assistant")
        return (answer, direct_tool_stats("calculate_datetime", request_started_ns), False), None

    working_messages = trim_messages_for_model(messages)
    if datetime_arguments is not None and remaining_request:
        for message in reversed(working_messages):
            if message.get("role") == "user":
                message["content"] = remaining_request
                break
    working_messages[0]["content"] = _build_runtime_system_prompt(
        model,
        working_messages[0]["content"],
        last_user_text,
    )
    return None, working_messages


def _build_live_renderable(thinking: str, answer: str) -> RenderableType | None:
    """Собрать объект Rich для live-отрисовки стрима модели."""
    renderables: list[RenderableType] = []
    if thinking:
        renderables.append(
            Panel(
                ChatMarkdown(thinking),
                title="Размышление",
                border_style="bright_black",
                padding=(0, 1),
            )
        )
    if answer:
        renderables.append(ChatMarkdown(answer))
    if len(renderables) == 1:
        return renderables[0]
    if len(renderables) > 1:
        return Group(*renderables)
    return None


def _stream_model_turn(
    model: str,
    working_messages: list[dict[str, Any]],
    think_mode: bool | str,
    force_final_answer: bool,
    followup_tool_functions: list[Callable[..., Any]] | None,
) -> tuple[str, str, list[Any], Any, bool, bool]:
    """Выполнить один потоковый запрос к Ollama и вернуть результат и флаг повтора без tools."""
    answer = ""
    thinking = ""
    tool_calls: list[Any] = []
    last_chunk = None
    interrupted = False
    stream = None
    chat_arguments: dict[str, Any] = {
        "model": model,
        "messages": working_messages,
        "stream": True,
        "think": think_mode,
        "keep_alive": get_keep_alive(),
        "options": {
            "temperature": 0.55 if force_final_answer else 0.72,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_ctx": get_context_tokens(),
            "num_predict": MAX_RESPONSE_TOKENS,
        },
    }
    if not force_final_answer:
        chat_arguments["tools"] = TOOL_FUNCTIONS
    elif followup_tool_functions:
        chat_arguments["tools"] = followup_tool_functions

    try:
        stream = chat(**chat_arguments)
        with Live(console=console, auto_refresh=False) as live:
            for chunk in stream:
                last_chunk = chunk
                chunk_thinking = getattr(chunk.message, "thinking", None) or ""
                chunk_answer = getattr(chunk.message, "content", None) or ""
                chunk_tools = getattr(chunk.message, "tool_calls", None) or []
                if chunk_tools:
                    tool_calls.extend(chunk_tools)
                if chunk_thinking:
                    thinking += chunk_thinking
                if chunk_answer:
                    answer += chunk_answer
                renderable = _build_live_renderable(thinking, answer)
                if renderable is not None:
                    live.update(renderable, refresh=True)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:
        error_msg = str(error)
        if chat_arguments.get("tools") and (
            "XML syntax error" in error_msg
            or "status code: -1" in error_msg
            or "tool" in error_msg.lower()
        ):
            return "", "", [], None, False, True
        raise error
    finally:
        if stream is not None:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    return answer, thinking, tool_calls, last_chunk, interrupted, False


def _escalate_weak_answer(
    last_user_text: str,
    answer: str,
    compact_tool_results: list[str],
    aggregate_stats: StatsRecord | None,
    request_started_ns: int,
) -> tuple[str, StatsRecord | None, bool]:
    """Передать вопрос в Antigravity после неудачной локальной попытки восстановления."""
    render_message(
        "Две локальные попытки не дали нормального ответа — ESCALATE",
        role="notice",
    )
    try:
        escalated = run_with_spinner(
            "Antigravity думает...",
            lambda: ask_antigravity(
                last_user_text,
                "\n\n".join(compact_tool_results)[-1200:],
            ),
            on_interrupt=cancel_active_agy,
        )
    except KeyboardInterrupt:
        return answer, finalize_stats(aggregate_stats, request_started_ns), True

    render_message("Antigravity ›", role="assistant_label")
    render_message(escalated, role="assistant")
    if aggregate_stats is not None:
        aggregate_stats["tool_answers"] = aggregate_stats.get("tool_answers", 0) + 1
    return escalated, finalize_stats(aggregate_stats, request_started_ns), False


def _execute_tool_batch(
    tool_calls: list[Any],
    working_messages: list[dict[str, Any]],
    compact_tool_results: list[str],
) -> list[tuple[str, str]]:
    """Выполнить пакет вызовов инструментов и зафиксировать результаты в истории."""
    round_tool_results: list[tuple[str, str]] = []
    for call in tool_calls:
        result = execute_tool_call(call)
        round_tool_results.append((call.function.name, result))
        compact_tool_results.append(f"{call.function.name}: {result[:2200]}")
        working_messages.append(
            {
                "role": "tool",
                "tool_name": call.function.name,
                "content": result,
            }
        )
    return round_tool_results


def _build_round_instruction(
    tool_rounds_used: int, used_names: set[str]
) -> tuple[bool, list[Callable[..., Any]] | None, str]:
    """Определить параметры следующего раунда диалога после вызова инструментов."""
    if (
        tool_rounds_used >= MAX_TOOL_ROUNDS
        or "search_web" in used_names
        or "run_python_code" in used_names
        or "memory" in used_names
    ):
        followup_tools: list[Callable[..., Any]] | None = (
            [calculate_datetime, calculate_expression]
            if "search_web" in used_names and tool_rounds_used < MAX_TOOL_ROUNDS
            else None
        )
        instruction = (
            "Теперь сформулируй итоговый, естественный и дружелюбный ответ для пользователя на основе полученных данных. "
            "Если использовалась память — ответь человеческим языком (например: 'Твоего хомяка зовут Ракета!'), без служебных маркеров. "
            "Если пользователь просил написать код — предоставь чистый код в блоке ```python ... ```. "
            "Не вызывай другие инструменты."
        )
        return True, followup_tools, instruction

    if tool_rounds_used == 1:
        return (
            False,
            None,
            "Проанализируй полученный результат. Если всё понятно — начни с прямого вывода и дай итоговый ответ.",
        )

    return (
        True,
        None,
        "Теперь ответь на исходный вопрос, используя результаты проверок выше. "
        "Начни с прямого вывода. Не вызывай другие инструменты.",
    )


def generate_answer(
    model: str,
    messages: Sequence[ChatMessage],
    think_mode: bool | str,
) -> tuple[str, StatsRecord | None, bool]:
    """Сгенерировать ответ модели, выполнив до MAX_TOOL_ROUNDS раундов инструментов."""
    request_started_ns = time.perf_counter_ns()
    last_user_text = next(
        (item["content"] for item in reversed(messages) if item.get("role") == "user"),
        "",
    )
    shortcut_result, working_messages = _prepare_turn_or_shortcut(
        model, messages, last_user_text, request_started_ns
    )
    if shortcut_result is not None:
        return shortcut_result
    assert working_messages is not None

    aggregate_stats: StatsRecord | None = None
    tools_used = False
    force_final_answer = False
    followup_tool_functions: list[Callable[..., Any]] | None = None
    recovery_attempted = False
    compact_tool_results: list[str] = []
    console.print()
    render_message("ИИ ›", role="assistant_label")

    tool_rounds_used = 0
    max_loop_attempts = MAX_TOOL_ROUNDS + 3
    for _attempt in range(max_loop_attempts):
        answer, thinking, tool_calls, last_chunk, interrupted, retry_no_tools = (
            _stream_model_turn(
                model,
                working_messages,
                think_mode,
                force_final_answer,
                followup_tool_functions,
            )
        )
        if retry_no_tools:
            force_final_answer = True
            followup_tool_functions = None
            continue

        should_recover = (
            tools_used
            and not tool_calls
            and weak_tool_answer(answer)
            and not recovery_attempted
        )
        should_escalate_after_recovery = (
            tools_used
            and not tool_calls
            and weak_tool_answer(answer)
            and recovery_attempted
        )

        aggregate_stats = merge_stats(aggregate_stats, collect_stats(last_chunk))
        if interrupted:
            return answer, finalize_stats(aggregate_stats, request_started_ns), True

        if thinking or answer or tool_calls:
            working_messages.append(
                {
                    "role": "assistant",
                    "thinking": thinking,
                    "content": answer,
                    "tool_calls": tool_calls,
                }
            )

        if not tool_calls:
            if should_escalate_after_recovery:
                return _escalate_weak_answer(
                    last_user_text,
                    answer,
                    compact_tool_results,
                    aggregate_stats,
                    request_started_ns,
                )
            if should_recover:
                recovery_attempted = True
                force_final_answer = True
                followup_tool_functions = None
                working_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Твой предыдущий ответ пустой или содержит отказ, хотя "
                            "инструменты уже вернули полезные данные. Ответь по "
                            "найденным данным прямо сейчас. Если информации точно "
                            "нет, ответь одним словом: ESCALATE"
                        ),
                    }
                )
                continue
            return answer, finalize_stats(aggregate_stats, request_started_ns), False

        if tool_rounds_used >= MAX_TOOL_ROUNDS:
            break
        tool_rounds_used += 1

        round_tool_results = _execute_tool_batch(
            tool_calls, working_messages, compact_tool_results
        )
        if round_tool_results:
            tools_used = True
        if round_tool_results and all(
            name in DIRECT_PASSTHROUGH_TOOLS for name, _ in round_tool_results
        ):
            answer = "\n\n".join(
                format_direct_tool_result(name, result)
                for name, result in round_tool_results
            )
            console.print()
            label = (
                "Antigravity ›"
                if any(name == "ask_antigravity" for name, _ in round_tool_results)
                else "Ответ ›"
            )
            render_message(label, role="assistant_label")
            render_message(answer, role="assistant")
            if aggregate_stats is not None:
                aggregate_stats["tool_answers"] = (
                    aggregate_stats.get("tool_answers", 0) + len(round_tool_results)
                )
            return answer, finalize_stats(aggregate_stats, request_started_ns), False

        used_names = {name for name, _ in round_tool_results}
        force_final_answer, followup_tool_functions, next_instruction = (
            _build_round_instruction(tool_rounds_used, used_names)
        )
        working_messages.append(
            {
                "role": "user",
                "content": f"{next_instruction}\nИсходный вопрос: {last_user_text}",
            }
        )

    answer = "Остановился: модель превысила лимит последовательных вызовов инструментов."
    render_message(answer, role="notice")
    return answer, finalize_stats(aggregate_stats, request_started_ns), False
