# -*- coding: utf-8 -*-
"""Движок генерации ответов модели, вызов инструментов и статистика."""
import re
import sys
import time
from datetime import datetime
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from ollama import chat

from .config import (
    MAX_TOOL_ROUNDS,
    MAX_RESPONSE_TOKENS,
    TOOL_INSTRUCTIONS,
    FORMAT_INSTRUCTIONS,
    SPINNER,
    get_keep_alive,
    get_context_tokens,
    get_max_history_chars,
)
from .terminal import (
    console,
    render_message,
    run_with_spinner,
    uses_modern_windows_terminal,
    ChatMarkdown,
)
from .tools import (
    calculate_datetime,
    calculate_expression,
    parse_math_request,
    previous_math_expression,
    split_datetime_request,
    datetime_answer,
    needs_clarification,
    ask_antigravity,
    TOOL_FUNCTIONS,
    DIRECT_PASSTHROUGH_TOOLS,
    format_direct_tool_result,
    execute_tool_call,
)
from .memory import relevant_memory_block


def trim_messages_for_model(messages):
    """Собрать историю для контекста модели с сохранением последних сообщений."""
    system_messages = [item for item in messages if item.get("role") == "system"]
    conversation = [
        item for item in messages
        if item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ]

    total_chars = 0
    selected = []
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


def weak_tool_answer(text):
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


def format_duration(nanoseconds):
    if not nanoseconds:
        return "—"
    if nanoseconds < 1_000_000:
        return "<1 мс"
    seconds = nanoseconds / 1_000_000_000
    return f"{seconds * 1000:.0f} мс" if seconds < 1 else f"{seconds:.2f} с"


def collect_stats(response):
    if response is None:
        return None
    prompt_tokens = getattr(response, "prompt_eval_count", None) or 0
    output_tokens = getattr(response, "eval_count", None) or 0
    eval_duration = getattr(response, "eval_duration", None) or 0
    speed = (
        output_tokens / (eval_duration / 1_000_000_000)
        if output_tokens and eval_duration
        else 0
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


def direct_tool_stats(tool_name, started_ns):
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
        "speed": 0,
    }


def merge_stats(aggregate, step_stats):
    if step_stats is None:
        return aggregate
    if aggregate is None:
        return dict(step_stats)

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


def finalize_stats(aggregate, started_ns):
    if aggregate is None:
        return None
    total_duration = time.perf_counter_ns() - started_ns
    eval_duration = aggregate.get("eval_duration") or 0
    output_tokens = aggregate.get("output_tokens") or 0
    speed = (
        output_tokens / (eval_duration / 1_000_000_000)
        if output_tokens and eval_duration
        else 0
    )
    aggregate["total_duration"] = total_duration
    aggregate["speed"] = speed
    return aggregate


def empty_totals():
    return {
        "requests": 0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_duration": 0,
    }


def add_to_totals(totals, stats):
    if stats is None:
        return
    totals["requests"] += 1
    totals["prompt_tokens"] += stats.get("prompt_tokens", 0)
    totals["output_tokens"] += stats.get("output_tokens", 0)
    totals["total_tokens"] += stats.get("total_tokens", 0)
    totals["total_duration"] += stats.get("total_duration") or 0


def print_stats(stats, totals):
    if stats is None:
        return

    if stats.get("source") == "direct_tool":
        duration = format_duration(stats.get("total_duration"))
        tool_label = stats.get("tool_name", "инструмент")
        console.print(f"[dim]Ответ: {tool_label}  •  время: {duration}[/dim]")
        return

    duration = format_duration(stats.get("total_duration"))
    eval_time = format_duration(stats.get("eval_duration"))
    prompt_tokens = stats.get("prompt_tokens", 0)
    output_tokens = stats.get("output_tokens", 0)
    speed = stats.get("speed", 0)
    console.print(
        f"[dim]Токены: {prompt_tokens} ввод + {output_tokens} ответ  •  "
        f"скорость: {speed:.1f} т/с  •  время: {duration} (генерация {eval_time})[/dim]"
    )


def generate_answer(model, messages, think_mode):
    request_started_ns = time.perf_counter_ns()
    last_user_text = next(
        (
            item["content"]
            for item in reversed(messages)
            if item.get("role") == "user"
        ),
        "",
    )

    if needs_clarification(last_user_text):
        answer = "Уточни, пожалуйста, что именно нужно сделать: повторить прошлый ответ, изменить его или решить новую задачу?"
        console.print()
        render_message("ИИ ›", role="assistant_label")
        render_message(answer, role="assistant")
        return answer, direct_tool_stats("clarification", request_started_ns), False

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
        return answer, direct_tool_stats("calculate_expression", request_started_ns), False

    datetime_arguments, remaining_request = split_datetime_request(last_user_text)
    if datetime_arguments is not None and not remaining_request:
        answer = datetime_answer(datetime_arguments)
        console.print()
        render_message("ИИ ›", role="assistant_label")
        console.print()
        render_message("Вычисляю точные дату и время", role="tool")
        render_message("Ответ ›", role="assistant_label")
        render_message(answer, role="assistant")
        return (
            answer,
            direct_tool_stats("calculate_datetime", request_started_ns),
            False,
        )

    working_messages = trim_messages_for_model(messages)
    if datetime_arguments is not None and remaining_request:
        for message in reversed(working_messages):
            if message.get("role") == "user":
                message["content"] = remaining_request
                break
    # Находим математические подвыражения в тексте для мгновенного точного авто-вычисления
    precomputed_math = []
    math_candidates = re.findall(r"[\d\s().,+\-*/%]{3,}", last_user_text.replace("×", "*").replace("÷", "/"))
    for cand in math_candidates:
        cand_clean = cand.strip(" ,.")
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b", cand_clean) or re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", cand_clean):
            continue
        if re.search(r"^\+?\d[\d\s-]{7,}\d$", cand_clean):
            continue
        if re.search(r"[+\-*/%]", cand_clean) and re.search(r"\d", cand_clean) and len(cand_clean) >= 3:
            val = calculate_expression(cand_clean)
            if not val.startswith("Ошибка"):
                precomputed_math.append(f"{cand_clean} = {val}")

    math_pre_block = ""
    if precomputed_math:
        math_pre_block = (
            "Точные математические вычисления системы:\n"
            + "\n".join(f"- {item}" for item in precomputed_math)
            + "\nИспользуй эти точные результаты в ответе без повторного вызова калькулятора."
        )

    now_dt = datetime.now()
    date_info_block = (
        f"Текущая реальная дата и время: {now_dt.strftime('%d.%m.%Y (%A)')}, текущий {now_dt.year} год. "
        f"Для актуальных вопросов (события, игры, патчи, новости 'сейчас/сегодня') всегда учитывай "
        f"{now_dt.year} год и составляй поисковые запросы с {now_dt.year} годом."
    )

    memory_block = relevant_memory_block(last_user_text)
    working_messages[0]["content"] = (
        f"{working_messages[0]['content']}\n\n"
        f"Сейчас внутри W1x работает базовая модель Ollama: {model}. "
        "Это не меняет твою личность W1x.\n\n"
        f"{date_info_block}\n\n"
        f"{TOOL_INSTRUCTIONS}\n\n{FORMAT_INSTRUCTIONS}"
        + (f"\n\n{memory_block}" if memory_block else "")
        + (f"\n\n{math_pre_block}" if math_pre_block else "")
    )
    aggregate_stats = None
    tools_used = False
    force_final_answer = False
    followup_tool_functions = None
    recovery_attempted = False
    compact_tool_results = []
    console.print()
    render_message("ИИ ›", role="assistant_label")

    for tool_round in range(MAX_TOOL_ROUNDS):
        answer = ""
        thinking = ""
        tool_calls = []
        last_chunk = None
        interrupted = False
        stream = None
        chat_arguments = {
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
            with Live(
                console=console,
                auto_refresh=False,
            ) as live:
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

                    renderables = []
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
                        live.update(renderables[0], refresh=True)
                    elif len(renderables) > 1:
                        live.update(Group(*renderables), refresh=True)
        except KeyboardInterrupt:
            interrupted = True
        except Exception as error:
            error_msg = str(error)
            if not force_final_answer and (
                "XML syntax error" in error_msg
                or "status code: -1" in error_msg
                or "tool" in error_msg.lower()
            ):
                force_final_answer = True
                continue
            raise error
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

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
            return (
                answer,
                finalize_stats(aggregate_stats, request_started_ns),
                True,
            )

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
                render_message(
                    "Две локальные попытки не дали нормального ответа — ESCALATE",
                    role="notice",
                )
                escalated = run_with_spinner(
                    "Antigravity думает...",
                    lambda: ask_antigravity(
                        last_user_text,
                        "\n\n".join(compact_tool_results)[-1200:],
                    ),
                )
                render_message("Antigravity ›", role="assistant_label")
                render_message(escalated, role="assistant")
                if aggregate_stats is not None:
                    aggregate_stats["tool_answers"] = (
                        aggregate_stats.get("tool_answers", 0) + 1
                    )
                return (
                    escalated,
                    finalize_stats(aggregate_stats, request_started_ns),
                    False,
                )
            if should_recover:
                recovery_attempted = True
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
            return (
                answer,
                finalize_stats(aggregate_stats, request_started_ns),
                False,
            )

        round_tool_results = []
        for call in tool_calls:
            result = execute_tool_call(call)
            round_tool_results.append((call.function.name, result))
            tools_used = True
            compact_tool_results.append(
                f"{call.function.name}: {result[:2200]}"
            )
            working_messages.append(
                {
                    "role": "tool",
                    "tool_name": call.function.name,
                    "content": result,
                }
            )
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
            return (
                answer,
                finalize_stats(aggregate_stats, request_started_ns),
                False,
            )
        used_names = {name for name, _ in round_tool_results}
        python_results = [
            result
            for name, result in round_tool_results
            if name == "run_python_code"
        ]
        python_check_unreliable = any(
            "СТАТУС ПРОВЕРКИ: НЕОПРЕДЕЛЁН" in result
            or not result.startswith("Код завершения: 0")
            for result in python_results
        )
        if "search_web" in used_names or "run_python_code" in used_names or "memory" in used_names:
            force_final_answer = True
            followup_tool_functions = [calculate_datetime, calculate_expression] if "search_web" in used_names else None
            next_instruction = (
                "Теперь сформулируй итоговый, естественный и дружелюбный ответ для пользователя на основе полученных данных. "
                "Если использовалась память — ответь человеческим языком (например: 'Твоего хомяка зовут Ракета!'), без служебных маркеров. "
                "Если пользователь просил написать код — предоставь чистый код в блоке ```python ... ```. "
                "Не вызывай другие инструменты."
            )
        elif tool_round == 0 and not python_check_unreliable:
            force_final_answer = False
            followup_tool_functions = None
            next_instruction = (
                "Проанализируй полученный результат. Если всё понятно — начни с прямого вывода и дай итоговый ответ."
            )
        else:
            force_final_answer = True
            followup_tool_functions = None
            next_instruction = (
                "Теперь ответь на исходный вопрос, используя результаты проверок выше. "
                "Начни с прямого вывода. Не вызывай другие инструменты."
            )
        working_messages.append(
            {
                "role": "user",
                "content": (
                    f"{next_instruction}\n"
                    f"Исходный вопрос: {last_user_text}"
                ),
            }
        )

    answer = "Остановился: модель превысила лимит последовательных вызовов инструментов."
    render_message(answer, role="notice")
    return answer, finalize_stats(aggregate_stats, request_started_ns), False
