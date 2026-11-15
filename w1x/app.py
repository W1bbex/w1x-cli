"""Главный интерфейс приложения W1x и цикл интерактивного диалога."""
import re
import sys
import time

from prompt_toolkit.formatted_text import HTML
from rich.markup import escape as escape_markup

from . import config
from . import models as models_mod
from .catalog_ui import first_time_setup_wizard
from .chat_store import (
    load_current_chat,
    save_current_chat,
)
from .cli import run_cli_oneshot
from .clipboard import copy_to_clipboard
from .commands import (
    dispatch_slash_command,
    handle_memory_cmd,
    handle_pull_cmd,
    handle_testpy_cmd,
    handle_think_cmd,
    handle_undo_cmd,
    handle_workspace_cmd,
    looks_like_launch_command,
    print_tools_panel,
)
from .config import ACTIVE_SESSION_DATA, EXIT_COMMANDS, load_user_config
from .engine import generate_answer
from .logging_setup import setup_logging
from .memory import (
    extract_forget_request,
    extract_remember_request,
    forget_facts,
    looks_like_memory_question,
    migrate_explicit_memory,
    recall_facts,
    remember_fact,
)
from .models import (
    register_console_ctrl_handler,
    reset_all_models,
    start_background_preload,
    wait_for_model_ready,
)
from .models_ui import choose_model, interactive_config_menu, print_models
from .prompts import (
    read_input,
    secondary_session,
    set_undo_persist_callback,
)
from .pull_manager import GLOBAL_PULL_MANAGER
from .session import (
    SessionState,
    print_header,
    print_help,
    print_status,
)
from .stats import direct_tool_stats
from .terminal import (
    configure_utf8_console,
    console,
    print_stats,
    render_message,
    show_terminal_emoji_hint,
)
from .testpy import run_interactive_python_test
from .tools.antigravity import run_agy_for_user, should_auto_escalate

__all__ = [
    "GLOBAL_PULL_MANAGER",
    "SessionState",
    "copy_to_clipboard",
    "first_time_setup_wizard",
    "interactive_config_menu",
    "main",
    "print_header",
    "print_help",
    "print_models",
    "print_status",
    "run_interactive_python_test",
]

# Обратная совместимость для тестов и внешних вызовов
_SessionState = SessionState
_run_cli_oneshot = run_cli_oneshot
_dispatch_slash_command = dispatch_slash_command
_handle_memory_cmd = handle_memory_cmd
_handle_testpy_cmd = handle_testpy_cmd
_handle_undo_cmd = handle_undo_cmd
_handle_pull_cmd = handle_pull_cmd
_handle_think_cmd = handle_think_cmd
_handle_workspace_cmd = handle_workspace_cmd
_print_tools_panel = print_tools_panel


def installed_models() -> list[str]:
    """Вернуть установленные модели через текущую ссылку модуля models."""
    return models_mod.installed_models()


def shutdown() -> None:
    """Выгрузить активные модели перед завершением CLI."""
    console.print()
    unloaded = reset_all_models()
    if unloaded:
        console.print(f"[dim]Модели выгружены из памяти: {', '.join(unloaded)}[/dim]")
    else:
        console.print("[dim]Сессия завершена.[/dim]")


def _record_direct_reply(
    prompt_text: str,
    reply_text: str,
    tool_name: str,
    started_ns: int,
    state: SessionState,
) -> None:
    """Записать мгновенный ответ инструмента или памяти в историю и обновить статистику."""
    stats = direct_tool_stats(tool_name, started_ns)
    state.append_user_message(prompt_text)
    state.append_assistant_reply(reply_text, stats)
    save_current_chat(state.system_prompt, state.messages)
    if state.stats_enabled and state.last_stats:
        print_stats(state.last_stats)


def _run_agy_turn(user_text: str, task_text: str, state: SessionState) -> None:
    """Выполнить запрос к старшей модели Antigravity с сохранением результата."""
    started_ns = time.perf_counter_ns()
    try:
        result = run_agy_for_user(task_text, state.messages)
    except KeyboardInterrupt:
        console.print("\n[yellow]Antigravity остановлена пользователем.[/yellow]")
        return
    if result.startswith("ESCALATE_ERROR:"):
        return
    _record_direct_reply(user_text, result, "Antigravity", started_ns, state)


def _try_memory_shortcut(prompt: str, state: SessionState) -> bool:
    """Обработать явный запрос к постоянной памяти без запуска модели."""
    fact_to_remember = extract_remember_request(prompt)
    fact_to_forget = extract_forget_request(prompt)
    if fact_to_remember:
        result = remember_fact(fact_to_remember)
    elif fact_to_forget:
        result = forget_facts(fact_to_forget)
    else:
        result = None

    if result is not None:
        started_ns = time.perf_counter_ns()
        console.print()
        render_message("Память ›", role="assistant_label")
        render_message(result, role="assistant")
        _record_direct_reply(prompt, result, "memory", started_ns, state)
        return True

    if looks_like_memory_question(prompt):
        memory_answer = recall_facts(prompt)
        if "нет подходящего факта" not in memory_answer and not memory_answer.endswith("пуста."):
            started_ns = time.perf_counter_ns()
            console.print()
            render_message("Память ›", role="assistant_label")
            render_message(memory_answer, role="assistant")
            _record_direct_reply(prompt, memory_answer, "memory", started_ns, state)
            return True

    return False


def _execute_chat_turn(
    prompt: str, retrying: bool, state: SessionState
) -> None:
    """Выполнить ход диалога с Ollama и сохранить результат."""
    messages = state.messages
    if retrying:
        if messages and messages[-1]["role"] == "assistant":
            messages.pop()
        if not messages or messages[-1]["role"] != "user":
            saved_cancelled = state.pop_cancelled_prompt()
            if saved_cancelled:
                state.append_user_message(saved_cancelled)
            else:
                console.print("[dim]Нет вопроса для повторного ответа.[/dim]")
                return
        retry_user_prompt = messages[-1]["content"]
    else:
        retry_user_prompt = prompt
        state.append_user_message(prompt)

    try:
        wait_for_model_ready(state.model)
        response, response_stats, interrupted = generate_answer(
            state.model,
            messages,
            state.think_mode,
        )
    except KeyboardInterrupt:
        interrupted = True
        response, response_stats = "", None
    except Exception as error:
        state.rollback_last_turn()
        save_current_chat(state.system_prompt, messages)
        console.print(
            f"\n[bold red]Ошибка Ollama:[/bold red] "
            f"{type(error).__name__}: {escape_markup(str(error))}"
        )
        return

    if interrupted:
        state.set_cancelled_prompt(retry_user_prompt)
        state.rollback_last_turn()
        save_current_chat(state.system_prompt, messages)
        if retrying:
            console.print("[yellow]Генерация остановлена. Исходный вопрос сохранён — можешь нажать [cyan]Ctrl+S[/cyan] или повторить `/retry`.[/yellow]")
        else:
            console.print("[yellow]Генерация остановлена. Запрос удалён из памяти модели (будто не писали).[/yellow]")
            console.print("[dim]💡 Нажми [cyan]Ctrl+S[/cyan], чтобы вернуть отменённый запрос в строку ввода.[/dim]")
        return

    if not response:
        state.rollback_last_turn()
        save_current_chat(state.system_prompt, messages)
        console.print("[yellow]Модель не вернула текст.[/yellow]")
        return

    state.append_assistant_reply(response, response_stats)
    save_current_chat(state.system_prompt, messages)

    if state.stats_enabled and state.last_stats:
        print_stats(state.last_stats)

    if "```" in response and any(kw in response for kw in ("```python", "```py", "def ", "import ")):
        console.print("[dim]💡 Протестировать код в отдельном окне: [cyan]/testpy[/cyan][/dim]")


def _run_interactive_chat(model: str) -> int:
    """Запустить основной интерактивный цикл чата с выбранной моделью."""
    start_background_preload(model)

    system_prompt, messages = load_current_chat()
    ACTIVE_SESSION_DATA["messages"] = messages
    ACTIVE_SESSION_DATA["system_prompt"] = system_prompt
    migrated_facts = migrate_explicit_memory(messages)
    save_current_chat(system_prompt, messages)

    last_response = next(
        (
            item["content"]
            for item in reversed(messages)
            if item["role"] == "assistant"
        ),
        "",
    )
    state = SessionState(
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        think_mode=False,
        stats_enabled=bool(load_user_config().get("stats_enabled", False)),
        last_response=last_response,
    )
    print_header(state.model)
    restored_turns = sum(item["role"] == "user" for item in state.messages)
    if restored_turns:
        console.print(
            f"[dim]Память восстановлена: {restored_turns} "
            "сообщений пользователя.[/dim]"
        )
    if migrated_facts:
        console.print(
            f"[dim]В постоянную память перенесено фактов: {migrated_facts}.[/dim]"
        )

    while True:
        try:
            prompt = read_input().strip()
        except KeyboardInterrupt:
            console.print("\n[dim]Для выхода введи q или /exit.[/dim]")
            continue
        except EOFError:
            shutdown()
            return 0

        if not prompt:
            continue

        command_line = prompt.split(maxsplit=1)
        command = command_line[0].lower()
        argument = command_line[1].strip() if len(command_line) > 1 else ""

        try:
            if prompt.lower() in EXIT_COMMANDS:
                shutdown()
                return 0

            if looks_like_launch_command(prompt):
                console.print(
                    "[yellow]Это команда запуска Python. Сейчас ты внутри чата: "
                    "введи q, а затем выполни команду в CMD или PowerShell.[/yellow]"
                )
                continue

            if dispatch_slash_command(command, argument, state, run_agy_turn_fn=_run_agy_turn):
                continue

            if _try_memory_shortcut(prompt, state):
                continue

            if command != "/retry" and should_auto_escalate(prompt):
                is_explicit_agy = bool(
                    re.search(r"\b(?:agy|антигравити)\b", prompt.casefold())
                    or re.search(
                        r"\b(?:передай|эскалируй|спроси|отдай)\w*\b.*\b(?:старш|сильн|другой)\w*\s+(?:ии|модел)",
                        prompt.casefold(),
                    )
                )
                if not is_explicit_agy:
                    console.print(
                        "\n[cyan]💡 Запрос похож на сложную задачу для старшей модели.[/cyan]"
                    )
                    try:
                        confirm = secondary_session.prompt(
                            HTML("<b><ansicyan>Передать задачу в Antigravity? [y/N] › </ansicyan></b>")
                        ).strip().lower()
                    except (KeyboardInterrupt, EOFError):
                        confirm = "n"
                    if confirm not in {"y", "yes", "д", "да"}:
                        _execute_chat_turn(prompt, retrying=False, state=state)
                        continue

                _run_agy_turn(prompt, prompt, state)
                continue

            _execute_chat_turn(prompt, retrying=(command == "/retry"), state=state)
        except KeyboardInterrupt:
            console.print("\n[dim]Действие прервано. Для выхода введи q или /exit.[/dim]")
            continue


def main() -> None:
    """Точка входа CLI-приложения W1x."""
    config.ensure_data_dirs()
    cli_args = sys.argv[1:]
    is_debug = "--debug" in cli_args
    if is_debug:
        cli_args = [a for a in cli_args if a != "--debug"]
    setup_logging(debug=is_debug)

    configure_utf8_console()
    register_console_ctrl_handler()
    set_undo_persist_callback(save_current_chat)

    if cli_args:
        exit_code = run_cli_oneshot(cli_args, run_interactive_fn=_run_interactive_chat)
        if exit_code:
            raise SystemExit(exit_code)
        return

    show_terminal_emoji_hint()
    model = choose_model()
    if model is None:
        reset_all_models()
        return

    _run_interactive_chat(model)
