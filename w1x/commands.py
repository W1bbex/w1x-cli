"""Диспетчер и обработчики слэш-команд интерактивного чата W1x."""
import sys
from collections.abc import Callable, Sequence
from typing import Any

from rich.markup import escape as escape_markup
from rich.panel import Panel

from . import config
from . import models as models_mod
from .catalog_ui import first_time_setup_wizard
from .chat_store import save_chat, save_current_chat
from .clipboard import copy_to_clipboard
from .config import ACTIVE_SESSION_DATA, load_user_config
from .memory import (
    forget_facts,
    load_facts,
)
from .models import (
    reset_all_models,
    start_background_preload,
)
from .models_ui import choose_model, interactive_config_menu, print_models
from .pull_manager import GLOBAL_PULL_MANAGER
from .session import SessionState, print_header, print_help, print_status
from .terminal import console, print_stats, render_message
from .testpy import extract_latest_python_code, run_interactive_python_test
from .tools.workspace import set_file_workspace
from .types import ChatMessage


def _app_attr(name: str, fallback: Any) -> Any:
    app_mod = sys.modules.get("w1x.app")
    if app_mod is not None and hasattr(app_mod, name):
        return getattr(app_mod, name)
    return fallback


def looks_like_launch_command(text: str) -> bool:
    """Определить ошибочный ввод команды запуска скрипта в окно чата."""
    if "\n" in text or "?" in text:
        return False
    normalized = text.strip().lower().replace('"', "")
    if any(q in normalized for q in ("почему", "как ", "ошибка", " fails", " error", "why ", "how ", "what ")):
        return False
    return (
        ".py" in normalized
        and (
            "python.exe" in normalized
            or normalized.startswith("python ")
            or normalized.startswith("py ")
            or normalized.startswith("& ")
        )
    )


def print_tools_panel() -> None:
    """Вывести справочную панель доступных инструментов."""
    console.print(
        Panel(
            "[bold]search_web[/bold] — поиск свежей информации\n"
            "[bold]calculate_datetime[/bold] — дата, время и точные смещения\n"
            "[bold]calculate_expression[/bold] — точная арифметика\n"
            "[bold]run_python_code[/bold] — проверка Python, только после подтверждения\n"
            "[bold]remember/recall/forget[/bold] — постоянная память фактов\n"
            "[bold]list/read_project[/bold] — безопасное чтение выбранной папки\n"
            "[bold]ask_antigravity[/bold] — старшая модель для сложных задач\n\n"
            "[dim]Просто сформулируй задачу: модель сама выберет инструмент.[/dim]",
            title="Инструменты",
            border_style="magenta",
        )
    )


_print_tools_panel = print_tools_panel


def handle_workspace_cmd(argument: str) -> None:
    if not argument:
        console.print(
            f"[dim]Текущая папка: {escape_markup(str(config.ACTIVE_FILE_ROOT)) if config.ACTIVE_FILE_ROOT else 'не выбрана'}[/dim]\n"
            "Используй: [cyan]/workspace C:\\путь\\к\\проекту[/cyan] "
            "или [cyan]/workspace off[/cyan]"
        )
        return
    if argument.casefold() in {"off", "none", "выкл"}:
        config.ACTIVE_FILE_ROOT = None
        console.print("[green]Доступ к папке проекта отключён.[/green]")
        return
    try:
        selected_root = set_file_workspace(argument.strip('"'))
        console.print(
            f"[green]Папка проекта выбрана только для чтения:[/green] {escape_markup(str(selected_root))}"
        )
    except Exception as error:
        console.print(f"[red]Не удалось выбрать папку: {escape_markup(str(error))}[/red]")


_handle_workspace_cmd = handle_workspace_cmd


def handle_think_cmd(argument: str, think_mode: bool | str) -> bool | str:
    value = argument.lower()
    if not value:
        think_mode = False if think_mode else True
    elif value in {"off", "false", "0"}:
        think_mode = False
    elif value in {"on", "true", "1"}:
        think_mode = True
    elif value in {"low", "medium", "high"}:
        think_mode = value
    else:
        console.print("[red]Используй: /think off|on|low|medium|high[/red]")
        return think_mode
    label = "выключен" if not think_mode else str(think_mode)
    console.print(f"[green]Thinking: {label}.[/green]")
    return think_mode


_handle_think_cmd = handle_think_cmd


def handle_memory_cmd(messages: Sequence[ChatMessage]) -> None:
    turns = sum(item["role"] == "user" for item in messages)
    facts = load_facts()
    fact_preview = "\n".join(
        f"• {escape_markup(item['fact'])}" for item in facts[-10:]
    ) or "Постоянных фактов пока нет."
    console.print(
        Panel(
            f"Сохранено сообщений пользователя: [bold]{turns}[/bold]\n"
            f"Лимит истории: [bold]{config.MAX_SAVED_MESSAGES} сообщений[/bold]\n"
            f"Постоянных фактов: [bold]{len(facts)}[/bold]\n"
            f"Файл диалога: {escape_markup(str(config.CURRENT_CHAT_PATH))}\n"
            f"Файл фактов: {escape_markup(str(config.FACTS_PATH))}\n\n"
            f"{fact_preview}\n\n"
            "[dim]/new очищает диалог, но не постоянные факты; "
            "/forget текст удаляет факт[/dim]",
            title="Постоянная память",
            border_style="cyan",
        )
    )


_handle_memory_cmd = handle_memory_cmd


def handle_testpy_cmd(
    argument: str,
    messages: list[ChatMessage],
    system_prompt: str,
    last_response: str,
) -> str:
    code_to_test = argument or extract_latest_python_code(messages)
    if not code_to_test:
        console.print("[yellow]В диалоге не найдено исполняемого Python-кода для тестирования.[/yellow]")
        console.print("[dim]Подсказка: вы можете передать код напрямую: /testpy print('hello')[/dim]")
        return last_response

    lines_count = len(code_to_test.splitlines())
    first_line = code_to_test.splitlines()[0][:60]
    console.print(f"[dim]Тестирую Python-код ({lines_count} строк: '{escape_markup(first_line)}')...[/dim]")

    test_runner = _app_attr("run_interactive_python_test", run_interactive_python_test)
    exit_code, log_content = test_runner(code_to_test)
    if exit_code == 0:
        console.print("[green]✓ Тестирование завершено (код 0). Результат сохранён в диалог.[/green]")
    else:
        console.print(f"[yellow]⚠ Тестирование завершилось с кодом {exit_code}. Результат сохранён в диалог.[/yellow]")

    session_snippet = log_content[:6000] if log_content else "Скрипт выполнился без вывода в консоль"
    messages.append({
        "role": "user",
        "content": (
            "[Системный отчёт: пользователь запустил /testpy в отдельной консоли для проверки последнего кода]\n"
            f"Код завершения: {exit_code}\n"
            f"Лог взаимодействия в консоли:\n```\n{session_snippet}\n```\n"
            "(Если в логе ошибка или неверный результат — разбери причину и предложи исправление; если всё прошло успешно — подтверди)."
        )
    })
    messages.append({
        "role": "assistant",
        "content": f"Лог тестирования получен (код {exit_code}). Если нужно исправить логику или ошибки ввода — напиши!"
    })
    save_current_chat(system_prompt, messages)
    return messages[-1]["content"]


_handle_testpy_cmd = handle_testpy_cmd


def handle_undo_cmd(messages: list[ChatMessage], system_prompt: str) -> str:
    session_state = ACTIVE_SESSION_DATA.get("session_state")
    if session_state is not None and hasattr(session_state, "undo_turn"):
        removed = session_state.undo_turn()
        console.print(
            "[green]Последний ход удалён.[/green]"
            if removed
            else "[dim]Удалять нечего.[/dim]"
        )
        if removed:
            save_current_chat(system_prompt, messages)
        return getattr(session_state, "last_response", "")

    ACTIVE_SESSION_DATA.pop("cancelled_prompt", None)
    removed = False
    if messages and messages[-1]["role"] == "assistant":
        messages.pop()
        removed = True
    if messages and messages[-1]["role"] == "user":
        messages.pop()
        removed = True
    last_response = next(
        (
            item["content"]
            for item in reversed(messages)
            if item["role"] == "assistant"
        ),
        "",
    )
    console.print(
        "[green]Последний ход удалён.[/green]"
        if removed
        else "[dim]Удалять нечего.[/dim]"
    )
    if removed:
        save_current_chat(system_prompt, messages)
    return last_response


_handle_undo_cmd = handle_undo_cmd


def handle_pull_cmd(argument: str, model: str) -> str:
    models_to_pull = argument.split()
    if not models_to_pull:
        installed = _app_attr("installed_models", models_mod.installed_models)()
        if not installed:
            wizard_fn = _app_attr("first_time_setup_wizard", first_time_setup_wizard)
            downloaded = wizard_fn()
            if downloaded:
                model = downloaded
                preload_fn = _app_attr("start_background_preload", start_background_preload)
                preload_fn(model)
                header_fn = _app_attr("print_header", print_header)
                header_fn(model)
        else:
            from .catalog_ui import interactive_model_catalog, wait_for_download_queue
            interactive_model_catalog()
            wait_for_download_queue()
    else:
        pull_mgr = _app_attr("GLOBAL_PULL_MANAGER", GLOBAL_PULL_MANAGER)
        pull_mgr.run_cli_pull(models_to_pull)
    return model


_handle_pull_cmd = handle_pull_cmd


def dispatch_slash_command(
    command: str,
    argument: str,
    state: SessionState,
    run_agy_turn_fn: Callable[[str, str, SessionState], None] | None = None,
) -> bool:
    """Обработать встроенную слэш-команду и вернуть True, если ход завершён."""
    def _run_config_and_sync() -> None:
        cfg_menu = _app_attr("interactive_config_menu", interactive_config_menu)
        cfg_menu()
        load_cfg = _app_attr("load_user_config", load_user_config)
        updated_cfg = load_cfg()
        state.stats_enabled = bool(updated_cfg.get("stats_enabled", False))
        configured_model = updated_cfg.get("default_model")
        if configured_model and configured_model != state.model:
            reset_fn = _app_attr("reset_all_models", reset_all_models)
            reset_fn()
            state.model = configured_model
            preload_fn = _app_attr("start_background_preload", start_background_preload)
            preload_fn(state.model)
            header_fn = _app_attr("print_header", print_header)
            header_fn(state.model)

    simple_commands: dict[str, Callable[[], object]] = {
        "/help": lambda: _app_attr("print_help", print_help)(),
        "/?": lambda: _app_attr("print_help", print_help)(),
        "/config": _run_config_and_sync,
        "/settings": _run_config_and_sync,
        "/настройки": _run_config_and_sync,
        "/unload": lambda: reset_all_models(show_message=True),
        "/free": lambda: reset_all_models(show_message=True),
        "/models": lambda: _app_attr("print_models", print_models)(
            _app_attr("installed_models", models_mod.installed_models)()
        ),
        "/tools": lambda: _app_attr("print_tools_panel", print_tools_panel)(),
        "/workspace": lambda: handle_workspace_cmd(argument),
        "/memory": lambda: handle_memory_cmd(state.messages),
        "/status": lambda: _app_attr("print_status", print_status)(
            state.model,
            state.think_mode,
            state.stats_enabled,
            state.messages,
            state.totals,
        ),
    }
    if command in simple_commands:
        simple_commands[command]()
        return True

    if command in {"/pull", "/download", "/скачать"}:
        state.model = handle_pull_cmd(argument, state.model)
        return True

    if command == "/model":
        selected = choose_model(current=state.model)
        if selected is not None and selected != state.model:
            reset_all_models()
            state.model = selected
            start_background_preload(state.model)
            print_header(state.model)
        return True

    if command in {"/new", "/clear"}:
        state.reset_context()
        save_current_chat(state.system_prompt, state.messages)
        console.print("[green]Новый диалог начат.[/green]")
        return True

    if command == "/cls":
        console.clear()
        print_header(state.model)
        return True

    if command == "/stats":
        state.stats_enabled = not state.stats_enabled
        label = "включена" if state.stats_enabled else "выключена"
        console.print(f"[green]Статистика {label}.[/green]")
        if state.stats_enabled:
            print_stats(state.last_stats)
        return True

    if command == "/agy":
        if not argument:
            console.print("[dim]Используй: /agy задача для старшей модели[/dim]")
        elif run_agy_turn_fn is not None:
            run_agy_turn_fn(argument, argument, state)
        return True

    if command == "/think":
        state.think_mode = handle_think_cmd(argument, state.think_mode)
        return True

    if command == "/forget":
        result = forget_facts(argument)
        render_message(result, role="notice" if result.startswith("Ошибка") else "assistant")
        return True

    if command == "/system":
        if not argument:
            console.print(Panel(escape_markup(state.system_prompt), title="System prompt"))
        else:
            state.system_prompt = argument
            state.messages[0] = {"role": "system", "content": state.system_prompt}
            state.sync_active_data()
            save_current_chat(state.system_prompt, state.messages)
            console.print("[green]Системная инструкция обновлена.[/green]")
        return True

    if command == "/copy":
        resp_to_copy = next(
            (item["content"] for item in reversed(state.messages) if item.get("role") == "assistant"),
            state.last_response,
        )
        if not resp_to_copy:
            console.print("[dim]Ответа для копирования пока нет.[/dim]")
            return True
        try:
            copy_fn = _app_attr("copy_to_clipboard", copy_to_clipboard)
            copy_fn(resp_to_copy)
            console.print("[green]Последний ответ скопирован.[/green]")
        except Exception as error:
            console.print(f"[red]Не удалось скопировать: {escape_markup(str(error))}[/red]")
        return True

    if command == "/testpy":
        state.last_response = handle_testpy_cmd(
            argument, state.messages, state.system_prompt, state.last_response
        )
        return True

    if command == "/save":
        try:
            path = save_chat(state.messages, argument)
            console.print(f"[green]Диалог сохранён:[/green] {escape_markup(str(path))}")
        except Exception as error:
            console.print(f"[red]Не удалось сохранить: {escape_markup(str(error))}[/red]")
        return True

    if command == "/undo":
        if state.undo_turn():
            save_current_chat(state.system_prompt, state.messages)
            console.print("[green]Последний ход удалён.[/green]")
        else:
            console.print("[dim]Удалять нечего.[/dim]")
        return True

    return False


_dispatch_slash_command = dispatch_slash_command
