"""Главный интерфейс приложения W1x, слэш-команды и цикл диалога."""
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rich.markup import escape as escape_markup
from rich.panel import Panel

from . import config
from . import models as models_mod
from .catalog_ui import first_time_setup_wizard
from .chat_store import (
    load_current_chat,
    new_context,
    save_chat,
    save_current_chat,
)
from .config import ACTIVE_SESSION_DATA, APP_VERSION, EXIT_COMMANDS, load_user_config
from .engine import generate_answer
from .memory import (
    extract_forget_request,
    extract_remember_request,
    forget_facts,
    load_facts,
    looks_like_memory_question,
    migrate_explicit_memory,
    recall_facts,
    remember_fact,
)
from .models import (
    get_last_model,
    register_console_ctrl_handler,
    reset_all_models,
    start_background_preload,
    unload_all_models,
    wait_for_model_ready,
)
from .models_ui import choose_model, interactive_config_menu, print_models
from .pull_manager import GLOBAL_PULL_MANAGER
from .stats import add_to_totals, direct_tool_stats, empty_totals
from prompt_toolkit.formatted_text import HTML

from .clipboard import copy_to_clipboard
from .prompts import (
    read_input,
    secondary_session,
    set_undo_persist_callback,
)
from .terminal import (
    configure_utf8_console,
    console,
    print_stats,
    render_message,
    show_terminal_emoji_hint,
)
from .testpy import extract_latest_python_code, run_interactive_python_test
from .tools.antigravity import run_agy_for_user, should_auto_escalate
from .tools.workspace import set_file_workspace
from .types import ChatMessage, SessionTotals, StatsRecord


def installed_models() -> list[str]:
    """Вернуть установленные модели через текущую ссылку модуля models."""
    return models_mod.installed_models()


@dataclass
class _SessionState:
    """Мутабельное состояние активной интерактивной сессии W1x."""

    model: str
    system_prompt: str
    messages: list[ChatMessage]
    think_mode: bool | str = False
    stats_enabled: bool = False
    last_stats: StatsRecord | None = None
    totals: SessionTotals = field(default_factory=empty_totals)
    last_response: str = ""
    cancelled_prompt: str = ""

    def __post_init__(self) -> None:
        self.sync_active_data()

    def sync_active_data(self) -> None:
        """Синхронизировать состояние с глобальным фасадом для обратной совместимости."""
        config.ACTIVE_SESSION_DATA["messages"] = self.messages
        config.ACTIVE_SESSION_DATA["system_prompt"] = self.system_prompt
        if self.cancelled_prompt:
            config.ACTIVE_SESSION_DATA["cancelled_prompt"] = self.cancelled_prompt
        else:
            config.ACTIVE_SESSION_DATA.pop("cancelled_prompt", None)

    def set_cancelled_prompt(self, prompt: str) -> None:
        self.cancelled_prompt = prompt
        config.ACTIVE_SESSION_DATA["cancelled_prompt"] = prompt

    def pop_cancelled_prompt(self) -> str:
        res = self.cancelled_prompt or str(config.ACTIVE_SESSION_DATA.pop("cancelled_prompt", "") or "")
        self.cancelled_prompt = ""
        config.ACTIVE_SESSION_DATA.pop("cancelled_prompt", None)
        return res

    def append_user_message(self, prompt: str) -> None:
        self.pop_cancelled_prompt()
        self.messages.append({"role": "user", "content": prompt})
        self.sync_active_data()

    def append_assistant_reply(self, reply: str, stats: StatsRecord | None) -> None:
        self.pop_cancelled_prompt()
        self.messages.append({"role": "assistant", "content": reply})
        self.last_response = reply
        self.last_stats = stats
        if stats:
            add_to_totals(self.totals, stats)
        self.sync_active_data()

    def reset_context(self) -> None:
        self.pop_cancelled_prompt()
        self.messages = new_context(self.system_prompt)
        self.last_stats = None
        self.last_response = ""
        self.totals = empty_totals()
        self.sync_active_data()

    def undo_turn(self) -> bool:
        self.pop_cancelled_prompt()
        removed = False
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages.pop()
            removed = True
        if self.messages and self.messages[-1].get("role") == "user":
            self.messages.pop()
            removed = True
        if removed:
            self.last_response = next(
                (item["content"] for item in reversed(self.messages) if item["role"] == "assistant"),
                "",
            )
            self.sync_active_data()
        return removed

    def rollback_last_turn(self) -> None:
        if self.messages and self.messages[-1].get("role") == "user":
            self.messages.pop()
        self.sync_active_data()


def print_status(
    model: str,
    think_mode: bool | str,
    stats_enabled: bool,
    messages: Sequence[ChatMessage],
    totals: SessionTotals | None = None,
) -> None:
    """Вывести панель текущего состояния сессии и модели."""
    think_text = "выключен" if not think_mode else str(think_mode)
    turns = sum(message["role"] == "user" for message in messages)
    lines = [
        f"Модель: [bold]{model}[/bold]",
        "ИИ: [bold]W1x[/bold]  •  разработчик: [bold]W1bbex[/bold]",
        f"Сборка: [bold]{APP_VERSION}[/bold]",
        f"Thinking: [bold]{think_text}[/bold]",
        "Инструменты: [bold]поиск, время, калькулятор, Python, память, "
        "файлы, Antigravity[/bold]",
        f"Папка проекта: [bold]{escape_markup(str(config.ACTIVE_FILE_ROOT)) if config.ACTIVE_FILE_ROOT else 'не выбрана'}[/bold]",
        f"Статистика: [bold]{'включена' if stats_enabled else 'выключена'}[/bold]",
        f"Сообщений пользователя: [bold]{turns}[/bold]",
    ]
    if totals and totals.get("requests", 0) > 0:
        reqs = totals.get("requests", 0)
        tot_tok = totals.get("total_tokens", 0)
        prompt_tok = totals.get("prompt_tokens", 0)
        out_tok = totals.get("output_tokens", 0)
        lines.append(
            f"Токенов за сессию: [bold]{tot_tok:,}[/bold] ([dim]вход: {prompt_tok:,}, выход: {out_tok:,}[/dim]) "
            f"в [bold]{reqs}[/bold] ответах"
        )
    console.print(
        Panel(
            "\n".join(lines),
            title="Состояние",
            border_style="cyan",
        )
    )


def print_header(model: str) -> None:
    """Вывести стартовый баннер с активной моделью."""
    console.print(
        Panel(
            f"[bold green]{model}[/bold green]\n"
            f"[dim]W1x CLI  •  v{APP_VERSION}[/dim]\n"
            "[dim]Enter — отправить  •  Shift+Enter — новая строка  •  "
            "/help — команды[/dim]",
            title="W1x",
            border_style="green",
        )
    )


def print_help() -> None:
    """Вывести справку по горячим клавишам и слэш-командам."""
    console.print(
        """
[bold]Ввод[/bold]
Enter               отправить сообщение
Shift+Enter          новая строка
Ctrl+Enter           новая строка
Alt+Enter            запасной вариант новой строки
Ctrl+S               вернуть прошлый вопрос в строку ввода и стереть из памяти
Ctrl+Q               выгрузить все модели и выйти
Ctrl+Backspace       удалить слово слева
Ctrl+Delete          удалить слово справа
Ctrl+A               выделить весь текст; Delete/Backspace — очистить
Ctrl+X               вырезать выделенный текст (или весь ввод) в буфер
Ctrl+C               копировать выделение / остановить генерацию
Ctrl+V               вставить текст из буфера обмена
/ком...             серое продолжение и список; Tab/Enter — подставить

[bold]Команды[/bold]
[cyan]/model[/cyan]             сменить модель (стрелки ↑/↓ или Enter)
[cyan]/pull [модели...][/cyan]  скачать модели из библиотеки Ollama
[cyan]/unload[/cyan], [cyan]/free[/cyan]    выгрузить активные модели из VRAM
[cyan]/config[/cyan]            настройки контекста, VRAM и модели
[cyan]/models[/cyan]            список установленных моделей
[cyan]/new[/cyan], [cyan]/clear[/cyan]       начать новый диалог
[cyan]/undo[/cyan]              удалить последний вопрос и ответ
[cyan]/retry[/cyan]             заново ответить на последний вопрос
[cyan]/think[/cyan]             включить/выключить размышления
[cyan]/think low|medium|high[/cyan]  выбрать уровень размышлений
[cyan]/tools[/cyan]             доступные инструменты
[cyan]/agy задача[/cyan]        передать задачу Antigravity напрямую
[cyan]/workspace путь[/cyan]    выбрать папку проекта для безопасного чтения
[cyan]/stats[/cyan]             статистика прошлого и следующих ответов
[cyan]/status[/cyan]            текущие настройки
[cyan]/system[/cyan]            показать системную инструкцию
[cyan]/system текст[/cyan]      заменить системную инструкцию
[cyan]/copy[/cyan]              скопировать последний ответ
[cyan]/testpy[/cyan]            протестировать последний Python-код в отдельном окне
[cyan]/save[/cyan] [имя]        сохранить диалог в Markdown
[cyan]/memory[/cyan]            показать состояние постоянной памяти
[cyan]/forget текст[/cyan]      удалить подходящий факт из памяти
[cyan]/cls[/cyan]               очистить экран
[cyan]/help[/cyan]              эта справка
[cyan]q[/cyan], [cyan]/exit[/cyan]           выгрузить модели и выйти
"""
    )


def shutdown() -> None:
    """Выгрузить активные модели перед завершением CLI."""
    console.print()
    unloaded = reset_all_models()
    if unloaded:
        console.print(f"[dim]Модели выгружены из памяти: {', '.join(unloaded)}[/dim]")
    else:
        console.print("[dim]Сессия завершена.[/dim]")


def looks_like_launch_command(text: str) -> bool:
    """Определить ошибочный ввод команды запуска скрипта в окно чата."""
    normalized = text.strip().lower().replace('"', "")
    return (
        ".py" in normalized
        and (
            "python.exe" in normalized
            or normalized.startswith("python ")
            or normalized.startswith("py ")
            or normalized.startswith("& ")
        )
    )


def _print_tools_panel() -> None:
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


def _handle_workspace_cmd(argument: str) -> None:
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


def _handle_think_cmd(argument: str, think_mode: bool | str) -> bool | str:
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


def _handle_memory_cmd(messages: Sequence[ChatMessage]) -> None:
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


def _handle_testpy_cmd(
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

    exit_code, log_content = run_interactive_python_test(code_to_test)
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


def _handle_undo_cmd(messages: list[ChatMessage], system_prompt: str) -> str:
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


def _handle_pull_cmd(argument: str, model: str) -> str:
    models_to_pull = argument.split()
    if not models_to_pull:
        downloaded = first_time_setup_wizard()
        if downloaded:
            model = downloaded
            start_background_preload(model)
            print_header(model)
    else:
        GLOBAL_PULL_MANAGER.run_cli_pull(models_to_pull)
    return model


def _run_cli_oneshot(cli_args: list[str]) -> int:
    """Обработать одиночные CLI-аргументы и флаги без входа в интерактивный цикл."""
    if any(a.lower() in ("-h", "--help") for a in cli_args):
        console.print(
            f"[bold]w1x[/bold] v{APP_VERSION} — CLI-ассистент для Ollama\n\n"
            "Использование:\n"
            "  w1x                       Запустить интерактивный чат\n"
            "  w1x [флаги] <вопрос>      Задать одиночный вопрос\n"
            "  w1x pull [модели...]      Скачать модели в очереди\n"
            "  w1x --config              Открыть меню настроек\n"
            "  w1x --unload              Выгрузить все модели из VRAM\n\n"
            "Флаги:\n"
            "  -m, --model <имя>         Выбрать установленную модель\n"
            "  -v, --version             Показать версию\n"
            "  -h, --help                Показать эту справку"
        )
        return 0

    first_arg = cli_args[0].lower()
    if len(cli_args) == 1 and first_arg in ("-v", "--version"):
        console.print(f"w1x {APP_VERSION}")
        return 0

    if first_arg in ("-q", "--unload", "/unload", "unload"):
        unloaded = unload_all_models()
        msg = ", ".join(unloaded) if unloaded else "нет активных моделей"
        console.print(f"[green]✓ Все модели выгружены из VRAM ({msg}).[/green]")
        return 0

    if first_arg in ("pull", "download", "--pull", "скачать"):
        models_to_pull = cli_args[1:]
        if not models_to_pull:
            first_time_setup_wizard()
            return 0
        success = GLOBAL_PULL_MANAGER.run_cli_pull(models_to_pull)
        return 0 if success else 1

    if first_arg in ("--config", "-c", "config", "/config", "/settings"):
        interactive_config_menu()
        return 0

    requested_model: str | None = None
    question_parts: list[str] = []
    i = 0
    while i < len(cli_args):
        arg = cli_args[i]
        if arg in ("-m", "--model"):
            if i + 1 >= len(cli_args):
                console.print("[red]Ошибка: для флага -m/--model не указано имя модели.[/red]")
                return 2
            requested_model = cli_args[i + 1]
            i += 2
        elif arg.lower().startswith("--model="):
            requested_model = arg.split("=", 1)[1].strip()
            if not requested_model:
                console.print("[red]Ошибка: для флага --model не указано имя модели.[/red]")
                return 2
            i += 1
        elif arg.startswith("-") and not question_parts:
            console.print(f"[red]Неизвестный флаг: {arg}. Используйте --help для справки.[/red]")
            return 2
        else:
            question_parts.append(arg)
            i += 1

    models = installed_models()
    if not models:
        if models_mod.LAST_LIST_ERROR is not None:
            return 1
        console.print("[yellow]W1x: первый запуск. Выберите стартовую модель для загрузки:[/yellow]")
        downloaded = first_time_setup_wizard()
        if not downloaded:
            return 1
        models = installed_models()
        if not models:
            return 1

    if requested_model is not None:
        if requested_model not in models:
            console.print(
                f"[red]Ошибка: модель '{requested_model}' не установлена в Ollama.[/red]\n"
                "[dim]Установите её через `w1x pull <модель>` или проверьте список в `w1x --config`.[/dim]"
            )
            return 1
        target_model = requested_model
    else:
        chosen_last = get_last_model(models)
        target_model = chosen_last if chosen_last is not None else models[0]

    if not question_parts and requested_model is not None:
        show_terminal_emoji_hint()
        return _run_interactive_chat(target_model)

    question = " ".join(question_parts).strip() or "Привет!"
    messages: list[ChatMessage] = [
        {"role": "system", "content": config.DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        wait_for_model_ready(target_model)
        _, _, interrupted = generate_answer(target_model, messages, think_mode=False)
        if interrupted:
            console.print("\n[yellow]Прервано пользователем.[/yellow]")
            return 130
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Прервано пользователем.[/yellow]")
        return 130
    except Exception as error:
        console.print(f"[red]Ошибка: {error}[/red]")
        return 1


def _record_direct_reply(
    prompt_text: str,
    reply_text: str,
    tool_name: str,
    started_ns: int,
    state: _SessionState,
) -> None:
    """Записать мгновенный ответ инструмента или памяти в историю и обновить статистику."""
    stats = direct_tool_stats(tool_name, started_ns)
    state.append_user_message(prompt_text)
    state.append_assistant_reply(reply_text, stats)
    save_current_chat(state.system_prompt, state.messages)
    if state.stats_enabled and state.last_stats:
        print_stats(state.last_stats)


def _run_agy_turn(user_text: str, task_text: str, state: _SessionState) -> None:
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


def _dispatch_slash_command(
    command: str, argument: str, state: _SessionState
) -> bool:
    """Обработать встроенную слэш-команду и вернуть True, если ход завершён."""
    def _run_config_and_sync() -> None:
        interactive_config_menu()
        updated_cfg = load_user_config()
        state.stats_enabled = bool(updated_cfg.get("stats_enabled", False))
        configured_model = updated_cfg.get("default_model")
        if configured_model and configured_model != state.model:
            reset_all_models()
            state.model = configured_model
            start_background_preload(state.model)
            print_header(state.model)

    simple_commands: dict[str, Callable[[], object]] = {
        "/help": print_help,
        "/?": print_help,
        "/config": _run_config_and_sync,
        "/settings": _run_config_and_sync,
        "/настройки": _run_config_and_sync,
        "/unload": lambda: reset_all_models(show_message=True),
        "/free": lambda: reset_all_models(show_message=True),
        "/models": lambda: print_models(installed_models()),
        "/tools": _print_tools_panel,
        "/workspace": lambda: _handle_workspace_cmd(argument),
        "/memory": lambda: _handle_memory_cmd(state.messages),
        "/status": lambda: print_status(
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
        state.model = _handle_pull_cmd(argument, state.model)
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
        else:
            _run_agy_turn(argument, argument, state)
        return True

    if command == "/think":
        state.think_mode = _handle_think_cmd(argument, state.think_mode)
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
            copy_to_clipboard(resp_to_copy)
            console.print("[green]Последний ответ скопирован.[/green]")
        except Exception as error:
            console.print(f"[red]Не удалось скопировать: {escape_markup(str(error))}[/red]")
        return True

    if command == "/testpy":
        state.last_response = _handle_testpy_cmd(
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
        state.last_response = _handle_undo_cmd(state.messages, state.system_prompt)
        return True

    return False


def _try_memory_shortcut(prompt: str, state: _SessionState) -> bool:
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
    prompt: str, retrying: bool, state: _SessionState
) -> None:
    """Выполнить ход диалога с Ollama и сохранить результат."""
    messages = state.messages
    if retrying:
        if messages and messages[-1]["role"] == "assistant":
            messages.pop()
        if not messages or messages[-1]["role"] != "user":
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
        if not retrying:
            state.rollback_last_turn()
        save_current_chat(state.system_prompt, messages)
        console.print(
            f"\n[bold red]Ошибка Ollama:[/bold red] "
            f"{type(error).__name__}: {escape_markup(str(error))}"
        )
        return

    if interrupted:
        state.set_cancelled_prompt(retry_user_prompt)
        if not retrying:
            state.rollback_last_turn()
        save_current_chat(state.system_prompt, messages)
        if retrying:
            console.print("[yellow]Генерация остановлена. Исходный вопрос сохранён — можешь повторить `/retry`.[/yellow]")
        else:
            console.print("[yellow]Генерация остановлена. Запрос удалён из памяти модели (будто не писали).[/yellow]")
            console.print("[dim]💡 Нажми [cyan]Ctrl+S[/cyan], чтобы вернуть отменённый запрос в строку ввода.[/dim]")
        return

    if not response:
        if not retrying:
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
    # Запускаем прогрев в фоне: поле ввода открывается мгновенно, не блокируя пользователя!
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
    state = _SessionState(
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

            if _dispatch_slash_command(command, argument, state):
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
    import sys

    configure_utf8_console()
    register_console_ctrl_handler()
    set_undo_persist_callback(save_current_chat)

    cli_args = sys.argv[1:]
    if cli_args:
        exit_code = _run_cli_oneshot(cli_args)
        if exit_code:
            raise SystemExit(exit_code)
        return

    show_terminal_emoji_hint()
    model = choose_model()
    if model is None:
        reset_all_models()
        return

    _run_interactive_chat(model)
