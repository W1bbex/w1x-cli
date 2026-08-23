# -*- coding: utf-8 -*-
"""Главный интерфейс приложения W1x, слэш-команды и цикл диалога."""
import json
import time
from datetime import datetime
from pathlib import Path
from rich.panel import Panel

from .config import (
    APP_VERSION,
    CHAT_DIR,
    CURRENT_CHAT_PATH,
    FACTS_PATH,
    MAX_SAVED_MESSAGES,
    DEFAULT_SYSTEM_PROMPT,
    EXIT_COMMANDS,
    ACTIVE_SESSION_DATA,
)
from .terminal import (
    console,
    render_message,
    read_input,
    copy_to_clipboard,
    show_terminal_emoji_hint,
)
from .puller import GLOBAL_PULL_QUEUE, first_time_setup_wizard, pull_model
from .models import (
    choose_model,
    installed_models,
    preload_model,
    start_background_preload,
    wait_for_model_ready,
    reset_all_models,
    unload_all_models,
    interactive_config_menu,
    print_models,
    get_last_model,
)
from .memory import (
    load_facts,
    remember_fact,
    recall_facts,
    forget_facts,
    extract_remember_request,
    extract_forget_request,
    looks_like_memory_question,
    migrate_explicit_memory,
)
from .tools import (
    set_file_workspace,
    run_agy_for_user,
    should_auto_escalate,
)
from .testpy import (
    extract_latest_python_code,
    run_interactive_python_test,
)
from .engine import (
    generate_answer,
    direct_tool_stats,
    empty_totals,
    add_to_totals,
    print_stats,
)


def new_context(system_prompt):
    return [{"role": "system", "content": system_prompt}]


def save_chat(messages, requested_name=""):
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    if requested_name:
        safe_name = Path(requested_name).name
        if not safe_name.lower().endswith(".md"):
            safe_name += ".md"
    else:
        safe_name = datetime.now().strftime("chat-%Y-%m-%d-%H%M%S.md")
    path = CHAT_DIR / safe_name

    lines = ["# Ollama chat", ""]
    for message in messages:
        if message["role"] == "system":
            continue
        title = "Вы" if message["role"] == "user" else "Ассистент"
        lines.extend([f"## {title}", "", message["content"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_current_chat(system_prompt, messages):
    """Автоматически сохраняет текущий диалог локально."""
    saved_messages = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in messages
        if message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
    ][-MAX_SAVED_MESSAGES:]

    state = {
        "version": 1,
        "system_prompt": system_prompt,
        "custom_system_prompt": system_prompt != DEFAULT_SYSTEM_PROMPT,
        "messages": saved_messages,
    }

    try:
        CHAT_DIR.mkdir(parents=True, exist_ok=True)
        temporary_path = CURRENT_CHAT_PATH.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(CURRENT_CHAT_PATH)
    except OSError as error:
        console.print(f"[yellow]Не удалось сохранить память: {error}[/yellow]")


def load_current_chat():
    """Восстанавливает последний диалог или создаёт новый."""
    if not CURRENT_CHAT_PATH.exists():
        return DEFAULT_SYSTEM_PROMPT, new_context(DEFAULT_SYSTEM_PROMPT)

    try:
        state = json.loads(CURRENT_CHAT_PATH.read_text(encoding="utf-8"))
        saved_system_prompt = state.get("system_prompt") or ""
        has_custom_prompt = state.get("custom_system_prompt") is True
        system_prompt = (
            saved_system_prompt
            if has_custom_prompt and saved_system_prompt
            else DEFAULT_SYSTEM_PROMPT
        )
        saved_messages = []

        discard_legacy_style = (
            "custom_system_prompt" not in state
            and saved_system_prompt
            and saved_system_prompt != DEFAULT_SYSTEM_PROMPT
        )

        for message in ([] if discard_legacy_style else state.get("messages", [])):
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                saved_messages.append({"role": role, "content": content})

        if discard_legacy_style:
            console.print("[dim]Старый экспериментальный стиль сброшен.[/dim]")

        return system_prompt, [
            {"role": "system", "content": system_prompt},
            *saved_messages[-MAX_SAVED_MESSAGES:],
        ]
    except (OSError, ValueError, TypeError) as error:
        console.print(f"[yellow]Не удалось восстановить память: {error}[/yellow]")
        return DEFAULT_SYSTEM_PROMPT, new_context(DEFAULT_SYSTEM_PROMPT)


def print_status(model, think_mode, stats_enabled, messages):
    from . import config
    think_text = "выключен" if not think_mode else str(think_mode)
    turns = sum(message["role"] == "user" for message in messages)
    console.print(
        Panel(
            f"Модель: [bold]{model}[/bold]\n"
            f"ИИ: [bold]W1x[/bold]  •  разработчик: [bold]W1bbex[/bold]\n"
            f"Сборка: [bold]{APP_VERSION}[/bold]\n"
            f"Thinking: [bold]{think_text}[/bold]\n"
            "Инструменты: [bold]поиск, время, калькулятор, Python, память, "
            "файлы, Antigravity[/bold]\n"
            f"Папка проекта: [bold]{config.ACTIVE_FILE_ROOT or 'не выбрана'}[/bold]\n"
            f"Статистика: [bold]{'включена' if stats_enabled else 'выключена'}[/bold]\n"
            f"Сообщений пользователя: [bold]{turns}[/bold]",
            title="Состояние",
            border_style="cyan",
        )
    )


def print_header(model):
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


def print_help():
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


def shutdown():
    console.print()
    reset_all_models()
    console.print("[dim]Все модели выгружены.[/dim]")


def looks_like_launch_command(text):
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


def main():
    import sys
    from . import config

    cli_args = sys.argv[1:]
    if cli_args:
        first_arg = cli_args[0].lower()
        if first_arg in ("-q", "--unload", "/unload", "unload"):
            unloaded = unload_all_models()
            msg = ", ".join(unloaded) if unloaded else "нет активных моделей"
            console.print(f"[green]✓ Все модели выгружены из VRAM ({msg}).[/green]")
            return

        if first_arg in ("pull", "download", "--pull", "скачать"):
            models_to_pull = cli_args[1:]
            if not models_to_pull:
                first_time_setup_wizard()
                return
            GLOBAL_PULL_QUEUE.add(models_to_pull)
            GLOBAL_PULL_QUEUE.run_all()
            return

        if first_arg in ("--config", "-c", "config", "/config", "/settings"):
            interactive_config_menu()
            return

        models = installed_models()
        if not models:
            console.print("[yellow]W1x: первый запуск. Выберите стартовую модель для загрузки:[/yellow]")
            downloaded = first_time_setup_wizard()
            if not downloaded:
                return
            models = installed_models()

        target_model = get_last_model(models)
        question_parts = []
        i = 0
        while i < len(cli_args):
            arg = cli_args[i]
            if arg in ("-m", "--model") and i + 1 < len(cli_args):
                custom_m = cli_args[i + 1]
                if custom_m in models:
                    target_model = custom_m
                i += 2
            else:
                question_parts.append(arg)
                i += 1

        question = " ".join(question_parts).strip()
        if not question:
            question = "Привет!"

        system_prompt = config.DEFAULT_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        try:
            wait_for_model_ready(target_model)
            generate_answer(target_model, messages, think_mode=False)
        except KeyboardInterrupt:
            console.print("\n[yellow]Прервано пользователем.[/yellow]")
        except Exception as error:
            console.print(f"[red]Ошибка: {error}[/red]")
        return

    show_terminal_emoji_hint()
    model = choose_model()
    if model is None:
        shutdown()
        return

    # Запускаем прогрев в фоне: поле ввода открывается мгновенно, не блокируя пользователя!
    start_background_preload(model)

    system_prompt, messages = load_current_chat()
    ACTIVE_SESSION_DATA["messages"] = messages
    ACTIVE_SESSION_DATA["system_prompt"] = system_prompt
    migrated_facts = migrate_explicit_memory(messages)
    save_current_chat(system_prompt, messages)
    think_mode = False
    stats_enabled = False
    last_stats = None
    totals = empty_totals()
    last_response = next(
        (
            item["content"]
            for item in reversed(messages)
            if item["role"] == "assistant"
        ),
        "",
    )
    print_header(model)
    restored_turns = sum(item["role"] == "user" for item in messages)
    if restored_turns:
        console.print(
            f"[dim]Память восстановлена: {restored_turns} "
            f"сообщений пользователя.[/dim]"
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
            return

        if not prompt:
            continue

        command_line = prompt.split(maxsplit=1)
        command = command_line[0].lower()
        argument = command_line[1].strip() if len(command_line) > 1 else ""

        if prompt.lower() in EXIT_COMMANDS:
            shutdown()
            return

        if looks_like_launch_command(prompt):
            console.print(
                "[yellow]Это команда запуска Python. Сейчас ты внутри чата: "
                "введи q, а затем выполни команду в CMD или PowerShell.[/yellow]"
            )
            continue

        if command in {"/help", "/?"}:
            print_help()
            continue

        if command in {"/config", "/settings", "/настройки"}:
            interactive_config_menu()
            continue

        if command in {"/unload", "/free"}:
            reset_all_models(show_message=True)
            continue

        if command == "/models":
            print_models(installed_models())
            continue

        if command in {"/pull", "/download", "/скачать"}:
            models_to_pull = argument.split()
            if not models_to_pull:
                downloaded = first_time_setup_wizard()
                if downloaded:
                    model = downloaded
                    start_background_preload(model)
                    print_header(model)
            else:
                GLOBAL_PULL_QUEUE.add(models_to_pull)
                GLOBAL_PULL_QUEUE.run_all()
            continue

        if command == "/model":
            selected = choose_model(current=model)
            if selected is None or selected == model:
                continue
            reset_all_models()
            model = selected
            start_background_preload(model)
            print_header(model)
            continue

        if command in {"/new", "/clear"}:
            messages = new_context(system_prompt)
            ACTIVE_SESSION_DATA["messages"] = messages
            last_stats = None
            last_response = ""
            totals = empty_totals()
            save_current_chat(system_prompt, messages)
            console.print("[green]Новый диалог начат.[/green]")
            continue

        if command == "/cls":
            console.clear()
            print_header(model)
            continue

        if command == "/stats":
            stats_enabled = not stats_enabled
            state = "включена" if stats_enabled else "выключена"
            console.print(f"[green]Статистика {state}.[/green]")
            if stats_enabled:
                print_stats(last_stats, totals)
            continue

        if command == "/tools":
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
            continue

        if command == "/workspace":
            if not argument:
                console.print(
                    f"[dim]Текущая папка: {config.ACTIVE_FILE_ROOT or 'не выбрана'}[/dim]\n"
                    "Используй: [cyan]/workspace C:\\путь\\к\\проекту[/cyan] "
                    "или [cyan]/workspace off[/cyan]"
                )
                continue
            if argument.casefold() in {"off", "none", "выкл"}:
                config.ACTIVE_FILE_ROOT = None
                console.print("[green]Доступ к папке проекта отключён.[/green]")
                continue
            try:
                selected_root = set_file_workspace(argument.strip('"'))
                console.print(
                    f"[green]Папка проекта выбрана только для чтения:[/green] {selected_root}"
                )
            except Exception as error:
                console.print(f"[red]Не удалось выбрать папку: {error}[/red]")
            continue

        if command == "/agy":
            if not argument:
                console.print("[dim]Используй: /agy задача для старшей модели[/dim]")
                continue
            started_ns = time.perf_counter_ns()
            try:
                result = run_agy_for_user(argument, messages)
            except KeyboardInterrupt:
                console.print("\n[yellow]Antigravity остановлена пользователем.[/yellow]")
                continue
            messages.append({"role": "user", "content": argument})
            messages.append({"role": "assistant", "content": result})
            save_current_chat(system_prompt, messages)
            last_response = result
            last_stats = direct_tool_stats("Antigravity", started_ns)
            add_to_totals(totals, last_stats)
            if stats_enabled:
                print_stats(last_stats, totals)
            continue

        if command == "/think":
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
                continue
            label = "выключен" if not think_mode else str(think_mode)
            console.print(f"[green]Thinking: {label}.[/green]")
            continue

        if command == "/status":
            print_status(model, think_mode, stats_enabled, messages)
            continue

        if command == "/memory":
            turns = sum(item["role"] == "user" for item in messages)
            facts = load_facts()
            fact_preview = "\n".join(
                f"• {item['fact']}" for item in facts[-10:]
            ) or "Постоянных фактов пока нет."
            console.print(
                Panel(
                    f"Сохранено сообщений пользователя: [bold]{turns}[/bold]\n"
                    f"Лимит истории: [bold]{MAX_SAVED_MESSAGES} сообщений[/bold]\n"
                    f"Постоянных фактов: [bold]{len(facts)}[/bold]\n"
                    f"Файл диалога: {CURRENT_CHAT_PATH}\n"
                    f"Файл фактов: {FACTS_PATH}\n\n"
                    f"{fact_preview}\n\n"
                    "[dim]/new очищает диалог, но не постоянные факты; "
                    "/forget текст удаляет факт[/dim]",
                    title="Постоянная память",
                    border_style="cyan",
                )
            )
            continue

        if command == "/forget":
            result = forget_facts(argument)
            render_message(result, role="notice" if result.startswith("Ошибка") else "assistant")
            continue

        if command == "/system":
            if not argument:
                console.print(Panel(system_prompt, title="System prompt"))
            else:
                system_prompt = argument
                messages[0] = {"role": "system", "content": system_prompt}
                ACTIVE_SESSION_DATA["system_prompt"] = system_prompt
                save_current_chat(system_prompt, messages)
                console.print("[green]Системная инструкция обновлена.[/green]")
            continue

        if command == "/copy":
            if not last_response:
                console.print("[dim]Ответа для копирования пока нет.[/dim]")
                continue
            try:
                copy_to_clipboard(last_response)
                console.print("[green]Последний ответ скопирован.[/green]")
            except Exception as error:
                console.print(f"[red]Не удалось скопировать: {error}[/red]")
            continue

        if command == "/testpy":
            code_to_test = argument or extract_latest_python_code(messages)
            if not code_to_test:
                console.print("[yellow]В диалоге не найдено исполняемого Python-кода для тестирования.[/yellow]")
                console.print("[dim]Подсказка: вы можете передать код напрямую: /testpy print('hello')[/dim]")
                continue

            lines_count = len(code_to_test.splitlines())
            first_line = code_to_test.splitlines()[0][:60]
            console.print(f"[dim]Тестирую Python-код ({lines_count} строк: '{first_line}')...[/dim]")

            exit_code, log_content = run_interactive_python_test(code_to_test)
            if exit_code == 0:
                console.print("[green]✓ Тестирование завершено (код 0). Результат сохранён в диалог.[/green]")
            else:
                console.print(f"[yellow]⚠ Тестирование завершилось с кодом {exit_code}. Результат сохранён в диалог.[/yellow]")

            session_snippet = log_content[:6000] if log_content else "Скрипт выполнился без вывода в консоль"
            messages.append({
                "role": "user",
                "content": (
                    f"[Системный отчёт: пользователь запустил /testpy в отдельной консоли для проверки последнего кода]\n"
                    f"Код завершения: {exit_code}\n"
                    f"Лог взаимодействия в консоли:\n```\n{session_snippet}\n```\n"
                    f"(Если в логе ошибка или неверный результат — разбери причину и предложи исправление; если всё прошло успешно — подтверди)."
                )
            })
            messages.append({
                "role": "assistant",
                "content": f"Лог тестирования получен (код {exit_code}). Если нужно исправить логику или ошибки ввода — напиши!"
            })
            save_current_chat(system_prompt, messages)
            last_response = messages[-1]["content"]
            continue

        if command == "/save":
            try:
                path = save_chat(messages, argument)
                console.print(f"[green]Диалог сохранён:[/green] {path}")
            except Exception as error:
                console.print(f"[red]Не удалось сохранить: {error}[/red]")
            continue

        if command == "/undo":
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
            continue

        fact_to_remember = extract_remember_request(prompt)
        fact_to_forget = extract_forget_request(prompt)
        if fact_to_remember or fact_to_forget:
            started_ns = time.perf_counter_ns()
            result = (
                remember_fact(fact_to_remember)
                if fact_to_remember
                else forget_facts(fact_to_forget)
            )
            console.print()
            render_message("Память ›", role="assistant_label")
            render_message(result, role="assistant")
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": result})
            save_current_chat(system_prompt, messages)
            last_response = result
            last_stats = direct_tool_stats("memory", started_ns)
            add_to_totals(totals, last_stats)
            if stats_enabled:
                print_stats(last_stats, totals)
            continue

        if looks_like_memory_question(prompt):
            memory_answer = recall_facts(prompt)
            if "нет подходящего факта" not in memory_answer and not memory_answer.endswith("пуста."):
                started_ns = time.perf_counter_ns()
                console.print()
                render_message("Память ›", role="assistant_label")
                render_message(memory_answer, role="assistant")
                messages.append({"role": "user", "content": prompt})
                messages.append({"role": "assistant", "content": memory_answer})
                save_current_chat(system_prompt, messages)
                last_response = memory_answer
                last_stats = direct_tool_stats("memory", started_ns)
                add_to_totals(totals, last_stats)
                if stats_enabled:
                    print_stats(last_stats, totals)
                continue

        if command != "/retry" and should_auto_escalate(prompt):
            started_ns = time.perf_counter_ns()
            try:
                result = run_agy_for_user(prompt, messages)
            except KeyboardInterrupt:
                console.print("\n[yellow]Antigravity остановлена пользователем.[/yellow]")
                continue
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": result})
            save_current_chat(system_prompt, messages)
            last_response = result
            last_stats = direct_tool_stats("Antigravity", started_ns)
            add_to_totals(totals, last_stats)
            if stats_enabled:
                print_stats(last_stats, totals)
            continue

        retrying = command == "/retry"
        if retrying:
            if messages and messages[-1]["role"] == "assistant":
                messages.pop()
            if not messages or messages[-1]["role"] != "user":
                console.print("[dim]Нет вопроса для повторного ответа.[/dim]")
                continue
        else:
            messages.append({"role": "user", "content": prompt})

        try:
            wait_for_model_ready(model)
            response, response_stats, interrupted = generate_answer(
                model,
                messages,
                think_mode,
            )
        except Exception as error:
            console.print(
                f"\n[bold red]Ошибка Ollama:[/bold red] "
                f"{type(error).__name__}: {error}"
            )
            continue

        if interrupted:
            if messages and messages[-1].get("role") == "user":
                messages.pop()
                save_current_chat(system_prompt, messages)
            console.print("[yellow]Генерация остановлена. Запрос удалён из памяти модели (будто не писали).[/yellow]")
            console.print("[dim]💡 Нажми [cyan]Ctrl+S[/cyan], чтобы вернуть отменённый запрос в строку ввода.[/dim]")
            continue

        if not response:
            console.print("[yellow]Модель не вернула текст.[/yellow]")
            continue

        messages.append({"role": "assistant", "content": response})
        save_current_chat(system_prompt, messages)
        last_response = response
        last_stats = response_stats
        add_to_totals(totals, response_stats)

        if stats_enabled:
            print_stats(last_stats, totals)

        if "```" in response and any(kw in response for kw in ("```python", "```py", "def ", "import ")):
            console.print("[dim]💡 Протестировать код в отдельном окне: [cyan]/testpy[/cyan][/dim]")
