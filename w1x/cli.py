"""Обработка аргументов командной строки и одиночных (oneshot) запросов W1x."""
from collections.abc import Callable
from typing import Any

from . import models as models_mod
from .catalog_ui import first_time_setup_wizard
from .config import APP_VERSION
from .engine import generate_answer
from .models import (
    get_last_model,
    installed_models,
    unload_all_models,
    wait_for_model_ready,
)
from .models_ui import interactive_config_menu
from .prompts_text import DEFAULT_SYSTEM_PROMPT
from .pull_manager import GLOBAL_PULL_MANAGER
from .terminal import console, show_terminal_emoji_hint
from .types import ChatMessage


def print_cli_help() -> None:
    """Вывести краткую справку командной строки."""
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
        "  --debug                   Включить подробный вывод отладки\n"
        "  -h, --help                Показать эту справку"
    )


def run_cli_oneshot(
    cli_args: list[str],
    *,
    installed_models_fn: Callable[..., list[str]] | None = None,
    wait_for_model_ready_fn: Callable[[str], Any] | None = None,
    generate_answer_fn: Callable[..., Any] | None = None,
    run_interactive_fn: Callable[[str], int] | None = None,
) -> int:
    """Обработать одиночные CLI-аргументы и флаги без входа в интерактивный цикл."""
    import sys

    def _app_attr(name: str, fallback: Any) -> Any:
        app_mod = sys.modules.get("w1x.app")
        if app_mod is not None and hasattr(app_mod, name):
            return getattr(app_mod, name)
        return fallback

    _inst_fn: Callable[..., list[str]] = installed_models_fn or _app_attr("installed_models", installed_models)
    _wait_fn: Callable[[str], Any] = wait_for_model_ready_fn or _app_attr("wait_for_model_ready", wait_for_model_ready)
    _gen_fn: Callable[..., Any] = generate_answer_fn or _app_attr("generate_answer", generate_answer)
    _interactive_fn: Any = run_interactive_fn or _app_attr("_run_interactive_chat", None)
    _wizard_fn: Any = _app_attr("first_time_setup_wizard", first_time_setup_wizard)
    _config_menu_fn: Any = _app_attr("interactive_config_menu", interactive_config_menu)
    _pull_mgr: Any = _app_attr("GLOBAL_PULL_MANAGER", GLOBAL_PULL_MANAGER)

    if len(cli_args) == 1 and cli_args[0].lower() in ("-h", "--help"):
        print_cli_help()
        return 0

    first_arg = cli_args[0].lower()
    if len(cli_args) == 1 and first_arg in ("-v", "--version"):
        console.print(f"w1x {APP_VERSION}")
        return 0

    if len(cli_args) == 1 and first_arg in ("-q", "--unload", "/unload", "unload"):
        unloaded = unload_all_models()
        msg = ", ".join(unloaded) if unloaded else "нет активных моделей"
        console.print(f"[green]✓ Все модели выгружены из VRAM ({msg}).[/green]")
        return 0

    is_pull_cmd = first_arg in ("pull", "download", "--pull", "скачать")
    is_question_pull = is_pull_cmd and any(
        kw in " ".join(cli_args).lower()
        for kw in ("request", "git", "репозитор", "запрос", "how", "как", "what", "что", "menu", "down")
    )
    if is_pull_cmd and not is_question_pull:
        models_to_pull = cli_args[1:]
        if not models_to_pull:
            models = _inst_fn()
            if not models:
                _wizard_fn()
            else:
                from .catalog_ui import interactive_model_catalog, wait_for_download_queue
                interactive_model_catalog()
                wait_for_download_queue()
            return 0
        success = _pull_mgr.run_cli_pull(models_to_pull)
        return 0 if success else 1

    if len(cli_args) == 1 and first_arg in ("--config", "-c", "config", "/config", "/settings"):
        _config_menu_fn()
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
        elif arg in ("-h", "--help") and not question_parts and i == len(cli_args) - 1:
            print_cli_help()
            return 0
        elif arg.startswith("-") and not question_parts:
            clean_num = arg[1:]
            if clean_num.isdigit() or (clean_num.replace(".", "", 1).isdigit() and clean_num != "."):
                question_parts.append(arg)
                i += 1
            else:
                console.print(f"[red]Неизвестный флаг: {arg}. Используйте --help для справки.[/red]")
                return 2
        else:
            question_parts.append(arg)
            i += 1

    models = _inst_fn()
    if not models:
        if models_mod.LAST_LIST_ERROR is not None:
            return 1
        console.print("[yellow]W1x: первый запуск. Выберите стартовую модель для загрузки:[/yellow]")
        downloaded = _wizard_fn()
        if not downloaded:
            return 1
        models = _inst_fn()
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
        if _interactive_fn is not None:
            return int(_interactive_fn(target_model))
        from .app import _run_interactive_chat
        return _run_interactive_chat(target_model)

    question = " ".join(question_parts).strip() or "Привет!"
    messages: list[ChatMessage] = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        _wait_fn(target_model)
        _, _, interrupted = _gen_fn(target_model, messages, think_mode=False)
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


_run_cli_oneshot = run_cli_oneshot
