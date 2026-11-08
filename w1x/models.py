# -*- coding: utf-8 -*-
import os
import sys
import time
import threading
from typing import Optional
from ollama import generate, list as ollama_list, ps
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

from .config import (
    EXIT_COMMANDS,
    DATA_DIR,
    load_user_config,
    save_user_config,
    get_keep_alive,
    get_context_tokens,
)
from .terminal import (
    console,
    session,
    run_with_spinner,
    HTML,
)
from .puller import first_time_setup_wizard, pull_model, GLOBAL_PULL_QUEUE


def _start_global_ctrl_q_listener():
    """Фоновый перехватчик Ctrl+Q для мгновенного снятия всех процессов и выгрузки VRAM."""
    if os.name != "nt":
        return

    import ctypes

    def _listener():
        user32 = ctypes.windll.user32
        while True:
            time.sleep(0.04)
            try:
                # Только если окно консоли находится на переднем плане
                kernel32 = ctypes.windll.kernel32
                console_hwnd = kernel32.GetConsoleWindow()
                if console_hwnd and user32.GetForegroundWindow() != console_hwnd:
                    continue
                # 0x11: VK_CONTROL, 0x51: VK_Q
                ctrl_down = bool(user32.GetAsyncKeyState(0x11) & 0x8000)
                q_down = bool(user32.GetAsyncKeyState(0x51) & 0x8000)
                if ctrl_down and q_down:
                    sys.stdout.write("\n\n[Мгновенный выход (Ctrl+Q). Выгружаю модели из VRAM...]\n")
                    sys.stdout.flush()
                    try:
                        unload_all_models()
                    except Exception:
                        pass
                    os._exit(0)
            except Exception:
                pass

    t = threading.Thread(target=_listener, daemon=True, name="CtrlQGlobalListener")
    t.start()


_start_global_ctrl_q_listener()


def model_name(item):
    return getattr(item, "model", None) or (item.get("name") if isinstance(item, dict) else getattr(item, "name", None))


def installed_models():
    try:
        names = [model_name(item) for item in ollama_list().models]
        return [name for name in names if name]
    except Exception as error:
        err_str = str(error).lower()
        if any(w in err_str for w in ("connection", "connect", "refused", "111", "10061")):
            console.print(
                Panel(
                    "[bold red]❌ Не удалось подключиться к серверу Ollama.[/bold red]\n\n"
                    "• Убедитесь, что [bold green]Ollama[/bold green] запущена в Windows (значок в трее рядом с часами).\n"
                    "• Если Ollama ещё не установлена — скачайте её с официального сайта: [bold cyan]https://ollama.com[/bold cyan]",
                    title="[bold]Сервер Ollama недоступен[/bold]",
                    border_style="red",
                    padding=(1, 2),
                )
            )
        else:
            console.print(f"[red]Не удалось получить список моделей: {error}[/red]")
        return []


def running_models():
    try:
        return [
            name
            for item in ps().models
            if (name := model_name(item))
        ]
    except Exception:
        return []


def unload_all_models():
    unloaded = []
    for name in running_models():
        try:
            generate(model=name, prompt="", keep_alive=0)
            unloaded.append(name)
        except Exception:
            pass
    return unloaded


if os.name == "nt":
    try:
        import ctypes
        from ctypes import wintypes

        _PHANDLER_ROUTINE = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def _console_ctrl_handler(ctrl_type):
            if ctrl_type in {2, 5, 6}:
                try:
                    unload_all_models()
                except Exception:
                    pass
            return False

        _handler = _PHANDLER_ROUTINE(_console_ctrl_handler)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_handler, True)
    except Exception:
        pass


def reset_all_models(show_message: bool = False):
    return unload_all_models()


class PreloadTask:
    def __init__(self, name: str):
        self.name = name
        self.event = threading.Event()
        self.error = None
        self.thread = None


_CURRENT_PRELOAD: Optional[PreloadTask] = None
_PRELOAD_LOCK = threading.Lock()


def _background_preload_worker(task: PreloadTask):
    try:
        generate(
            model=task.name,
            prompt="",
            keep_alive=get_keep_alive(),
            options={"num_ctx": get_context_tokens()},
        )
    except Exception as e:
        task.error = e
    finally:
        task.event.set()


def start_background_preload(name: str):
    global _CURRENT_PRELOAD
    with _PRELOAD_LOCK:
        if _CURRENT_PRELOAD is not None and _CURRENT_PRELOAD.name == name and not _CURRENT_PRELOAD.event.is_set():
            return
        task = PreloadTask(name)
        task.thread = threading.Thread(
            target=_background_preload_worker,
            args=(task,),
            daemon=True,
            name=f"Preload-{name}",
        )
        _CURRENT_PRELOAD = task
        task.thread.start()


def wait_for_model_ready(name=None):
    """Подождать готовность модели перед отправкой первого запроса (если она ещё догружается в фоне)."""
    global _CURRENT_PRELOAD
    with _PRELOAD_LOCK:
        task = _CURRENT_PRELOAD
    if task is None:
        return
    if name and task.name != name:
        return

    if task.event.is_set():
        if task.error:
            with _PRELOAD_LOCK:
                _CURRENT_PRELOAD = None
            raise task.error
        return

    run_with_spinner(
        f"Догружаю {name or task.name or 'модель'} в память...",
        task.event.wait,
    )
    if task.error:
        with _PRELOAD_LOCK:
            _CURRENT_PRELOAD = None
        raise task.error


def preload_model(name):
    run_with_spinner(
        f"Загружаю {name}...",
        lambda: generate(
            model=name,
            prompt="",
            keep_alive=get_keep_alive(),
            options={"num_ctx": get_context_tokens()},
        ),
    )
    console.print(f"[green]✓[/green] [bold]{name}[/bold] готова.")


def get_last_model(models):
    """Получить уже загруженную в VRAM модель, либо модель из настроек."""
    if not models:
        return None

    # 1. Если прямо сейчас в памяти VRAM уже есть загруженная модель — используем её в первую очередь!
    try:
        active = running_models()
        if active:
            for act in active:
                for m in models:
                    if act == m or act.split(":")[0] == m.split(":")[0]:
                        return m
    except Exception:
        pass

    # 2. Иначе читаем модель по умолчанию из настроек пользователя
    cfg = load_user_config()
    saved = cfg.get("default_model")
    if saved in models:
        return saved

    # 3. Иначе ищем легковесную быструю модель
    for pref in ("qwen3.5:4b", "qwen2.5:7b", "qwen2.5:3b", "llama3.2:3b"):
        for m in models:
            if pref in m:
                return m
    return models[0]


def save_last_model(name):
    cfg = load_user_config()
    cfg["default_model"] = name
    save_user_config(cfg)


def render_models_panel(models, selected_idx, active_in_vram):
    """Отрисовать панель моделей с подсветкой текущего курсора (черным по белому)."""
    lines = []
    total_items = len(models) + 1

    for i in range(total_items):
        is_cursor = (i == selected_idx)
        if i < len(models):
            name = models[i]
            is_warm = any(name == act or name.split(":")[0] == act.split(":")[0] for act in active_in_vram)
            num_str = f"{i+1:2d}."
            
            tag_str = "  [VRAM]" if is_warm else ""

            if is_cursor:
                line_content = f" > {num_str} {name:<35} {tag_str}  [ВЫБРАНА - Enter]"
                lines.append(f"[bold black on bright_white]{line_content:<78}[/bold black on bright_white]")
            else:
                lines.append(f"   [cyan]{num_str}[/cyan] [white]{name:<35}[/white] [dim magenta]{tag_str}[/dim magenta]")
        else:
            num_str = " +."
            label = "+ Скачать новую модель из библиотеки Ollama (Тир-Лист)"
            if is_cursor:
                line_content = f" > {num_str} {label}"
                lines.append(f"\n[bold black on bright_cyan]{line_content:<78}[/bold black on bright_cyan]")
            else:
                lines.append(f"\n   [cyan]{num_str}[/cyan] [dim cyan]{label}[/dim cyan]")

    return Panel(
        "\n".join(lines),
        title="[bold]Установленные модели Ollama[/bold]",
        subtitle="[dim]↑ / ↓ — выбор стрелками  •  Enter — подтвердить  •  q — выход[/dim]",
        border_style="green",
        padding=(0, 1),
    )


def print_models(models, default_model=None):
    active_in_vram = set()
    try:
        active_in_vram = set(running_models())
    except Exception:
        pass
    idx = models.index(default_model) if default_model in models else 0
    console.print(render_models_panel(models, idx, active_in_vram))


def choose_model(current=None):
    models = installed_models()

    if not models:
        # Если ни одной модели нет — запускаем приветственный мастер скачивания
        new_model = first_time_setup_wizard()
        if new_model:
            models = installed_models()
        if not models:
            console.print("[yellow]Модель не выбрана.[/yellow]")
            return None

    default_model = current or get_last_model(models)
    selected_idx = models.index(default_model) if default_model in models else 0
    total_items = len(models) + 1

    active_in_vram = set()
    try:
        active_in_vram = set(running_models())
    except Exception:
        pass

    # Интерактивный выбор по стрелкам (Windows / macOS / Linux)
    from .puller import poll_key_crossplatform
    console.print()
    with Live(render_models_panel(models, selected_idx, active_in_vram), console=console, auto_refresh=False, transient=True) as live:
        while True:
            live.update(render_models_panel(models, selected_idx, active_in_vram), refresh=True)
            action = poll_key_crossplatform()
            if action:
                if action == "UP":
                    selected_idx = (selected_idx - 1) % total_items
                elif action == "DOWN":
                    selected_idx = (selected_idx + 1) % total_items
                elif action == "ENTER":
                    break
                elif action == "EXIT":
                    return None
                elif action.isdigit():
                    num = int(action)
                    if 1 <= num <= len(models):
                        selected_idx = num - 1
                        break
            time.sleep(0.04)

    # Если выбрали пункт скачивания новой модели:
    if selected_idx == len(models):
        downloaded = first_time_setup_wizard()
        if downloaded:
            save_last_model(downloaded)
            return downloaded
        return default_model

    chosen = models[selected_idx]
    save_last_model(chosen)
    console.print(render_models_panel(models, selected_idx, active_in_vram))
    return chosen


def interactive_config_menu():
    """Интерактивное меню настройки W1x CLI."""
    models = installed_models()
    while True:
        cfg = load_user_config()
        current_m = cfg.get("default_model", "не выбрана")
        keep_mins = cfg.get("keep_alive_minutes", 30)
        ctx_tokens = cfg.get("context_tokens", 8192)
        stats = cfg.get("stats_enabled", False)

        console.print()
        console.print(
            Panel(
                f"[bold cyan]1.[/bold cyan] Модель по умолчанию: [bold green]{current_m}[/bold green]\n"
                f"[bold cyan]2.[/bold cyan] Время удержания в VRAM: [bold yellow]{keep_mins} мин.[/bold yellow] (0 = выгружать сразу, 999 = держать всегда)\n"
                f"[bold cyan]3.[/bold cyan] Размер контекста (num_ctx): [bold magenta]{ctx_tokens} токенов[/bold magenta] (~{int(ctx_tokens*3.5)} симв.)\n"
                f"[bold cyan]4.[/bold cyan] Статистика скорости ответов: [bold]{'Включена' if stats else 'Выключена'}[/bold]\n"
                f"[bold cyan]5.[/bold cyan] Скачать новую модель из библиотеки Ollama\n"
                f"[bold cyan]6.[/bold cyan] Выгрузить все модели из VRAM прямо сейчас\n"
                "[bold cyan]0.[/bold cyan] Выйти из настроек (Enter)",
                title="[bold]⚙️ Настройки W1x CLI[/bold]",
                border_style="cyan",
                padding=(0, 1),
            )
        )

        try:
            choice = session.prompt(
                HTML("<b><ansigreen>Настройка [0] › </ansigreen></b>")
            ).strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not choice or choice in ("0", "q", "exit"):
            break

        if choice == "1":
            console.print()
            new_m = choose_model(current=current_m)
            if new_m:
                cfg["default_model"] = new_m
                save_user_config(cfg)
                console.print(f"[green]✓ Модель по умолчанию изменена на {new_m}[/green]")

        elif choice == "2":
            console.print()
            val = session.prompt(
                HTML(f"<b><ansigreen>Минут удержания в VRAM [{keep_mins}] › </ansigreen></b>")
            ).strip()
            if val.isdigit():
                cfg["keep_alive_minutes"] = int(val)
                save_user_config(cfg)
                console.print(f"[green]✓ Удержание в VRAM установлено: {val} мин.[/green]")

        elif choice == "3":
            console.print()
            console.print("[dim]Популярные размеры: 2048 (минимум), 4096, 8192 (базовый), 16384 (большой), 32768 (макс)[/dim]")
            val = session.prompt(
                HTML(f"<b><ansigreen>Размер контекста (токенов) [{ctx_tokens}] › </ansigreen></b>")
            ).strip()
            if val.isdigit() and int(val) >= 512:
                cfg["context_tokens"] = int(val)
                save_user_config(cfg)
                console.print(f"[green]✓ Размер контекста установлен: {val} токенов (~{int(int(val)*3.5)} симв.)[/green]")

        elif choice == "4":
            cfg["stats_enabled"] = not stats
            save_user_config(cfg)
            state = "включена" if not stats else "выключена"
            console.print(f"[green]✓ Статистика {state}.[/green]")

        elif choice == "5":
            downloaded = first_time_setup_wizard()
            if downloaded:
                cfg["default_model"] = downloaded
                save_user_config(cfg)

        elif choice == "6":
            unloaded = unload_all_models()
            msg = ", ".join(unloaded) if unloaded else "нет активных моделей"
            console.print(f"[green]✓ Модели выгружены из VRAM ({msg}).[/green]")
