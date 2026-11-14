"""Интерактивные меню выбора модели и конфигурации W1x CLI."""
import time
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from typing import Any

from rich.live import Live
from rich.panel import Panel

from .catalog_ui import first_time_setup_wizard
from .config import load_user_config, save_user_config
from .models import (
    _normalize_model_tag,
    get_last_model,
    installed_models,
    running_models,
    save_last_model,
    unload_all_models,
)
from .terminal import HTML, console, poll_key_crossplatform, secondary_session


def render_models_panel(
    models: Sequence[str],
    selected_idx: int,
    active_in_vram: AbstractSet[str],
) -> Panel:
    """Отрисовать панель моделей с подсветкой текущего курсора (чёрным по белому)."""
    lines: list[str] = []
    total_items = len(models) + 1
    active_exact = set(active_in_vram)
    active_norm = {_normalize_model_tag(act) for act in active_in_vram}

    for i in range(total_items):
        is_cursor = i == selected_idx
        if i < len(models):
            name = models[i]
            is_warm = (name in active_exact) or (_normalize_model_tag(name) in active_norm)
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


def print_models(models: Sequence[str], default_model: str | None = None) -> None:
    """Вывести список установленных моделей и пометить модели в VRAM."""
    active_in_vram: set[str] = set()
    try:
        active_in_vram = set(running_models())
    except Exception:
        pass
    idx = models.index(default_model) if default_model in models else 0
    console.print(render_models_panel(models, idx, active_in_vram))


def choose_model(current: str | None = None) -> str | None:
    """Интерактивно выбрать активную модель или запустить загрузку новой."""
    models = installed_models()

    if not models:
        new_model = first_time_setup_wizard()
        if new_model:
            models = installed_models()
        if not models:
            console.print("[yellow]Модель не выбрана.[/yellow]")
            return None

    default_model = current or get_last_model(models)
    selected_idx = models.index(default_model) if default_model in models else 0
    total_items = len(models) + 1

    active_in_vram: set[str] = set()
    try:
        active_in_vram = set(running_models())
    except Exception:
        pass

    console.print()
    with Live(
        render_models_panel(models, selected_idx, active_in_vram),
        console=console,
        auto_refresh=False,
        transient=True,
    ) as live:
        while True:
            action = poll_key_crossplatform()
            if action:
                prev_idx = selected_idx
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
                if selected_idx != prev_idx:
                    live.update(render_models_panel(models, selected_idx, active_in_vram), refresh=True)
            time.sleep(0.04)

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


def interactive_config_menu() -> None:
    """Интерактивное меню настройки W1x CLI."""
    while True:
        cfg: dict[str, Any] = dict(load_user_config())
        current_m = str(cfg.get("default_model", "не выбрана"))
        keep_mins = int(cfg.get("keep_alive_minutes", 30))
        ctx_tokens = int(cfg.get("context_tokens", 8192))
        stats = bool(cfg.get("stats_enabled", False))

        console.print()
        console.print(
            Panel(
                f"[bold cyan]1.[/bold cyan] Модель по умолчанию: [bold green]{current_m}[/bold green]\n"
                f"[bold cyan]2.[/bold cyan] Удержание в VRAM (Keep-Alive): [bold yellow]{keep_mins} мин[/bold yellow] [dim](0 = мгновенная выгрузка)[/dim]\n"
                f"[bold cyan]3.[/bold cyan] Размер контекста (num_ctx): [bold magenta]{ctx_tokens} токенов[/bold magenta] (~{int(ctx_tokens*3.5)} симв.)\n"
                f"[bold cyan]4.[/bold cyan] Статистика скорости ответов: [bold]{'Включена' if stats else 'Выключена'}[/bold]\n"
                "[bold cyan]5.[/bold cyan] Скачать новую модель из библиотеки Ollama\n"
                "[bold cyan]6.[/bold cyan] Выгрузить все модели из VRAM прямо сейчас\n"
                "[bold cyan]0.[/bold cyan] Выйти из настроек (Enter)",
                title="[bold]⚙️ Настройки W1x CLI[/bold]",
                border_style="cyan",
                padding=(0, 1),
            )
        )

        try:
            choice = secondary_session.prompt(
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
            try:
                val = secondary_session.prompt(
                    HTML(f"<b><ansigreen>Минут удержания в VRAM [{keep_mins}] › </ansigreen></b>")
                ).strip()
            except (KeyboardInterrupt, EOFError):
                continue
            if val.isdigit():
                cfg["keep_alive_minutes"] = int(val)
                save_user_config(cfg)
                console.print(f"[green]✓ Удержание в VRAM установлено: {val} мин.[/green]")

        elif choice == "3":
            console.print()
            console.print("[dim]Популярные размеры: 2048 (минимум), 4096, 8192 (базовый), 16384 (большой), 32768 (макс)[/dim]")
            try:
                val = secondary_session.prompt(
                    HTML(f"<b><ansigreen>Размер контекста (токенов) [{ctx_tokens}] › </ansigreen></b>")
                ).strip()
            except (KeyboardInterrupt, EOFError):
                continue
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
