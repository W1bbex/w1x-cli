"""Интерактивный каталог моделей Ollama и мастер первичной настройки."""
import shutil
import time
from collections.abc import Sequence
from typing import TypedDict

from prompt_toolkit.formatted_text import HTML
from rich.live import Live
from rich.markup import escape as escape_markup
from rich.panel import Panel
from rich.text import Text

from .config import load_user_config, save_user_config
from .hardware import classify_model_fit, detect_hardware
from .key_input import (
    cooked_terminal_mode,
    poll_key_crossplatform,
    raw_terminal_mode,
)
from .prompts import secondary_session
from .pull_manager import GLOBAL_PULL_MANAGER
from .terminal import console
from .types import HardwareInfo


class _CatalogItem(TypedDict):
    category: str
    name: str
    size_gb: float
    vram_min: float
    note: str


MODEL_CATALOG: list[_CatalogItem] = [
    # ── 1B–4B: Легкие и быстрые (полностью в VRAM) ──────────────────────────
    {
        "category": "light",
        "name": "qwen3.5:4b",
        "size_gb": 3.2,
        "vram_min": 3.5,
        "note": "Qwen 3.5 4B — умная и сбалансированная модель. Отличный универсальный помощник",
    },
    {
        "category": "light",
        "name": "gemma3:4b",
        "size_gb": 3.1,
        "vram_min": 3.5,
        "note": "Google Gemma 3 4B — четкое следование инструкциям, логика и качественный код",
    },
    {
        "category": "light",
        "name": "phi4-mini:latest",
        "size_gb": 2.3,
        "vram_min": 2.8,
        "note": "Microsoft Phi-4 Mini 3.8B — рекорды по математике и STEM-задачам в малом весе",
    },
    {
        "category": "light",
        "name": "llama3.2:3b",
        "size_gb": 1.9,
        "vram_min": 2.5,
        "note": "Meta Llama 3.2 3B — компактная и шустрая, мгновенные ответы на любом ПК",
    },
    {
        "category": "light",
        "name": "ministral-3:3b",
        "size_gb": 2.8,
        "vram_min": 3.2,
        "note": "Mistral AI Ministral 3B — оптимизирована под работу с инструментами и вызовами",
    },
    {
        "category": "light",
        "name": "smollm2:1.7b",
        "size_gb": 1.1,
        "vram_min": 1.8,
        "note": "Hugging Face SmolLM2 1.7B — ультралегкая модель с молниеносной генерацией",
    },
    {
        "category": "light",
        "name": "nemotron-3-nano:4b",
        "size_gb": 2.6,
        "vram_min": 3.2,
        "note": "NVIDIA Nemotron 3 Nano 4B — высокая точность структурирования и синтеза ответов",
    },
    {
        "category": "light",
        "name": "granite4.1:3b",
        "size_gb": 2.0,
        "vram_min": 2.6,
        "note": "IBM Granite 3.1 3B — корпоративная модель для документации и точных фактов",
    },

    # ── 7B–9B: Средние (универсальные рабочие лошадки) ───────────────────────
    {
        "category": "medium",
        "name": "deepseek-r1:7b",
        "size_gb": 4.4,
        "vram_min": 6.3,
        "note": "DeepSeek R1 7B — цепочки рассуждений (CoT). Пошагово решает логические задачи",
    },
    {
        "category": "medium",
        "name": "qwen2.5:7b",
        "size_gb": 4.4,
        "vram_min": 6.6,
        "note": "Alibaba Qwen 2.5 7B — великолепный русский язык, аналитика, эрудиция и диалоги",
    },
    {
        "category": "medium",
        "name": "qwen2.5-coder:7b",
        "size_gb": 4.7,
        "vram_min": 6.6,
        "note": "Qwen 2.5 Coder 7B — лидер среди 7B по написанию кода, рефакторингу и поиску багов",
    },
    {
        "category": "medium",
        "name": "llama3.1:8b",
        "size_gb": 4.7,
        "vram_min": 6.8,
        "note": "Meta Llama 3.1 8B — золотой стандарт открытого ИИ с контекстом до 128k токенов",
    },
    {
        "category": "medium",
        "name": "qwen3.5:9b",
        "size_gb": 6.1,
        "vram_min": 7.5,
        "note": "Qwen 3.5 9B — топовый интеллект в среднем классе, превосходит большинство 7B",
    },
    {
        "category": "medium",
        "name": "mistral:7b",
        "size_gb": 4.1,
        "vram_min": 6.2,
        "note": "Mistral 7B v0.3 — классическая быстрая европейская модель общего назначения",
    },
    {
        "category": "medium",
        "name": "hermes3:8b",
        "size_gb": 4.9,
        "vram_min": 7.0,
        "note": "Nous Hermes 3 8B — лучшая для живого общения, ролевых диалогов и креатива",
    },
    {
        "category": "medium",
        "name": "gemma2:9b",
        "size_gb": 5.5,
        "vram_min": 7.2,
        "note": "Google Gemma 2 9B — сильная аналитическая модель с высокой плотностью знаний",
    },
    {
        "category": "medium",
        "name": "starcoder2:7b",
        "size_gb": 4.1,
        "vram_min": 6.2,
        "note": "BigCode StarCoder2 7B — обучена на 600+ языках программирования и GitHub коде",
    },

    # ── 12B–22B: Продвинутые и профессиональные ──────────────────────────────
    {
        "category": "heavy",
        "name": "mistral-nemo:12b",
        "size_gb": 7.1,
        "vram_min": 9.0,
        "note": "Mistral NeMo 12B (NVIDIA + Mistral) — 128k контекст, отличная многоязычность",
    },
    {
        "category": "heavy",
        "name": "phi4:latest",
        "size_gb": 9.1,
        "vram_min": 10.5,
        "note": "Microsoft Phi-4 14B — передовая модель для сложных математических и научных задач",
    },
    {
        "category": "heavy",
        "name": "qwen2.5:14b",
        "size_gb": 9.0,
        "vram_min": 10.5,
        "note": "Qwen 2.5 14B — профессиональный уровень рассуждений и анализа больших текстов",
    },
    {
        "category": "heavy",
        "name": "qwen2.5-coder:14b",
        "size_gb": 9.0,
        "vram_min": 10.5,
        "note": "Qwen 2.5 Coder 14B — уровень Senior-разработчика: сложные алгоритмы и архитектура",
    },
    {
        "category": "heavy",
        "name": "deepseek-r1:14b",
        "size_gb": 9.0,
        "vram_min": 10.5,
        "note": "DeepSeek R1 14B — глубокие цепочки рассуждений для сложных головоломок и кода",
    },
    {
        "category": "heavy",
        "name": "codestral:22b",
        "size_gb": 13.0,
        "vram_min": 15.0,
        "note": "Mistral Codestral 22B — мощная специализированная модель для программирования",
    },

    # ── 27B–70B: Флагманские тяжеловесы (красная зона) ───────────────────────
    {
        "category": "flagship",
        "name": "qwen2.5:32b",
        "size_gb": 20.0,
        "vram_min": 21.0,
        "note": "Qwen 2.5 32B — флагманское качество (уровень GPT-4o). Требует 24 GB памяти",
    },
    {
        "category": "flagship",
        "name": "deepseek-r1:32b",
        "size_gb": 20.0,
        "vram_min": 21.0,
        "note": "DeepSeek R1 32B — мощнейшая открытая модель для логики и доказательств",
    },
    {
        "category": "flagship",
        "name": "gemma3:27b",
        "size_gb": 17.0,
        "vram_min": 18.5,
        "note": "Google Gemma 3 27B — флагман от Google для анализа огромных документов",
    },
    {
        "category": "flagship",
        "name": "command-r:latest",
        "size_gb": 20.0,
        "vram_min": 22.0,
        "note": "Cohere Command R 35B — чемпион по поиску по документам (RAG) и интеграциям",
    },
    {
        "category": "flagship",
        "name": "llama3.3:70b",
        "size_gb": 43.0,
        "vram_min": 45.0,
        "note": "Meta Llama 3.3 70B — супертяжеловес. Требует 64+ GB RAM или несколько GPU",
    },

    # ── Зрение и Мультимодальность (Vision) ──────────────────────────────────
    {
        "category": "vision",
        "name": "llava:7b",
        "size_gb": 4.7,
        "vram_min": 6.5,
        "note": "LLaVA 1.6 7B — распознавание изображений, чтение скриншотов и графиков",
    },
    {
        "category": "vision",
        "name": "minicpm-v:latest",
        "size_gb": 5.5,
        "vram_min": 7.0,
        "note": "MiniCPM-V 2.6 — высокая детализация OCR, распознает текст с фото и диаграммы",
    },

    # ── Без цензуры (Abliterated) ─────────────────────────────────────────────
    {
        "category": "uncensored",
        "name": "huihui_ai/qwen3.5-abliterated:4b",
        "size_gb": 3.1,
        "vram_min": 3.5,
        "note": "Qwen 3.5 4B Abliterated — сняты корпоративные фильтры, свободные ответы",
    },
    {
        "category": "uncensored",
        "name": "huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M",
        "size_gb": 4.4,
        "vram_min": 6.6,
        "note": "Qwen 2.5 7B Abliterated — 7B без цензуры, прямые ответы на любые темы",
    },
    {
        "category": "uncensored",
        "name": "huihui_ai/deepseek-r1-abliterated:8b",
        "size_gb": 4.9,
        "vram_min": 7.0,
        "note": "DeepSeek R1 8B Abliterated — рассуждения без цензуры и ограничений",
    },

    # ── Ввод вручную ─────────────────────────────────────────────────────────
    {
        "category": "custom",
        "name": "Ввести своё имя модели...",
        "size_gb": 0,
        "vram_min": 0,
        "note": "Любой тег из библиотеки Ollama, например: solar-pro, codellama, dolphin3",
    },
]


def _render_catalog(
    items: Sequence[_CatalogItem],
    cursor: int,
    scroll_offset: int,
    page_size: int,
    hw: HardwareInfo,
    notification: str = "",
) -> Text:
    vram_gb = hw["vram_gb"]
    gpu_name = hw["gpu_name"]
    ram_gb = hw["ram_gb"]
    is_apple_silicon = "Apple Silicon" in gpu_name
    effective_vram = max(0.0, vram_gb - 1.0) if not is_apple_silicon else vram_gb
    installed_set = GLOBAL_PULL_MANAGER.get_installed()
    total = len(items)

    term_cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    is_narrow = term_cols < 96
    name_col_w = 26 if is_narrow else 38
    mem_header = "Память" if is_narrow else "Потребление памяти"
    mem_col_w = 10 if is_narrow else 18
    divider_len = min(86, max(60, term_cols - 2))

    t = Text()

    if is_apple_silicon:
        t.append(f" Платформа: {gpu_name}   Память (Unified): {ram_gb} GB\n", style="dim")
    else:
        t.append(
            f" Видеокарта: {gpu_name}   VRAM: {vram_gb} GB (доступно ~{effective_vram:.0f} GB)   RAM: {ram_gb} GB\n",
            style="dim",
        )
    t.append(" " + chr(0x2500) * divider_len + "\n", style="dim")
    t.append(
        f"  {'№':>2}  {'Модель':<{name_col_w}}  {'Вес':>8}   {mem_header:>{mem_col_w}}   {'Статус':<15} [{cursor+1}/{total}]\n",
        style="dim",
    )
    t.append(" " + chr(0x2500) * divider_len + "\n", style="dim")

    if scroll_offset > 0:
        t.append(f"      ▲  ... еще {scroll_offset} выше (листайте ↑) ...\n", style="dim cyan")

    visible_end = min(total, scroll_offset + page_size)
    for i in range(scroll_offset, visible_end):
        item = items[i]
        name = item["name"]
        size = item["size_gb"]
        vram_min = item["vram_min"]
        is_cursor = i == cursor
        is_custom = item.get("category") == "custom"
        fit = classify_model_fit(vram_min, size, hw)

        idx_str = f"{i+1:>2}."
        disp_name = name if len(name) <= name_col_w else (name[: name_col_w - 3] + "...")
        size_str = f"{size:.1f} GB" if size > 0 else "   --   "
        vram_str = f"~{vram_min:.1f} GB" if vram_min > 0 else "    --    "

        snap = GLOBAL_PULL_MANAGER.get_task_snapshot(name)
        is_installed = name in installed_set or (snap is not None and snap["status"] == "completed")
        is_downloading = snap is not None and snap["status"] == "downloading"
        is_queued = snap is not None and snap["status"] == "queued"

        if snap is not None and snap["status"] == "downloading":
            status_text = f"Загрузка {snap['percentage']:>2.0f}% ({snap['speed']})"
            status_style = "bold cyan"
        elif is_queued:
            pos = GLOBAL_PULL_MANAGER.get_queue_position(name)
            status_text = f"В очереди #{pos}" if pos > 0 else "В очереди"
            status_style = "bold yellow"
        elif is_installed:
            status_text = "Установлена"
            status_style = "bold green"
        elif is_custom:
            status_text = "Ввести тег"
            status_style = "dim"
        elif fit == "green":
            status_text = "Легко (VRAM)" if not is_apple_silicon else "Легко (Memory)"
            status_style = "green"
        elif fit == "orange":
            status_text = "В RAM (медленнее)"
            status_style = "rgb(255,170,0)"
        else:
            status_text = "Не потянет"
            status_style = "bold red"

        row_content = f"  {idx_str} {disp_name:<{name_col_w}} {size_str:>8}   {vram_str:>{mem_col_w}}   "

        if is_cursor:
            t.append(row_content, style="bold black on bright_white")
            t.append(f"{status_text:<18}\n", style=f"{status_style} on bright_white")
            t.append(f"       ▸ {item['note']}\n", style="cyan")
            if fit == "orange" and not is_installed and not is_downloading and not is_queued:
                t.append(
                    f"       • Потребует ~{vram_min:.1f} GB (VRAM доступно ~{effective_vram:.0f} GB) — часть уйдет в оперативную память (RAM)\n",
                    style="rgb(255,170,0)",
                )
            elif fit == "red" and not is_installed and not is_downloading and not is_queued:
                target_hw_label = "на Mac доступно ~{ram_gb:.0f} GB объединённой памяти" if is_apple_silicon else "на ПК доступно ~{ram_gb:.0f} GB памяти"
                t.append(
                    f"       ! Не рекомендуется: модели нужно ~{vram_min:.1f} GB, {target_hw_label.format(ram_gb=ram_gb)}\n",
                    style="bold red",
                )
        else:
            t.append(row_content)
            t.append(f"{status_text:<18}\n", style=status_style)

    if visible_end < total:
        t.append(f"      ▼  ... еще {total - visible_end} ниже (листайте ↓) ...\n", style="dim cyan")

    t.append(" " + chr(0x2500) * divider_len + "\n", style="dim")

    active_cnt, pending_cnt = GLOBAL_PULL_MANAGER.active_and_pending_count()
    if active_cnt > 0 or pending_cnt > 0:
        t.append(f" ▶ Загрузка: {active_cnt} качается  |  В очереди: {pending_cnt}   ", style="bold cyan")

    if notification:
        t.append(f" {notification}\n", style="bold yellow")
    else:
        t.append("\n")

    hint_line = (
        " ↑↓/PgUp/PgDn — выбор   Enter — скачать   q/Esc — выход\n"
        if is_narrow
        else " Стрелки ↑↓ / PgUp PgDn — выбор       Enter — скачать / в очередь       q / Esc — выход\n"
    )
    t.append(hint_line, style="dim")
    return t


def _enqueue_status_message(res: str, model_name: str) -> str:
    if res == "already_installed":
        return f"✓ {model_name} уже установлена"
    if res == "already_in_progress":
        return f"⏳ {model_name} уже качается или в очереди"
    if res == "already_completed":
        return f"✓ {model_name} уже скачана"
    if res == "enqueued":
        return f"▶ {model_name} добавлена в очередь загрузки"
    return ""


def interactive_model_catalog() -> str | None:
    """Отобразить интерактивный каталог моделей с прокруткой и очередью загрузок."""
    hw = detect_hardware()
    GLOBAL_PULL_MANAGER.refresh_installed()
    items = MODEL_CATALOG
    cursor = 0
    scroll_offset = 0
    total = len(items)
    notification = ""
    last_selected_model = None

    term_height = shutil.get_terminal_size((80, 25)).lines
    page_size = max(2, min(12, term_height - 11))

    console.print()
    last_render = time.monotonic()
    with Live(
        _render_catalog(items, cursor, scroll_offset, page_size, hw, notification),
        console=console,
        auto_refresh=False,
        transient=True,
    ) as live:
        with raw_terminal_mode():
            while True:
                action = poll_key_crossplatform()
                needs_render = False
                if action:
                    notification = ""
                    needs_render = True
                    if action == "UP":
                        cursor = (cursor - 1) % total
                    elif action == "DOWN":
                        cursor = (cursor + 1) % total
                    elif action == "PGUP":
                        cursor = max(0, cursor - page_size)
                    elif action == "PGDN":
                        cursor = min(total - 1, cursor + page_size)
                    elif action == "HOME":
                        cursor = 0
                    elif action == "END":
                        cursor = total - 1
                    elif action == "ENTER":
                        chosen = items[cursor]
                        if chosen.get("category") == "custom":
                            live.stop()
                            with cooked_terminal_mode():
                                try:
                                    raw = secondary_session.prompt(
                                        HTML("<b><ansigreen>Тег модели из Ollama › </ansigreen></b>")
                                    ).strip()
                                except (KeyboardInterrupt, EOFError):
                                    raw = ""
                            if raw:
                                res = GLOBAL_PULL_MANAGER.enqueue(raw)
                                if res != "empty":
                                    last_selected_model = raw
                                notification = _enqueue_status_message(res, raw)
                            live.start()
                        else:
                            model_name = chosen["name"]
                            fit = classify_model_fit(chosen["vram_min"], chosen["size_gb"], hw)

                            if fit == "red":
                                live.stop()
                                console.print()
                                is_apple = "Apple" in hw.get("gpu_name", "")
                                hw_desc = (
                                    f"На вашем Mac доступно ~{hw['ram_gb']:.0f} GB объединённой памяти (Unified Memory).\n"
                                    if is_apple
                                    else f"На вашем ПК доступно ~{hw['vram_gb']:.0f} GB VRAM + {hw['ram_gb']:.0f} GB RAM.\n"
                                )
                                console.print(
                                    f"[bold red]⚠️ Внимание:[/bold red] Модель [bold]{escape_markup(model_name)}[/bold] требует ~{chosen['vram_min']:.1f} GB памяти.\n"
                                    f"{hw_desc}"
                                    "[yellow]Модель может вызвать зависание системы или не запуститься.[/yellow]"
                                )
                                with cooked_terminal_mode():
                                    try:
                                        ans = secondary_session.prompt(
                                            HTML("<b><ansired>Всё равно скачать? (y/n) › </ansired></b>")
                                        ).strip().lower()
                                    except (KeyboardInterrupt, EOFError):
                                        ans = "n"
                                live.start()
                                if ans not in ("y", "yes", "д", "да"):
                                    notification = f"Загрузка {model_name} отменена"
                                    live.update(
                                        _render_catalog(items, cursor, scroll_offset, page_size, hw, notification),
                                        refresh=True,
                                    )
                                    last_render = time.monotonic()
                                    continue

                            last_selected_model = model_name
                            res = GLOBAL_PULL_MANAGER.enqueue(model_name)
                            notification = _enqueue_status_message(res, model_name)

                    elif action == "EXIT":
                        break
                    elif action.isascii() and action.isdigit():
                        n = int(action)
                        if 1 <= n <= total:
                            cursor = n - 1

                if cursor < scroll_offset:
                    scroll_offset = cursor
                elif cursor >= scroll_offset + page_size:
                    scroll_offset = cursor - page_size + 1

                now = time.monotonic()
                active_cnt, pending_cnt = GLOBAL_PULL_MANAGER.active_and_pending_count()
                if needs_render or ((active_cnt > 0 or pending_cnt > 0) and (now - last_render) >= 0.25):
                    live.update(
                        _render_catalog(items, cursor, scroll_offset, page_size, hw, notification),
                        refresh=True,
                    )
                    last_render = now

                time.sleep(0.04)

    return last_selected_model


def wait_for_download_queue() -> None:
    """Дождаться завершения всех активных и запланированных загрузок в очереди."""
    active_cnt, pending_cnt = GLOBAL_PULL_MANAGER.active_and_pending_count()
    if active_cnt == 0 and pending_cnt == 0:
        return

    console.print(
        "\n[cyan]Ожидание завершения загрузок из каталога... (Ctrl+C — отменить ожидание фоновых задач)[/cyan]"
    )
    try:
        with Live(console=console, auto_refresh=False, transient=True) as live:
            while True:
                active_cnt, pending_cnt = GLOBAL_PULL_MANAGER.active_and_pending_count()
                if active_cnt == 0 and pending_cnt == 0:
                    break
                active_str_parts = []
                for name, task in list(GLOBAL_PULL_MANAGER.tasks.items()):
                    if task.status == "downloading":
                        active_str_parts.append(f"{name}: {task.percentage:>2.0f}% ({task.speed})")
                line_text = " | ".join(active_str_parts) if active_str_parts else "Подготовка..."
                if pending_cnt > 0:
                    line_text += f" (+{pending_cnt} в очереди)"
                live.update(Text(f"  ▶ {line_text}", style="cyan"), refresh=True)
                time.sleep(0.25)
    except KeyboardInterrupt:
        console.print("\n[yellow]Ожидание завершения загрузок прервано. Загрузки продолжаются в фоне.[/yellow]")
    finally:
        GLOBAL_PULL_MANAGER.refresh_installed(force=True)


def first_time_setup_wizard() -> str | None:
    """Мастер первичной загрузки модели при отсутствии установленных моделей."""
    console.print()
    console.print(
        Panel(
            "[bold]Добро пожаловать в W1x![/bold]\n"
            "В вашей Ollama пока нет моделей.\n"
            "Выберите стартовую модель -- W1x скачает её и сохранит как модель по умолчанию.",
            title="Первый запуск W1x",
            border_style="green",
            padding=(1, 2),
        )
    )

    selected = interactive_model_catalog()
    if not selected:
        return None

    snap = GLOBAL_PULL_MANAGER.get_task_snapshot(selected)
    if snap and snap["status"] in ("queued", "downloading"):
        console.print(
            f"\n[cyan]Ожидание завершения загрузки {escape_markup(selected)}... (Ctrl+C — отменить ожидание)[/cyan]"
        )
        try:
            with Live(console=console, auto_refresh=False, transient=True) as live:
                while True:
                    snap = GLOBAL_PULL_MANAGER.get_task_snapshot(selected)
                    if snap is None or snap["status"] not in ("queued", "downloading"):
                        break
                    live.update(
                        Text(
                            f"  ▶ {selected}: {snap['percentage']:>3.0f}% ({snap['speed']}) — {snap['status_msg']}",
                            style="cyan",
                        ),
                        refresh=True,
                    )
                    time.sleep(0.25)
        except KeyboardInterrupt:
            console.print("\n[yellow]Ожидание загрузки прервано пользователем.[/yellow]")
            return None

    snap = GLOBAL_PULL_MANAGER.get_task_snapshot(selected)
    if snap and snap["status"] == "error":
        console.print(
            Panel(
                f"[bold red]Ошибка загрузки {escape_markup(selected)}: {escape_markup(snap['error_msg'] or 'Сбой соединения')}[/bold red]\n"
                "Модель не была установлена как модель по умолчанию.\n"
                "[dim]Попробуйте скачать повторно через /pull[/dim]",
                title="Ошибка загрузки",
                border_style="red",
                padding=(1, 2),
            )
        )
        return None

    cfg = load_user_config()
    cfg["default_model"] = selected
    if not save_user_config(cfg):
        console.print("[yellow]Предупреждение: модель скачана, но не удалось сохранить её как модель по умолчанию.[/yellow]")
    console.print(f"[bold green]✓ Стартовая модель {escape_markup(selected)} готова к работе![/bold green]")
    return selected
