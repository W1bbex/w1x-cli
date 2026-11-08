# -*- coding: utf-8 -*-
"""W1x: Кроссплатформенный каталог моделей Ollama (Windows, macOS, Linux) с окном прокрутки (viewport), фоновой загрузкой и адаптацией под GPU/Apple Silicon/RAM."""
import os
import sys
import time
import shutil
import ctypes
import subprocess
import threading
from typing import List, Optional, Dict, Any
from ollama import pull

from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn,
)
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

from .terminal import console, session, HTML


# ---------------------------------------------------------------------------
# Определение параметров железа (Windows / macOS Apple Silicon / Linux)
# ---------------------------------------------------------------------------

def detect_hardware() -> Dict[str, Any]:
    hw = {"gpu_name": "Не определена", "vram_gb": 0.0, "ram_gb": 8.0}

    # 1. Определение оперативной памяти (RAM)
    if sys.platform == "win32":
        try:
            class _MEM(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("sullAvailExtendedVirtual", ctypes.c_uint64),
                ]
            stat = _MEM()
            stat.dwLength = ctypes.sizeof(_MEM)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                hw["ram_gb"] = round(stat.ullTotalPhys / (1024 ** 3), 1)
        except Exception:
            pass

    elif sys.platform == "darwin":  # macOS (Apple Silicon Unified Memory)
        try:
            mem_raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL).strip()
            if mem_raw.isdigit():
                hw["ram_gb"] = round(int(mem_raw) / (1024 ** 3), 1)
                # На Mac с Apple Silicon видеопамять объединена с оперативной (Unified Memory)
                hw["gpu_name"] = "Apple Silicon (Unified Memory)"
                hw["vram_gb"] = hw["ram_gb"]
                return hw
        except Exception:
            pass

    else:  # Linux
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        hw["ram_gb"] = round(kb / (1024 ** 2), 1)
                        break
        except Exception:
            pass

    # 2. Определение дискретной GPU через nvidia-smi (Windows & Linux)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        ).strip()
        if out:
            gpu_name, vram_mb = [x.strip() for x in out.splitlines()[0].split(",")]
            hw["gpu_name"] = gpu_name
            hw["vram_gb"] = round(float(vram_mb) / 1024, 1)
    except Exception:
        pass

    return hw


# ---------------------------------------------------------------------------
# Каталог моделей Ollama
# ---------------------------------------------------------------------------

MODEL_CATALOG = [
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


def classify_model_fit(vram_min: float, size_gb: float, hw: Dict[str, Any]) -> str:
    """Определяет категорию нагрузки: green (легко), orange (в RAM), red (не потянет)."""
    if size_gb <= 0 or vram_min <= 0:
        return "custom"

    vram_gb = hw.get("vram_gb", 0.0)
    ram_gb = hw.get("ram_gb", 8.0)

    # На Apple Silicon вся память является общей (Unified Memory)
    if "Apple Silicon" in hw.get("gpu_name", ""):
        eff_unified = max(0.0, ram_gb - 2.5)
        if vram_min <= eff_unified:
            return "green"
        return "red"

    eff_vram = max(0.0, vram_gb - 1.0)
    eff_total = eff_vram + max(0.0, ram_gb - 3.5)

    if eff_vram > 0 and vram_min <= eff_vram:
        return "green"
    elif vram_min <= eff_total:
        return "orange"
    else:
        return "red"


# ---------------------------------------------------------------------------
# Фоновый менеджер загрузки и очереди
# ---------------------------------------------------------------------------

class DownloadTask:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.status = "queued"
        self.completed_bytes = 0
        self.total_bytes = 0
        self.percentage = 0.0
        self.speed = "0.0 MB/s"
        self.status_msg = "В очереди"
        self.error_msg = ""

    @property
    def error(self) -> str:
        return self.error_msg


class AsyncPullManager:
    def __init__(self):
        self.tasks: Dict[str, DownloadTask] = {}
        self.queue: List[str] = []
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._installed_cache: Optional[set] = None

    def get_installed(self) -> set:
        if self._installed_cache is None:
            try:
                from .models import installed_models
                self._installed_cache = set(installed_models())
            except Exception:
                self._installed_cache = set()
        return self._installed_cache

    def refresh_installed(self):
        try:
            from .models import installed_models
            self._installed_cache = set(installed_models())
        except Exception:
            pass

    def enqueue(self, model_name: str) -> str:
        clean = model_name.strip()
        if not clean:
            return "empty"
        with self._lock:
            if clean in self.get_installed():
                return "already_installed"
            if clean in self.tasks:
                task = self.tasks[clean]
                if task.status in ("queued", "downloading"):
                    return "already_in_progress"
                if task.status == "completed":
                    return "already_completed"

            task = DownloadTask(clean)
            self.tasks[clean] = task
            self.queue.append(clean)
            self._ensure_worker()
            return "enqueued"

    def get_task(self, model_name: str) -> Optional[DownloadTask]:
        with self._lock:
            return self.tasks.get(model_name)

    def get_queue_position(self, model_name: str) -> int:
        with self._lock:
            if model_name in self.queue:
                return self.queue.index(model_name) + 1
            return 0

    def active_and_pending_count(self) -> tuple[int, int]:
        with self._lock:
            active = sum(1 for t in self.tasks.values() if t.status == "downloading")
            pending = sum(1 for t in self.tasks.values() if t.status == "queued")
            return active, pending

    def _ensure_worker(self):
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

    def _worker_loop(self):
        while True:
            current_model = None
            with self._lock:
                if self.queue:
                    current_model = self.queue.pop(0)
                else:
                    self._worker_thread = None
                    break

            with self._lock:
                task = self.tasks.get(current_model)
                if not task:
                    continue
                task.status = "downloading"
                task.status_msg = "Подключение..."

            last_calc_time = time.time()
            last_bytes = 0

            try:
                for chunk in pull(current_model, stream=True):
                    status = getattr(chunk, "status", "") or (chunk.get("status", "") if isinstance(chunk, dict) else "")
                    completed = getattr(chunk, "completed", 0) or (chunk.get("completed", 0) if isinstance(chunk, dict) else 0)
                    total = getattr(chunk, "total", 0) or (chunk.get("total", 0) if isinstance(chunk, dict) else 0)

                    now = time.time()
                    dt = now - last_calc_time
                    if dt >= 0.4:
                        speed_bps = (completed - last_bytes) / dt if dt > 0 else 0
                        speed_mb = speed_bps / (1024 * 1024)
                        with self._lock:
                            task.speed = f"{speed_mb:.1f} MB/s"
                        last_calc_time = now
                        last_bytes = completed

                    with self._lock:
                        task.completed_bytes = completed
                        task.total_bytes = total
                        if total > 0:
                            task.percentage = (completed / total) * 100.0
                        task.status_msg = status

                with self._lock:
                    task.status = "completed"
                    task.percentage = 100.0
                    task.status_msg = "Установлена"
                    if self._installed_cache is not None:
                        self._installed_cache.add(current_model)

            except Exception as err:
                with self._lock:
                    task.status = "error"
                    task.error_msg = str(err)
                    task.status_msg = f"Ошибка: {err}"


GLOBAL_PULL_MANAGER = AsyncPullManager()


# ---------------------------------------------------------------------------
# Отрисовка каталога с окном прокрутки (Viewport Scrolling)
# ---------------------------------------------------------------------------

def _render_catalog(
    items: List[Dict[str, Any]],
    cursor: int,
    scroll_offset: int,
    page_size: int,
    hw: Dict[str, Any],
    notification: str = ""
) -> Text:
    vram_gb = hw["vram_gb"]
    gpu_name = hw["gpu_name"]
    ram_gb = hw["ram_gb"]
    is_apple_silicon = "Apple Silicon" in gpu_name
    effective_vram = max(0.0, vram_gb - 1.0) if not is_apple_silicon else vram_gb
    installed_set = GLOBAL_PULL_MANAGER.get_installed()
    total = len(items)

    t = Text()

    # Шапка информации
    if is_apple_silicon:
        t.append(f" Платформа: {gpu_name}   Память (Unified): {ram_gb} GB\n", style="dim")
    else:
        t.append(f" Видеокарта: {gpu_name}   VRAM: {vram_gb} GB (доступно ~{effective_vram:.0f} GB)   RAM: {ram_gb} GB\n", style="dim")
    t.append(" " + chr(0x2500) * 86 + "\n", style="dim")

    # Заголовок колонок + счетчик позиции
    t.append(f"  {'№':>2}  {'Модель':<38}  {'Вес':>8}   {'Потребление памяти':>18}   {'Статус':<15} [{cursor+1}/{total}]\n", style="dim")
    t.append(" " + chr(0x2500) * 86 + "\n", style="dim")

    # Индикатор скрытых строк сверху
    if scroll_offset > 0:
        t.append(f"      ▲  ... еще {scroll_offset} выше (листайте ↑) ...\n", style="dim cyan")

    # Видимый срез элементов
    visible_end = min(total, scroll_offset + page_size)
    for i in range(scroll_offset, visible_end):
        item = items[i]
        name = item["name"]
        size = item["size_gb"]
        vram_min = item["vram_min"]
        is_cursor = (i == cursor)
        is_custom = (item.get("category") == "custom")
        fit = classify_model_fit(vram_min, size, hw)

        # Колонки
        idx_str = f"{i+1:>2}."
        disp_name = name if len(name) <= 38 else (name[:35] + "...")
        size_str = f"{size:.1f} GB" if size > 0 else "   --   "
        vram_str = f"~{vram_min:.1f} GB" if vram_min > 0 else "    --    "

        # Статус
        task = GLOBAL_PULL_MANAGER.get_task(name)
        is_installed = name in installed_set or (task and task.status == "completed")
        is_downloading = (task and task.status == "downloading")
        is_queued = (task and task.status == "queued")

        if is_downloading:
            status_text = f"Загрузка {task.percentage:>2.0f}% ({task.speed})"
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

        row_content = f"  {idx_str} {disp_name:<38} {size_str:>8}   {vram_str:>18}   "

        if is_cursor:
            # Подсветка строки курсора
            t.append(row_content, style="bold black on bright_white")
            t.append(f"{status_text:<18}\n", style=f"{status_style} on bright_white")

            desc = item["note"]
            t.append(f"       ▸ {desc}\n", style="cyan")
            if fit == "orange" and not is_installed and not is_downloading and not is_queued:
                t.append(
                    f"       • Потребует ~{vram_min:.1f} GB (VRAM доступно ~{effective_vram:.0f} GB) — часть уйдет в оперативную память (RAM)\n",
                    style="rgb(255,170,0)"
                )
            elif fit == "red" and not is_installed and not is_downloading and not is_queued:
                t.append(
                    f"       ! Не рекомендуется: модели нужно ~{vram_min:.1f} GB, на ПК доступно ~{ram_gb:.0f} GB памяти\n",
                    style="bold red"
                )
        else:
            t.append(row_content)
            t.append(f"{status_text:<18}\n", style=status_style)

    # Индикатор скрытых строк снизу
    if visible_end < total:
        hidden_below = total - visible_end
        t.append(f"      ▼  ... еще {hidden_below} ниже (листайте ↓) ...\n", style="dim cyan")

    # Подвал
    t.append(" " + chr(0x2500) * 86 + "\n", style="dim")

    active_cnt, pending_cnt = GLOBAL_PULL_MANAGER.active_and_pending_count()
    if active_cnt > 0 or pending_cnt > 0:
        t.append(f" ▶ Загрузка: {active_cnt} качается  |  В очереди: {pending_cnt}   ", style="bold cyan")

    if notification:
        t.append(f" {notification}\n", style="bold yellow")
    else:
        t.append("\n")

    t.append(" Стрелки ↑↓ / PgUp PgDn — выбор       Enter — скачать / в очередь       q / Esc — выход\n", style="dim")
    return t


# ---------------------------------------------------------------------------
# Скачивание отдельной модели (для вызова через CLI / команды)
# ---------------------------------------------------------------------------

def pull_model(model_name: str) -> bool:
    clean_name = model_name.strip()
    if not clean_name:
        console.print("[red]Имя модели не может быть пустым.[/red]")
        return False

    console.print()
    console.print(f"[cyan]Загрузка:[/cyan] [bold]{clean_name}[/bold]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.fields[model_name]}[/bold cyan]"),
            BarColumn(bar_width=36),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            TextColumn("[dim]{task.description}[/dim]"),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                description="Подключение...", total=100, completed=0,
                model_name=clean_name,
            )
            for chunk in pull(clean_name, stream=True):
                status = getattr(chunk, "status", "") or (chunk.get("status", "") if isinstance(chunk, dict) else "")
                completed = getattr(chunk, "completed", 0) or (chunk.get("completed", 0) if isinstance(chunk, dict) else 0)
                total = getattr(chunk, "total", 0) or (chunk.get("total", 0) if isinstance(chunk, dict) else 0)
                if total > 0:
                    progress.update(task_id, total=total, completed=completed, description=status)
                else:
                    progress.update(task_id, description=status)

        GLOBAL_PULL_MANAGER.refresh_installed()
        console.print(f"[bold green]OK -- {clean_name} готова к работе[/bold green]\n")
        return True

    except KeyboardInterrupt:
        console.print(f"\n[yellow]Загрузка {clean_name} прервана.[/yellow]")
        return False
    except Exception as error:
        console.print(f"\n[red]Ошибка: {error}[/red]")
        return False


# ---------------------------------------------------------------------------
# Очередь загрузки (для CLI w1x pull model1 model2 ...)
# ---------------------------------------------------------------------------

class PullQueue:
    def __init__(self):
        self._queue: List[str] = []
        self._lock = threading.Lock()

    def add(self, model_names: List[str]):
        with self._lock:
            for name in model_names:
                clean = name.strip()
                if clean and clean not in self._queue:
                    self._queue.append(clean)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def pending_list(self) -> List[str]:
        with self._lock:
            return list(self._queue)

    def run_all(self):
        if self.is_empty():
            console.print("[yellow]Очередь пуста.[/yellow]")
            return
        with self._lock:
            total = len(self._queue)
            items = list(self._queue)
        console.print(f"\n[cyan]Очередь загрузки:[/cyan] {total} моделей")
        for n in items:
            console.print(f"  - {n}")
        console.print()
        done = 0
        successful = 0
        while True:
            current = None
            with self._lock:
                if self._queue:
                    current = self._queue.pop(0)
            if current is None:
                break
            done += 1
            console.print(f"[cyan]▶ [{done}/{total}][/cyan] {current}")
            if pull_model(current):
                successful += 1
            else:
                console.print(f"[yellow]Очередь остановлена после прерывания {current}.[/yellow]")
                break
        console.print(f"[bold green]Готово: {successful}/{total} скачано[/bold green]\n")


GLOBAL_PULL_QUEUE = PullQueue()


# ---------------------------------------------------------------------------
# Кроссплатформенное считывание клавиш (Windows / Linux / macOS)
# ---------------------------------------------------------------------------

def _poll_key_windows():
    import msvcrt
    if not msvcrt.kbhit():
        return None
    key = msvcrt.getch()
    if key in (b"\x00", b"\xe0"):
        ext = msvcrt.getch()
        if ext == b"H": return "UP"
        if ext == b"P": return "DOWN"
        if ext == b"I": return "PGUP"
        if ext == b"Q": return "PGDN"
        if ext == b"G": return "HOME"
        if ext == b"O": return "END"
        return None
    if key in (b"\r", b"\n"): return "ENTER"
    if key in (b"q", b"Q", b"\x1b"): return "EXIT"
    if key.isdigit(): return key.decode("ascii", errors="ignore")
    return None


def _poll_key_unix():
    import select, termios, tty
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], 0.04)
        if not rlist:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            r2, _, _ = select.select([sys.stdin], [], [], 0.04)
            if r2:
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A": return "UP"
                    if ch3 == "B": return "DOWN"
                    if ch3 == "5":
                        sys.stdin.read(1)
                        return "PGUP"
                    if ch3 == "6":
                        sys.stdin.read(1)
                        return "PGDN"
                    if ch3 == "H": return "HOME"
                    if ch3 == "F": return "END"
                elif ch2 == "O":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A": return "UP"
                    if ch3 == "B": return "DOWN"
                return None
            return "EXIT"
        if ch in ("\r", "\n"): return "ENTER"
        if ch in ("q", "Q"): return "EXIT"
        if ch.isdigit(): return ch
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def poll_key_crossplatform():
    if os.name == "nt":
        return _poll_key_windows()
    return _poll_key_unix()


# ---------------------------------------------------------------------------
# Интерактивный каталог с окном прокрутки (Viewport)
# ---------------------------------------------------------------------------

def interactive_model_catalog() -> Optional[str]:
    hw = detect_hardware()
    items = MODEL_CATALOG
    cursor = 0
    scroll_offset = 0
    total = len(items)
    notification = ""
    last_selected_model = None

    term_height = shutil.get_terminal_size((80, 25)).lines
    page_size = max(6, min(12, term_height - 11))

    console.print()
    with Live(_render_catalog(items, cursor, scroll_offset, page_size, hw, notification),
              console=console, auto_refresh=False, transient=True) as live:
        while True:
            # Следование камеры за курсором
            if cursor < scroll_offset:
                scroll_offset = cursor
            elif cursor >= scroll_offset + page_size:
                scroll_offset = cursor - page_size + 1

            live.update(_render_catalog(items, cursor, scroll_offset, page_size, hw, notification), refresh=True)
            
            action = poll_key_crossplatform()
            if action:
                notification = ""
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
                        try:
                            raw = session.prompt(
                                HTML("<b><ansigreen>Тег модели из Ollama › </ansigreen></b>")
                            ).strip()
                        except (KeyboardInterrupt, EOFError):
                            raw = ""
                        if raw:
                            res = GLOBAL_PULL_MANAGER.enqueue(raw)
                            last_selected_model = raw
                            notification = f"Добавлена в загрузку: {raw}"
                        live.start()
                    else:
                        model_name = chosen["name"]
                        fit = classify_model_fit(chosen["vram_min"], chosen["size_gb"], hw)

                        if fit == "red":
                            live.stop()
                            console.print()
                            console.print(
                                f"[bold red]⚠️ Внимание:[/bold red] Модель [bold]{model_name}[/bold] требует ~{chosen['vram_min']:.1f} GB памяти.\n"
                                f"На вашем ПК доступно ~{hw['vram_gb']:.0f} GB VRAM + {hw['ram_gb']:.0f} GB RAM.\n"
                                "[yellow]Модель может вызвать зависание системы или не запуститься.[/yellow]"
                            )
                            try:
                                ans = session.prompt(
                                    HTML("<b><ansired>Всё равно скачать? (y/n) › </ansired></b>")
                                ).strip().lower()
                            except (KeyboardInterrupt, EOFError):
                                ans = "n"
                            live.start()
                            if ans not in ("y", "yes", "д", "да"):
                                notification = f"Загрузка {model_name} отменена"
                                continue

                        last_selected_model = model_name
                        res = GLOBAL_PULL_MANAGER.enqueue(model_name)
                        if res == "already_installed":
                            notification = f"✓ {model_name} уже установлена"
                        elif res == "already_in_progress":
                            notification = f"⏳ {model_name} уже качается или в очереди"
                        elif res == "enqueued":
                            notification = f"▶ {model_name} добавлена в очередь загрузки"

                elif action == "EXIT":
                    break
                elif action.isdigit():
                    n = int(action)
                    if 1 <= n <= total:
                        cursor = n - 1

            time.sleep(0.04)

    return last_selected_model


# ---------------------------------------------------------------------------
# Мастер первого запуска
# ---------------------------------------------------------------------------

def first_time_setup_wizard() -> Optional[str]:
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

    # Ожидаем завершения загрузки если она запущена через менеджер
    task = GLOBAL_PULL_MANAGER.get_task(selected)
    if task and task.status in ("queued", "downloading"):
        console.print(f"\n[cyan]Ожидание завершения загрузки {selected}...[/cyan]")
        while task.status in ("queued", "downloading"):
            time.sleep(0.5)

    if selected:
        if task and task.status == "error":
            console.print(
                Panel(
                    f"[bold red]Ошибка загрузки {selected}: {task.error_msg or 'Сбой соединения'}[/bold red]\n"
                    "Модель не была установлена как модель по умолчанию.\n"
                    "[dim]Попробуйте скачать повторно через /pull[/dim]",
                    title="Ошибка загрузки",
                    border_style="red",
                    padding=(1, 2),
                )
            )
            return None
        from .config import load_user_config, save_user_config
        cfg = load_user_config()
        cfg["default_model"] = selected
        save_user_config(cfg)
        console.print(
            Panel(
                f"[bold green]{selected}[/bold green] установлена как модель по умолчанию.\n"
                f"Контекст: {cfg.get('context_tokens', 8192)} токенов   "
                f"VRAM keep-alive: {cfg.get('keep_alive_minutes', 30)} мин.\n\n"
                "[dim]Настройки: /config[/dim]",
                title="W1x готов",
                border_style="green",
                padding=(1, 2),
            )
        )
        return selected
    return None
