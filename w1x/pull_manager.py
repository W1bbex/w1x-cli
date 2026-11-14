"""Единый менеджер загрузки и очереди моделей Ollama."""
import threading
import time
from typing import Any

from ollama import pull
from rich.markup import escape as escape_markup
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from .terminal import console
from .types import PullTaskSnapshot


class DownloadTask:
    """Состояние загрузки одной модели Ollama."""

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


def _extract_pull_chunk(chunk: Any) -> tuple[str, int, int]:
    """Безопасно извлечь (status, completed, total) из чанка ollama.pull."""
    raw_status = getattr(chunk, "status", None)
    if raw_status is None and isinstance(chunk, dict):
        raw_status = chunk.get("status")
    status = str(raw_status or "")

    raw_completed = getattr(chunk, "completed", None)
    if raw_completed is None and isinstance(chunk, dict):
        raw_completed = chunk.get("completed")
    try:
        completed = max(0, int(raw_completed or 0))
    except (TypeError, ValueError):
        completed = 0

    raw_total = getattr(chunk, "total", None)
    if raw_total is None and isinstance(chunk, dict):
        raw_total = chunk.get("total")
    try:
        total = max(0, int(raw_total or 0))
    except (TypeError, ValueError):
        total = 0

    return status, completed, total


class AsyncPullManager:
    """Единый потокобезопасный менеджер фоновой и CLI-загрузки моделей."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tasks: dict[str, DownloadTask] = {}
        self.queue: list[str] = []
        self._worker_thread: threading.Thread | None = None
        self._installed_cache: set[str] | None = None

    def _fetch_installed_set(self) -> set[str] | None:
        from . import models

        models.LAST_LIST_ERROR = None
        names = models.installed_models()
        if models.LAST_LIST_ERROR is not None:
            return None
        return set(names)

    def refresh_installed(self) -> set[str]:
        """Обновить кэш установленных моделей из Ollama вне мьютекса."""
        fetched = self._fetch_installed_set()
        with self._lock:
            if fetched is not None:
                self._installed_cache = fetched
            return set(self._installed_cache) if self._installed_cache is not None else set()

    def get_installed(self) -> set[str]:
        """Вернуть копию набора установленных моделей."""
        with self._lock:
            if self._installed_cache is not None:
                return set(self._installed_cache)
        return self.refresh_installed()

    def enqueue(self, model_name: str) -> str:
        """Поставить модель в общую очередь загрузки, если она ещё не установлена."""
        clean = model_name.strip()
        if not clean:
            return "empty"

        installed = self.get_installed()
        with self._lock:
            if clean in installed or (self._installed_cache is not None and clean in self._installed_cache):
                return "already_installed"

            existing = self.tasks.get(clean)
            if existing:
                if existing.status in ("queued", "downloading"):
                    return "already_in_progress"
                if existing.status == "completed":
                    return "already_completed"

            task = DownloadTask(clean)
            self.tasks[clean] = task
            self.queue.append(clean)

            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._worker_thread = threading.Thread(
                    target=self._worker_loop,
                    daemon=True,
                    name="W1xPullWorker",
                )
                self._worker_thread.start()
            return "enqueued"

    def get_task(self, model_name: str) -> DownloadTask | None:
        with self._lock:
            return self.tasks.get(model_name)

    def get_task_snapshot(self, model_name: str) -> PullTaskSnapshot | None:
        """Получить согласованный снимок состояния задачи под блокировкой."""
        with self._lock:
            task = self.tasks.get(model_name)
            if task is None:
                return None
            return {
                "status": task.status,
                "percentage": task.percentage,
                "speed": task.speed,
                "status_msg": task.status_msg,
                "error_msg": task.error_msg,
                "completed_bytes": task.completed_bytes,
                "total_bytes": task.total_bytes,
            }

    def get_queue_position(self, model_name: str) -> int:
        with self._lock:
            try:
                return self.queue.index(model_name) + 1
            except ValueError:
                return 0

    def active_and_pending_count(self) -> tuple[int, int]:
        with self._lock:
            downloading = sum(1 for t in self.tasks.values() if t.status == "downloading")
            queued = len(self.queue)
            return downloading, queued

    def run_cli_pull(self, model_names: list[str]) -> bool:
        """Скачать список моделей последовательно в одной общей очереди с прогресс-баром."""
        ordered_names: list[str] = []
        for raw in model_names:
            clean = raw.strip()
            if clean and clean not in ordered_names:
                ordered_names.append(clean)

        if not ordered_names:
            console.print("[yellow]Очередь загрузки пуста.[/yellow]")
            return False

        self.refresh_installed()
        console.print(f"\n[cyan]Очередь загрузки:[/cyan] {len(ordered_names)} моделей")
        tracked: list[str] = []
        already_ok = 0

        for name in ordered_names:
            res = self.enqueue(name)
            if res in ("already_installed", "already_completed"):
                console.print(f"  [green]✓ {escape_markup(name)} уже установлена.[/green]")
                already_ok += 1
            elif res in ("enqueued", "already_in_progress"):
                console.print(f"  ▶ {escape_markup(name)} — в очереди")
                tracked.append(name)

        if not tracked:
            return True

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
                bar_ids = {
                    name: progress.add_task(
                        description="В очереди...",
                        total=100,
                        completed=0,
                        model_name=escape_markup(name),
                    )
                    for name in tracked
                }
                while True:
                    all_finished = True
                    for name in tracked:
                        snap = self.get_task_snapshot(name)
                        if snap is None:
                            continue
                        status = snap["status"]
                        bar_id = bar_ids[name]
                        if status in ("queued", "downloading"):
                            all_finished = False
                            if snap["total_bytes"] > 0:
                                progress.update(
                                    bar_id,
                                    total=snap["total_bytes"],
                                    completed=snap["completed_bytes"],
                                    description=escape_markup(snap["status_msg"]),
                                )
                            else:
                                progress.update(bar_id, description=escape_markup(snap["status_msg"]))
                        elif status == "completed":
                            total_val = snap["total_bytes"] or 100
                            progress.update(bar_id, total=total_val, completed=total_val, description="Готово")
                        elif status == "error":
                            progress.update(
                                bar_id,
                                description=f"Ошибка: {escape_markup(snap['error_msg'])}",
                            )
                    if all_finished:
                        break
                    time.sleep(0.15)
        except KeyboardInterrupt:
            console.print("\n[yellow]Ожидание загрузки прервано пользователем.[/yellow]")
            return False

        self.refresh_installed()
        failed = [
            name
            for name in tracked
            if (snap := self.get_task_snapshot(name)) is not None and snap["status"] == "error"
        ]
        succeeded = already_ok + (len(tracked) - len(failed))
        for name in failed:
            snap = self.get_task_snapshot(name)
            err = snap["error_msg"] if snap else "неизвестная ошибка"
            console.print(f"[red]✗ Ошибка загрузки {escape_markup(name)}: {escape_markup(err)}[/red]")
        console.print(f"[bold green]Готово: {succeeded}/{len(ordered_names)} моделей доступно[/bold green]\n")
        return not failed

    def _worker_loop(self) -> None:
        while True:
            with self._lock:
                if not self.queue:
                    self._worker_thread = None
                    break
                current_model = self.queue.pop(0)
                task = self.tasks.get(current_model)
                if not task:
                    continue
                task.status = "downloading"
                task.status_msg = "Подключение..."

            last_calc_time = time.time()
            last_bytes = 0

            try:
                for chunk in pull(current_model, stream=True):
                    status, completed, total = _extract_pull_chunk(chunk)

                    now = time.time()
                    if completed < last_bytes:
                        last_bytes = completed
                        last_calc_time = now
                    dt = now - last_calc_time
                    if dt >= 0.4:
                        speed_bps = max(0.0, (completed - last_bytes) / dt) if dt > 0 else 0.0
                        speed_mb = speed_bps / (1024 * 1024)
                        with self._lock:
                            task.speed = f"{speed_mb:.1f} MB/s"
                        last_calc_time = now
                        last_bytes = completed

                    with self._lock:
                        if total > 0:
                            task.completed_bytes = completed
                            task.total_bytes = total
                            task.percentage = min(100.0, (completed / total) * 100.0)
                        if status:
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
