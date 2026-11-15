"""Централизованная настройка логирования W1x CLI."""
import logging
import os
from logging.handlers import RotatingFileHandler

from . import config

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(*, debug: bool = False) -> logging.Logger:
    """Инициализировать логирование с ротацией файлов в директории данных."""
    logger = logging.getLogger("w1x")
    logger.setLevel(logging.DEBUG)

    # Очищаем существующие обработчики
    logger.handlers.clear()

    # Директория логов внутри DATA_DIR
    logs_dir = config.DATA_DIR / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "w1x.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG if (debug or os.environ.get("W1X_DEBUG") == "1") else logging.INFO)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(file_handler)
    except Exception:
        # Если файловый доступ недоступен, fallback на NullHandler
        logger.addHandler(logging.NullHandler())

    # Если включен режим отладки, добавляем вывод в консоль
    if debug or os.environ.get("W1X_DEBUG") == "1":
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter("[dim]%(levelname)s: %(message)s[/dim]"))
        logger.addHandler(console_handler)

    return logger
