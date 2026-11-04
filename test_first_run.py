# -*- coding: utf-8 -*-
"""Скрипт симуляции первого запуска W1x CLI (когда нет моделей и нет конфига)."""
import os
import sys
import tempfile
from pathlib import Path

# Добавляем путь к w1x-cli
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import w1x.config as config
import w1x.models as models

# Создаём временную чистую папку настроек для теста
temp_dir = tempfile.TemporaryDirectory(prefix="w1x-firstrun-")
sandbox_path = Path(temp_dir.name)

config.DATA_DIR = sandbox_path
config.CHAT_DIR = sandbox_path / "chats"
config.CURRENT_CHAT_PATH = sandbox_path / "current-chat.json"
config.FACTS_PATH = sandbox_path / "memory-facts.json"
config.CONFIG_FILE_PATH = sandbox_path / "config.json"

# Симулируем чистую систему с 0 скачанными моделями
_real_installed = models.installed_models


def mock_installed_models():
    # Если в песочнице уже был скачан/выбран конфиг — возвращаем реальный список
    if config.CONFIG_FILE_PATH.exists():
        return _real_installed()
    return []


models.installed_models = mock_installed_models

from w1x.app import main

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" 🧪 РЕЖИМ СИМУЛЯЦИИ: ПЕРВЫЙ ЗАПУСК W1X (0 МОДЕЛЕЙ, ЧИСТАЯ СИСТЕМА)")
    print("=" * 80 + "\n")
    try:
        main()
    finally:
        try:
            temp_dir.cleanup()
        except Exception:
            pass
