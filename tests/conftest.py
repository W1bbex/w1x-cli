"""Изоляция тестового окружения от пользовательской директории ~/.w1x."""
from types import SimpleNamespace

import pytest

from w1x import config, models


@pytest.fixture(autouse=True)
def isolate_w1x_data_and_config(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    """Гарантирует, что тесты никогда не читают и не модифицируют файлы в ~/.w1x."""
    test_data_dir = tmp_path / "w1x_test_data"
    test_chat_dir = test_data_dir / "chats"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    test_chat_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "DATA_DIR", test_data_dir)
    monkeypatch.setattr(config, "CHAT_DIR", test_chat_dir)
    monkeypatch.setattr(config, "CURRENT_CHAT_PATH", test_data_dir / "current-chat.json")
    monkeypatch.setattr(config, "FACTS_PATH", test_data_dir / "memory-facts.json")
    monkeypatch.setattr(config, "CONFIG_FILE_PATH", test_data_dir / "config.json")
    monkeypatch.setattr(models, "ps", lambda: SimpleNamespace(models=[]))
    monkeypatch.setattr(models, "generate", lambda **kwargs: None)

    config.ACTIVE_SESSION_DATA.clear()
    config.ACTIVE_FILE_ROOT = None
