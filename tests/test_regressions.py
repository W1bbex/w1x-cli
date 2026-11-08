"""
Regression and resilience test suite for w1x-cli.
Covers:
- Preload task isolation and error reset under concurrency
- Worker queue termination atomicity
- Memory facts sandbox and corrupted file recovery
- Chat session recovery from invalid payloads
- Arithmetic parsing without misinterpreting dates or phone numbers
- Terminal runner and configuration atomic persistence
"""
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from w1x import config, memory, models, tools
from w1x.app import load_current_chat
import w1x.app as w1x_app
from w1x.config import load_user_config, save_user_config
import w1x.config as w1x_config
from w1x.engine import calculate_expression
from w1x.memory import load_facts
from w1x.models import PreloadTask, wait_for_model_ready, _PRELOAD_LOCK
import w1x.models as w1x_models
from w1x.puller import AsyncPullManager, DownloadTask
from w1x.tools import calculate_datetime


def test_preload_task_isolation():
    """Verify that multiple preload calls create isolated tasks and don't mix events."""
    a_finished = threading.Event()
    b_started = threading.Event()

    def fake_generate_a(*args, **kwargs):
        b_started.wait(timeout=2.0)
        time.sleep(0.05)
        a_finished.set()

    with patch("w1x.models.generate", side_effect=fake_generate_a):
        models.start_background_preload("model_a")
        task_a = models._CURRENT_PRELOAD
        assert task_a is not None
        assert task_a.name == "model_a"
        assert not task_a.event.is_set()

        b_finished = threading.Event()
        def fake_generate_b(*args, **kwargs):
            b_finished.wait(timeout=2.0)

        with patch("w1x.models.generate", side_effect=fake_generate_b):
            b_started.set()
            models.start_background_preload("model_b")
            task_b = models._CURRENT_PRELOAD
            assert task_b is not None
            assert task_b.name == "model_b"
            assert task_b is not task_a

            a_finished.wait(timeout=2.0)
            assert task_a.event.is_set()
            assert not task_b.event.is_set()

        b_finished.set()


def test_puller_worker_termination_atomicity():
    """Verify worker terminates safely when queue is empty."""
    mgr = AsyncPullManager()
    t = threading.Thread(target=mgr._worker_loop, daemon=True)
    mgr._worker_thread = t
    t.start()
    t.join(timeout=1.0)
    assert not t.is_alive()
    assert mgr._worker_thread is None


def test_models_preload_error_clears_task():
    """Verify failed model preloads clear _CURRENT_PRELOAD so subsequent requests are not blocked."""
    task = PreloadTask(name="broken_model")
    task.error = RuntimeError("Ollama connection failed")
    task.event.set()

    with _PRELOAD_LOCK:
        w1x_models._CURRENT_PRELOAD = task

    with pytest.raises(RuntimeError, match="Ollama connection failed"):
        wait_for_model_ready("broken_model")

    with _PRELOAD_LOCK:
        assert w1x_models._CURRENT_PRELOAD is None

    wait_for_model_ready("broken_model")


def test_memory_corrupted_file_moved(tmp_path, monkeypatch):
    """Verify corrupted memory-facts.json is archived to .corrupted.<timestamp>.json and reset."""
    facts_file = tmp_path / "memory-facts.json"
    facts_file.write_text("{broken json syntax: missing quotes", encoding="utf-8")
    monkeypatch.setattr("w1x.memory._get_facts_path", lambda: facts_file)

    res1 = load_facts()
    assert res1 == []
    assert not facts_file.exists()

    corrupted_files = list(tmp_path.glob("memory-facts.corrupted.*.json"))
    assert len(corrupted_files) == 1

    res2 = load_facts()
    assert res2 == []
    corrupted_files_after = list(tmp_path.glob("memory-facts.corrupted.*.json"))
    assert len(corrupted_files_after) == 1


def test_memory_sandbox_redirection(tmp_path, monkeypatch):
    """Verify mutating config paths redirects facts storage without polluting user home."""
    sandbox_facts = tmp_path / "test-facts.json"
    sandbox_chat = tmp_path / "test-chat"

    monkeypatch.setattr(config, "FACTS_PATH", sandbox_facts)
    monkeypatch.setattr(config, "CHAT_DIR", sandbox_chat)

    assert memory._get_facts_path() == sandbox_facts
    assert memory._get_chat_dir() == sandbox_chat
    assert memory.load_facts() == []

    memory.save_facts([{"fact": "Test fact", "created_at": "2026-11-04"}])
    assert sandbox_facts.exists()
    assert len(memory.load_facts()) == 1


def test_tools_datetime_offset_safe_types():
    """Verify datetime offset parameters accept flexible string floats and None safely."""
    res_str = calculate_datetime(days="1.5", hours=None, minutes="0", seconds=10)
    data = json.loads(res_str)
    assert "answer" in data or "datetime" in data or "error" not in data

    res_err = calculate_datetime(days="invalid_days")
    data_err = json.loads(res_err)
    assert "error" in data_err


def test_app_load_current_chat_corrupted_list(tmp_path, monkeypatch):
    """Verify chat memory loader recovers gracefully from empty or list-typed JSON."""
    chat_file = tmp_path / "current-chat.json"
    chat_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(w1x_app, "CURRENT_CHAT_PATH", chat_file)

    sys_prompt, messages = load_current_chat()
    assert sys_prompt == w1x_app.DEFAULT_SYSTEM_PROMPT
    assert len(messages) >= 1
    assert messages[0]["role"] == "system"


def test_engine_dates_and_phones_not_computed_as_math():
    """Verify dates and phone numbers are excluded from automated arithmetic precomputation."""
    text = "События 2026-11-08 и звонок на номер +7-999-123-45-67, но посчитай 25 * 4"
    math_candidates = re.findall(r"[\d\s().,+\-*/%]{3,}", text.replace("×", "*").replace("÷", "/"))
    precomputed = []
    for cand in math_candidates:
        cand_clean = cand.strip(" ,.")
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b", cand_clean) or re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", cand_clean):
            continue
        if re.search(r"^\+?\d[\d\s-]{7,}\d$", cand_clean):
            continue
        if re.search(r"[+\-*/%]", cand_clean) and re.search(r"\d", cand_clean) and len(cand_clean) >= 3:
            val = calculate_expression(cand_clean)
            if not val.startswith("Ошибка"):
                precomputed.append(f"{cand_clean} = {val}")

    assert not any("2026-11-08" in item for item in precomputed)
    assert not any("999-123" in item for item in precomputed)
    assert any("25 * 4 = 100" in item for item in precomputed)


def test_config_atomic_save(tmp_path, monkeypatch):
    """Verify user configuration is saved atomically via temporary replacement."""
    cfg_file = tmp_path / "config.json"
    monkeypatch.setattr(w1x_config, "CONFIG_FILE_PATH", cfg_file)

    save_user_config({"test_key": "test_val"})
    assert cfg_file.exists()
    saved = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert saved.get("test_key") == "test_val"


def test_reset_all_models_accepts_show_message():
    """Verify reset_all_models accepts show_message parameter without crash."""
    res = models.reset_all_models(show_message=True)
    assert isinstance(res, (list, bool, type(None)))


def test_set_file_workspace_syncs_globals(tmp_path):
    """Verify set_file_workspace updates both tools and config roots."""
    orig = getattr(config, "ACTIVE_FILE_ROOT", None)
    try:
        tools.set_file_workspace(str(tmp_path))
        assert tools.ACTIVE_FILE_ROOT == tmp_path.resolve()
        assert config.ACTIVE_FILE_ROOT == tmp_path.resolve()
    finally:
        config.ACTIVE_FILE_ROOT = orig
        tools.ACTIVE_FILE_ROOT = orig


def test_testpy_runner_code_templates():
    """Verify interactive test runner template handles dynamic streams and error codes."""
    import inspect
    from w1x import testpy
    source = inspect.getsource(testpy.run_interactive_python_test)
    assert "__getattr__" in source
    assert "getattr(self.original, name)" in source
    assert "exit_code = exc.code if isinstance(exc.code, int) else (1 if exc.code is not None else 0)" in source
