"""
Regression and resilience test suite for w1x-cli.
Covers:
- Preload task isolation and error reset under concurrency
- Worker shutdown on an empty queue
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


def test_worker_exits_and_clears_reference_on_empty_queue():
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


def test_datetime_normalizes_string_and_none_offsets():
    """Fractional strings follow the integer offset contract; None becomes zero."""
    res_str = calculate_datetime(
        days="1.5", hours=None, minutes="0", seconds=10,
        base_datetime="2026-01-01T12:00:00+03:00",
    )
    data = json.loads(res_str)
    assert "error" not in data
    assert data["current"] == "2026-01-01T12:00:00+03:00"
    assert data["result"] == "2026-01-02T12:00:10+03:00"
    assert data["offset"] == {"days": 1, "hours": 0, "minutes": 0, "seconds": 10}
    assert data["answer"]


def test_datetime_rejects_invalid_offset():
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


def test_model_prompt_precomputes_math_without_dates_or_phone_numbers(monkeypatch):
    """Exercise arithmetic precomputation inside the real model prompt path."""
    import w1x.engine as engine

    text = "События 2026-11-08 и звонок на номер +7-999-123-45-67, но посчитай 25 * 4"
    # Bypass direct tool routing to exercise precomputation in the model prompt.
    monkeypatch.setattr(engine, "parse_math_request", lambda _: None)
    monkeypatch.setattr(engine, "split_datetime_request", lambda _: (None, text))
    monkeypatch.setattr(engine, "relevant_memory_block", lambda _: "")
    monkeypatch.setattr(engine, "get_keep_alive", lambda: "0")
    monkeypatch.setattr(engine, "get_context_tokens", lambda: 8192)
    chat = MagicMock(return_value=iter([]))
    calculator = MagicMock(wraps=calculate_expression)
    monkeypatch.setattr(engine, "chat", chat)
    monkeypatch.setattr(engine, "calculate_expression", calculator)

    engine.generate_answer("offline-model", [
        {"role": "system", "content": "Test assistant"},
        {"role": "user", "content": text},
    ], False)

    calculator.assert_called_once_with("25 * 4")
    chat.assert_called_once()
    prompt = chat.call_args.kwargs["messages"][0]["content"]
    math_block = prompt.split("Точные математические вычисления системы:\n", 1)[1]
    assert "- 25 * 4 = 100" in math_block
    assert "2026-11-08" not in math_block
    assert "999-123" not in math_block


def test_config_save_replaces_complete_temporary_file(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.json"
    monkeypatch.setattr(w1x_config, "CONFIG_FILE_PATH", cfg_file)
    cfg_file.write_text('{"old": true}', encoding="utf-8")
    replace = Path.replace
    replacements = []

    def observe_replace(source, destination):
        assert source == cfg_file.with_suffix(".tmp")
        assert destination == cfg_file
        assert json.loads(cfg_file.read_text(encoding="utf-8")) == {"old": True}
        assert json.loads(source.read_text(encoding="utf-8")) == {"test_key": "test_val"}
        replacements.append(source)
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", observe_replace)

    save_user_config({"test_key": "test_val"})
    assert cfg_file.exists()
    saved = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert saved.get("test_key") == "test_val"
    assert replacements == [cfg_file.with_suffix(".tmp")]
    assert not replacements[0].exists()


def test_config_save_preserves_previous_file_when_replacement_fails(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.json"
    original = '{"old": true}\n'
    cfg_file.write_text(original, encoding="utf-8")
    monkeypatch.setattr(w1x_config, "CONFIG_FILE_PATH", cfg_file)
    replace = MagicMock(side_effect=OSError("replacement denied"))
    monkeypatch.setattr(Path, "replace", replace)

    save_user_config({"new": True})

    replace.assert_called_once_with(cfg_file)
    assert cfg_file.read_text(encoding="utf-8") == original


def test_reset_all_models_delegates_without_forwarding_display_option(monkeypatch):
    unload = MagicMock(return_value=["test-model"])
    monkeypatch.setattr(models, "unload_all_models", unload)
    res = models.reset_all_models(show_message=True)
    assert res == ["test-model"]
    unload.assert_called_once_with()


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


@pytest.mark.parametrize("exit_value, expected_code", [(None, 0), (7, 7), ("failure", 1)])
def test_python_runner_forwards_stream_attributes_and_exit_codes(tmp_path, monkeypatch, exit_value, expected_code):
    """Execute the generated runner through pipes, without opening a terminal window."""
    import subprocess
    from w1x import testpy

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    monkeypatch.setattr(testpy.tempfile, "mkdtemp", lambda **_: str(session_dir))
    run = subprocess.run
    child_results = []

    def run_without_window(arguments, **_):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = run(arguments, input="q\n", capture_output=True, text=True,
                     encoding="utf-8", env=env, timeout=10)
        child_results.append(result)
        return result

    monkeypatch.setattr(testpy.subprocess, "run", run_without_window)
    code = (
        "import sys\n"
        "print('stream-encoding=' + sys.stdout.encoding)\n"
        "print('stream-fd=' + str(sys.stdout.fileno()))\n"
        f"raise SystemExit({exit_value!r})\n"
    )
    exit_code, log = testpy.run_interactive_python_test(code)

    assert len(child_results) == 1
    assert child_results[0].returncode == 0
    assert exit_code == expected_code
    assert "stream-encoding=utf-8" in log
    assert "stream-fd=" in log
    assert "Traceback" not in log
    assert not session_dir.exists()


def test_app_imports_load_user_config():
    from w1x import app
    assert callable(getattr(app, "load_user_config", None))


def test_weak_tool_answer_escalate():
    from w1x.engine import weak_tool_answer
    assert weak_tool_answer("ESCALATE") is True
    assert weak_tool_answer("Пожалуйста, escalate") is True


def test_calculate_datetime_overflow_safety():
    from w1x.tools import calculate_datetime
    res = json.loads(calculate_datetime(days="999999999999"))
    assert "error" in res


def test_testpy_null_byte_safety():
    from w1x.testpy import is_valid_python
    assert is_valid_python("print('hello')\x00extra") is False


def test_terminal_runner_captures_subprocess_crash_returncode(tmp_path, monkeypatch):
    import subprocess
    from w1x import testpy

    session_dir = tmp_path / "crash_session"
    session_dir.mkdir()
    monkeypatch.setattr(testpy.tempfile, "mkdtemp", lambda **_: str(session_dir))

    fake_proc = subprocess.CompletedProcess(args=["python", "runner.py"], returncode=42)
    monkeypatch.setattr(testpy.subprocess, "run", lambda *args, **kwargs: fake_proc)

    exit_code, log = testpy.run_interactive_python_test("print('crash')")
    assert exit_code == 42


def test_save_user_config_returns_boolean_and_handles_error(tmp_path, monkeypatch):
    from w1x import config as w1x_config
    cfg_file = tmp_path / "config.json"
    monkeypatch.setattr(w1x_config, "CONFIG_FILE_PATH", cfg_file)

    # Success case returns True
    assert w1x_config.save_user_config({"test_status": "ok"}) is True
    assert cfg_file.exists()

    # Failure case returns False honestly
    monkeypatch.setattr(Path, "replace", MagicMock(side_effect=OSError("write denied")))
    assert w1x_config.save_user_config({"test_status": "fail"}) is False


def test_floor_division_and_modulo_preserve_python_operator_semantics():
    from w1x.tools import calculate_expression

    observed = (calculate_expression("-5 // 2"), calculate_expression("-5 % 2"))
    assert observed == ("-3", "1")


def test_decimal_literal_is_not_rounded_through_binary_float():
    from w1x.tools import calculate_expression

    assert calculate_expression("9007199254740993.0 - 9007199254740992.0") == "1"


def test_numeric_base_date_is_used_instead_of_today():
    import json
    from datetime import datetime
    from w1x.tools import calculate_datetime, parse_datetime_request

    arguments = parse_datetime_request("если сейчас 01.01.2025 10:00, через 2 часа")
    assert arguments is not None
    result = datetime.fromisoformat(json.loads(calculate_datetime(**arguments))["result"])
    assert result.replace(tzinfo=None) == datetime(2025, 1, 1, 12)


def test_time_components_keep_their_individual_directions():
    import json
    from datetime import datetime
    from w1x.tools import calculate_datetime, parse_datetime_request

    arguments = parse_datetime_request(
        "если сейчас 1 января 2025 12:00 прибавь 1 час и вычти 30 минут"
    )
    assert arguments is not None
    result = datetime.fromisoformat(json.loads(calculate_datetime(**arguments))["result"])
    assert result.replace(tzinfo=None) == datetime(2025, 1, 1, 12, 30)


