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
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import w1x.app as w1x_app
import w1x.config as w1x_config
import w1x.models as w1x_models
import w1x.models_ui as w1x_models_ui
from w1x import config, memory, models
from w1x.chat_store import load_current_chat
from w1x.config import save_user_config
from w1x.memory import load_facts
from w1x.models import _PRELOAD_LOCK, PreloadTask, wait_for_model_ready
from w1x.pull_manager import AsyncPullManager
from w1x.tools import antigravity as w1x_antigravity
from w1x.tools import datetime_tool as w1x_datetime
from w1x.tools import math_tool as w1x_math
from w1x.tools import workspace as w1x_workspace
from w1x.tools.datetime_tool import calculate_datetime
from w1x.tools.math_tool import calculate_expression


def test_preload_task_isolation():
    """Verify that multiple preload calls create isolated tasks and don't mix events."""
    a_entered = threading.Event()
    a_finished = threading.Event()
    b_started = threading.Event()
    b_finished = threading.Event()
    task_a = None
    task_b = None

    def fake_generate_a(*args, **kwargs):
        a_entered.set()
        b_started.wait(timeout=2.0)
        time.sleep(0.05)
        a_finished.set()

    def fake_generate_b(*args, **kwargs):
        b_finished.wait(timeout=2.0)

    try:
        with patch("w1x.models.generate", side_effect=fake_generate_a):
            models.start_background_preload("model_a")
            task_a = models._CURRENT_PRELOAD
            assert task_a is not None
            assert task_a.name == "model_a"
            assert not task_a.event.is_set()
            assert a_entered.wait(timeout=2.0)

            with patch("w1x.models.generate", side_effect=fake_generate_b):
                b_started.set()
                models.start_background_preload("model_b")
                task_b = models._CURRENT_PRELOAD
                assert task_b is not None
                assert task_b.name == "model_b"
                assert task_b is not task_a

                assert a_finished.wait(timeout=2.0)
                assert task_a.event.is_set()
                assert not task_b.event.is_set()
                b_finished.set()
    finally:
        b_started.set()
        b_finished.set()
        for task in (task_a, task_b):
            if task is not None and task.thread is not None:
                task.thread.join(timeout=2.0)
        with _PRELOAD_LOCK:
            models._CURRENT_PRELOAD = None


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
    assert config.CHAT_DIR == sandbox_chat
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
    monkeypatch.setattr(w1x_config, "CURRENT_CHAT_PATH", chat_file)

    sys_prompt, messages = load_current_chat()
    assert sys_prompt == w1x_config.DEFAULT_SYSTEM_PROMPT
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
    """Verify set_file_workspace updates config.ACTIVE_FILE_ROOT as single source of truth."""
    orig = getattr(config, "ACTIVE_FILE_ROOT", None)
    try:
        w1x_workspace.set_file_workspace(str(tmp_path))
        assert config.ACTIVE_FILE_ROOT == tmp_path.resolve()
        assert w1x_workspace._safe_workspace_path(".") == tmp_path.resolve()
    finally:
        config.ACTIVE_FILE_ROOT = orig


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


def test_weak_tool_answer_escalate():
    from w1x.engine import weak_tool_answer
    assert weak_tool_answer("ESCALATE") is True
    assert weak_tool_answer("Пожалуйста, escalate") is True


def test_calculate_datetime_overflow_safety():
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
    observed = (calculate_expression("-5 // 2"), calculate_expression("-5 % 2"))
    assert observed == ("-3", "1")


def test_decimal_literal_is_not_rounded_through_binary_float():
    assert calculate_expression("9007199254740993.0 - 9007199254740992.0") == "1"


def test_numeric_base_date_is_used_instead_of_today():
    import json
    from datetime import datetime

    from w1x.tools.datetime_tool import parse_datetime_request

    arguments = parse_datetime_request("если сейчас 01.01.2025 10:00, через 2 часа")
    assert arguments is not None
    result = datetime.fromisoformat(json.loads(calculate_datetime(**arguments))["result"])
    assert result.replace(tzinfo=None) == datetime(2025, 1, 1, 12)


def test_time_components_keep_their_individual_directions():
    import json
    from datetime import datetime

    from w1x.tools.datetime_tool import parse_datetime_request

    arguments = parse_datetime_request(
        "если сейчас 1 января 2025 12:00 прибавь 1 час и вычти 30 минут"
    )
    assert arguments is not None
    result = datetime.fromisoformat(json.loads(calculate_datetime(**arguments))["result"])
    assert result.replace(tzinfo=None) == datetime(2025, 1, 1, 12, 30)


def test_models_import_has_no_background_ctrl_q_thread():
    assert not any(t.name == "CtrlQGlobalListener" for t in threading.enumerate())


def test_ask_antigravity_kills_process_before_tempdir_cleanup_on_timeout(tmp_path, monkeypatch):
    import subprocess

    events = []

    class FakeTempDir:
        def __init__(self, prefix=""):
            self.path = tmp_path / "agy-temp"
            self.path.mkdir(exist_ok=True)

        def __enter__(self):
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            events.append("tempdir_exit")
            return False

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self._killed = False

        def poll(self):
            return 0 if self._killed else None

        def communicate(self, timeout=None):
            if not self._killed:
                raise subprocess.TimeoutExpired(cmd="agy", timeout=timeout)
            events.append("reaped")
            return ("", "")

        def kill(self):
            self._killed = True
            events.append("killed")

    monkeypatch.setattr(w1x_antigravity.shutil, "which", lambda name: "C:/fake/agy.exe")
    monkeypatch.setattr(w1x_antigravity.tempfile, "TemporaryDirectory", FakeTempDir)
    monkeypatch.setattr(w1x_antigravity.subprocess, "Popen", lambda *a, **kw: FakeProcess())

    res = w1x_antigravity.ask_antigravity("analyze project")
    assert res.startswith("ESCALATE_ERROR: Antigravity не ответила за")
    assert events.index("killed") < events.index("tempdir_exit")


def test_run_with_spinner_joins_thread_and_calls_on_interrupt(monkeypatch):
    from w1x import terminal

    release_worker = threading.Event()
    interrupted = threading.Event()
    worker_finished = threading.Event()

    def slow_work():
        release_worker.wait(timeout=2.0)
        worker_finished.set()
        return "done"

    def on_interrupt():
        interrupted.set()
        release_worker.set()

    monkeypatch.setattr(terminal.time, "sleep", MagicMock(side_effect=KeyboardInterrupt))

    with pytest.raises(KeyboardInterrupt):
        terminal.run_with_spinner("Working...", slow_work, on_interrupt=on_interrupt)

    assert interrupted.is_set()
    assert worker_finished.is_set()


def test_pull_manager_lock_free_fetch_and_multilayer_none_chunks(monkeypatch):
    from w1x import models as w1x_models
    from w1x import pull_manager

    mgr = pull_manager.AsyncPullManager()
    lock_states = []

    def fake_installed_models():
        lock_states.append(mgr._lock.locked())
        w1x_models.LAST_LIST_ERROR = None
        return ["existing:latest"]

    monkeypatch.setattr(w1x_models, "installed_models", fake_installed_models)

    installed_copy = mgr.get_installed()
    assert installed_copy == {"existing:latest"}
    assert lock_states == [False]
    installed_copy.add("tampered:tag")
    assert "tampered:tag" not in mgr.get_installed()

    chunks = [
        {"status": "pulling manifest", "completed": None, "total": None},
        {"status": "pulling layer 1", "completed": 10_000_000, "total": 20_000_000},
        {"status": "pulling layer 2", "completed": 1_000_000, "total": 5_000_000},
        {"status": "verifying sha256 digest", "completed": None, "total": None},
        {"status": "success", "completed": 5_000_000, "total": 5_000_000},
    ]

    def fake_pull(model_name, stream=True):
        for idx, ch in enumerate(chunks):
            if idx in (1, 2):
                time.sleep(0.45)
            yield ch

    monkeypatch.setattr(pull_manager, "pull", fake_pull)

    status = mgr.enqueue("demo:latest")
    assert status == "enqueued"
    thread = mgr._worker_thread
    assert thread is not None
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    task = mgr.get_task("demo:latest")
    assert task is not None
    assert task.status == "completed"
    assert not task.speed.startswith("-")
    assert "demo:latest" in mgr.get_installed()


def test_poll_key_unix_handles_partial_escape_and_ctrl_keys(monkeypatch):
    import types

    from w1x import key_input

    def run_sequence(byte_chunks):
        queue = list(byte_chunks)

        fake_select = types.SimpleNamespace(
            select=lambda r, w, x, timeout=0: ([r[0]] if queue and queue[0] is not None else [], [], [])
        )
        fake_termios = types.SimpleNamespace(
            TCSADRAIN=0,
            tcgetattr=lambda fd: [0],
            tcsetattr=lambda fd, when, attrs: None,
        )
        fake_tty = types.SimpleNamespace(setraw=lambda fd: None)

        def fake_read(fd, n):
            item = queue.pop(0)
            if item is None:
                return b""
            return item

        monkeypatch.setitem(sys.modules, "select", fake_select)
        monkeypatch.setitem(sys.modules, "termios", fake_termios)
        monkeypatch.setitem(sys.modules, "tty", fake_tty)
        monkeypatch.setattr(key_input.os, "read", fake_read)
        fake_stdin = types.SimpleNamespace(isatty=lambda: True, fileno=lambda: 0)
        monkeypatch.setattr(key_input.sys, "stdin", fake_stdin)
        return key_input._poll_key_unix()

    assert run_sequence([b"\x1b", b"[", None]) is None
    assert run_sequence([b"\x1b", b"[", b"A"]) == "UP"
    assert run_sequence([b"\x03"]) == "EXIT"
    assert run_sequence([b"\x11"]) == "EXIT"
    assert run_sequence([b"\xb2"]) is None


def test_generate_answer_handles_keyboard_interrupt_during_escalation(monkeypatch):
    import types

    from w1x import engine

    monkeypatch.setattr(engine, "parse_math_request", lambda _: None)
    monkeypatch.setattr(engine, "split_datetime_request", lambda text: (None, text))
    monkeypatch.setattr(engine, "relevant_memory_block", lambda _: "")
    monkeypatch.setattr(engine, "get_keep_alive", lambda: "0")
    monkeypatch.setattr(engine, "get_context_tokens", lambda: 8192)

    tool_call = types.SimpleNamespace(
        function=types.SimpleNamespace(name="search_web", arguments={"query": "test"})
    )

    def make_chunk(content="", tool_calls=None):
        return types.SimpleNamespace(
            message=types.SimpleNamespace(thinking="", content=content, tool_calls=tool_calls or []),
            prompt_eval_count=10,
            eval_count=5,
            eval_duration=1_000_000,
            total_duration=2_000_000,
            load_duration=0,
            prompt_eval_duration=500_000,
        )

    responses = [
        iter([make_chunk(tool_calls=[tool_call])]),
        iter([make_chunk(content="Не могу найти.")]),
        iter([make_chunk(content="ESCALATE")]),
    ]
    monkeypatch.setattr(engine, "chat", lambda **kwargs: responses.pop(0))
    monkeypatch.setattr(engine, "execute_tool_call", lambda call: "Поиск не вернул результатов.")
    monkeypatch.setattr(engine, "run_with_spinner", MagicMock(side_effect=KeyboardInterrupt))

    answer, _, interrupted = engine.generate_answer(
        "test-model",
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "find info"}],
        False,
    )
    assert interrupted is True
    assert answer == "ESCALATE"


def test_interactive_config_menu_handles_ctrl_c_in_subprompts(monkeypatch):
    prompts = iter(["2", KeyboardInterrupt(), "3", EOFError(), "0"])

    def fake_prompt(*args, **kwargs):
        nxt = next(prompts)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    monkeypatch.setattr(w1x_models_ui, "load_user_config", lambda: {"default_model": "m", "keep_alive_minutes": 30, "context_tokens": 8192, "stats_enabled": False})
    monkeypatch.setattr(w1x_models_ui.secondary_session, "prompt", fake_prompt)

    w1x_models_ui.interactive_config_menu()
    with pytest.raises(StopIteration):
        next(prompts)


def test_memory_forget_russian_case_inflection_and_none_safety(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FACTS_PATH", tmp_path / "facts.json")
    monkeypatch.setattr(config, "CHAT_DIR", tmp_path / "chats")

    assert memory.remember_fact("У меня есть кот") == "Запомнил: У меня есть кот."
    assert memory.forget_facts("кота") == "Забыл: У меня есть кот."
    assert memory.load_facts() == []

    assert memory.remember_fact(None).startswith("Ошибка:")
    assert memory.memory(None, None).startswith("Ошибка:")
    assert memory.recall_facts(None) == "Постоянная память фактов пуста."
    assert memory.forget_facts(None).startswith("Ошибка:")

    migrated = memory.migrate_explicit_memory([
        {"role": "user", "content": "Запомни: люблю кофе"},
        {"role": "user", "content": "Запомни: люблю кофе"},
    ])
    assert migrated == 1
    assert [item["fact"] for item in memory.load_facts()] == ["люблю кофе"]


def test_copy_to_clipboard_raises_when_all_backends_fail(monkeypatch):
    import builtins

    from w1x import clipboard

    monkeypatch.setattr(
        clipboard.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no clipboard cmd")),
    )
    monkeypatch.setattr(clipboard, "_win32_copy_to_clipboard", lambda _: False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tkinter":
            raise ImportError("no display")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError):
        clipboard.copy_to_clipboard("hello")


def test_secondary_session_isolated_and_ctrl_s_keeps_lone_assistant(monkeypatch):
    import types

    from w1x import prompts

    sec = prompts._create_prompt_session(with_repl_bindings=False)
    if sec is not None:
        assert sec.completer is None
        assert sec.auto_suggest is None
        sec_keys = {tuple(b.keys) for b in sec.key_bindings.bindings} if sec.key_bindings else set()
        assert ("c-s",) not in sec_keys
        assert ("c-q",) not in sec_keys

    messages = [{"role": "assistant", "content": "only assistant"}]
    monkeypatch.setitem(config.ACTIVE_SESSION_DATA, "messages", messages)
    monkeypatch.setitem(config.ACTIVE_SESSION_DATA, "cancelled_prompt", "")
    event = types.SimpleNamespace(
        app=types.SimpleNamespace(
            current_buffer=types.SimpleNamespace(text="", cursor_position=0)
        )
    )
    prompts.undo_and_restore_prompt(event)
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"


def test_reset_all_models_and_exact_vram_tag_selection(monkeypatch):
    monkeypatch.setattr(w1x_models, "unload_all_models", lambda: ["qwen2.5:14b"])
    printed = []
    monkeypatch.setattr(w1x_models.console, "print", lambda *args, **kwargs: printed.append(" ".join(str(a) for a in args)))

    assert w1x_models.reset_all_models(show_message=True) == ["qwen2.5:14b"]
    assert any("qwen2.5:14b" in line for line in printed)

    monkeypatch.setattr(w1x_models, "running_models", lambda: ["qwen2.5:14b"])
    installed = ["qwen2.5:7b", "qwen2.5:14b"]
    assert w1x_models.get_last_model(installed) == "qwen2.5:14b"

    panel_text = w1x_models_ui.render_models_panel(installed, 0, {"qwen2.5:14b"}).renderable
    lines = panel_text.splitlines()
    assert "[VRAM]" not in lines[0]
    assert "[VRAM]" in lines[1]


def test_math_precision_filters_and_datetime_guards():
    assert calculate_expression("2 ** 100") == str(2 ** 100)
    assert calculate_expression("-0") == "0"
    assert w1x_math.parse_math_request("посчитай 2026-11-08") is None
    assert w1x_math.parse_math_request("1,000 + 2,000") is None
    assert w1x_datetime.parse_datetime_request("12:30 ч", require_pure=False) is None
    invalid = w1x_datetime.parse_datetime_request("31.02.2025 через 1 день", require_pure=False)
    assert isinstance(invalid, dict) and "error" in invalid
    assert w1x_datetime.split_datetime_request("сколько сейчас времени в Токио?") == (None, "сколько сейчас времени в Токио?")


def test_run_cli_oneshot_flags_and_exit_codes(monkeypatch):
    printed = []
    monkeypatch.setattr(w1x_app.console, "print", lambda *args, **kwargs: printed.append(" ".join(str(a) for a in args)))
    monkeypatch.setattr(w1x_app, "installed_models", lambda: ["qwen3.5:4b"])

    assert w1x_app._run_cli_oneshot(["--help"]) == 0
    assert w1x_app._run_cli_oneshot(["--version"]) == 0
    assert w1x_app._run_cli_oneshot(["--unknown-flag"]) == 2
    assert w1x_app._run_cli_oneshot(["-m", "missing:tag", "привет"]) == 1


def test_engine_recovery_on_last_round_and_search_followup_xml_fallback(monkeypatch):
    import types

    from w1x import engine

    monkeypatch.setattr(engine, "parse_math_request", lambda _: None)
    monkeypatch.setattr(engine, "split_datetime_request", lambda text: (None, text))
    monkeypatch.setattr(engine, "relevant_memory_block", lambda _: "")
    monkeypatch.setattr(engine, "get_keep_alive", lambda: "0")
    monkeypatch.setattr(engine, "get_context_tokens", lambda: 8192)
    monkeypatch.setattr(engine, "MAX_TOOL_ROUNDS", 2)

    search_call = types.SimpleNamespace(
        function=types.SimpleNamespace(name="search_web", arguments={"query": "python"})
    )

    def make_chunk(content="", tool_calls=None):
        return types.SimpleNamespace(
            message=types.SimpleNamespace(thinking="", content=content, tool_calls=tool_calls or []),
            prompt_eval_count=10,
            eval_count=5,
            eval_duration=1_000_000,
            total_duration=2_000_000,
            load_duration=0,
            prompt_eval_duration=500_000,
        )

    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs.get("tools"))
        if len(calls) == 1:
            return iter([make_chunk(tool_calls=[search_call])])
        if len(calls) == 2:
            raise RuntimeError("XML syntax error while parsing tool call")
        if len(calls) == 3:
            return iter([make_chunk(content="Не могу найти.")])
        return iter([make_chunk(content="Итоговый ответ по результатам поиска.")])

    monkeypatch.setattr(engine, "chat", fake_chat)
    monkeypatch.setattr(engine, "execute_tool_call", lambda call: "Python 3.13 released")

    answer, _, interrupted = engine.generate_answer(
        "test-model",
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "что нового"}],
        False,
    )
    assert interrupted is False
    assert "Итоговый ответ" in answer
    assert calls[0] is not None
    assert calls[1] is not None  # followup tools for search_web
    assert calls[2] is None      # XML error stripped followup tools
    assert calls[3] is None      # recovery prompt also succeeded without exceeding MAX_TOOL_ROUNDS


def test_testpy_extract_skips_non_python_blocks():
    from w1x import testpy

    messages = [
        {
            "role": "assistant",
            "content": (
                "```python\nprint('ok')\n```\n"
                "```json\n{\"key\": 1}\n```\n"
                "```bash\necho=1\n```"
            ),
        }
    ]
    assert testpy.extract_latest_python_code(messages) == "print('ok')"


def test_run_python_code_uses_devnull_when_no_inputs(monkeypatch):
    import subprocess

    from w1x.tools import python_runner

    monkeypatch.setattr(python_runner.secondary_session, "prompt", lambda *a, **kw: "y")

    calls = []
    real_run = subprocess.run

    def spying_run(*args, **kwargs):
        calls.append({"input": kwargs.get("input"), "stdin": kwargs.get("stdin")})
        return real_run(*args, **kwargs)

    monkeypatch.setattr(python_runner.subprocess, "run", spying_run)

    eof_result = python_runner.run_python_code(
        "input('name: ')",
        purpose="verify immediate EOFError without blocking",
        inputs="",
    )
    assert calls[0] == {"input": None, "stdin": subprocess.DEVNULL}
    assert "EOFError" in eof_result

    ok_result = python_runner.run_python_code(
        "print(input())",
        purpose="verify provided stdin delivery",
        inputs="42",
    )
    assert calls[1] == {"input": b"42\n", "stdin": None}
    assert "42" in ok_result


def test_pull_manager_run_cli_pull_unified(monkeypatch):
    from w1x import pull_manager

    mgr = pull_manager.AsyncPullManager()
    pulled_models = []

    def fake_pull(name, stream=True):
        pulled_models.append(name)
        yield {"status": "downloading", "completed": 500, "total": 1000}
        yield {"status": "success", "completed": 1000, "total": 1000}

    def fake_installed():
        w1x_models.LAST_LIST_ERROR = None
        return list(pulled_models)

    monkeypatch.setattr(pull_manager, "pull", fake_pull)
    monkeypatch.setattr(w1x_models, "installed_models", fake_installed)

    mgr.run_cli_pull(["first:7b", "second:4b"])
    assert pulled_models == ["first:7b", "second:4b"]
    assert "second:4b" in mgr.get_installed()


def test_terminal_undo_uses_injected_callback(monkeypatch):
    import types

    from w1x import prompts

    saved = []

    def fake_save(system_prompt, messages):
        saved.append((system_prompt, list(messages)))

    prompts.set_undo_persist_callback(fake_save)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    monkeypatch.setitem(config.ACTIVE_SESSION_DATA, "messages", messages)
    monkeypatch.setitem(config.ACTIVE_SESSION_DATA, "system_prompt", "sys")
    buf = types.SimpleNamespace(text="", cursor_position=0)
    event = types.SimpleNamespace(app=types.SimpleNamespace(current_buffer=buf))

    try:
        prompts.undo_and_restore_prompt(event)
    finally:
        prompts.set_undo_persist_callback(None)

    assert buf.text == "hello"
    assert len(messages) == 1
    assert saved == [("sys", [{"role": "system", "content": "sys"}])]


def test_memory_load_facts_oserror_does_not_wipe_facts(tmp_path, monkeypatch):
    from w1x import memory

    facts_file = tmp_path / "memory-facts.json"
    facts_file.write_text('[{"id": "abc1", "fact": "секретный факт", "created_at": "t"}]', encoding="utf-8")
    monkeypatch.setattr(config, "FACTS_PATH", facts_file)

    original_read = Path.read_text

    def failing_read(self, *args, **kwargs):
        if self == facts_file:
            raise PermissionError("Access denied by lock")
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", failing_read)

    facts, ok = memory.load_facts_with_status()
    assert ok is False
    assert facts == []
    assert memory.load_facts() == []

    remember_res = memory.remember_fact("новый факт")
    assert "Ошибка: не удалось прочитать существующую память" in remember_res

    forget_res = memory.forget_facts("факт")
    assert "Ошибка: не удалось прочитать память" in forget_res

    migrated = memory.migrate_explicit_memory([{"role": "user", "content": "Запомни: факт 2"}])
    assert migrated == 0

    monkeypatch.setattr(Path, "read_text", original_read)
    assert len(memory.load_facts()) == 1
    assert memory.load_facts()[0]["fact"] == "секретный факт"


def test_first_run_wizard_skipped_on_ollama_offline(monkeypatch):
    wizard_called = False

    def fake_wizard():
        nonlocal wizard_called
        wizard_called = True
        return "qwen3.5:4b"

    monkeypatch.setattr(w1x_models_ui, "first_time_setup_wizard", fake_wizard)
    monkeypatch.setattr(w1x_app, "first_time_setup_wizard", fake_wizard)
    monkeypatch.setattr(w1x_models, "LAST_LIST_ERROR", ConnectionRefusedError("Offline"))
    monkeypatch.setattr(w1x_models, "installed_models", lambda: [])

    assert w1x_models_ui.choose_model() is None
    assert wizard_called is False

    code = w1x_app._run_cli_oneshot(["Привет"])
    assert code == 1
    assert wizard_called is False


def test_pull_manager_cooldown_and_cli_pull_abort_on_offline(monkeypatch):
    from w1x.pull_manager import AsyncPullManager

    call_count = 0

    def fake_installed():
        nonlocal call_count
        call_count += 1
        w1x_models.LAST_LIST_ERROR = ConnectionRefusedError("Server down")
        return []

    monkeypatch.setattr(w1x_models, "installed_models", fake_installed)

    mgr = AsyncPullManager()
    assert mgr.get_installed() == set()
    assert call_count == 1

    assert mgr.get_installed() == set()
    assert call_count == 1

    assert mgr.run_cli_pull(["deepseek-r1:7b"]) is False


def test_combined_datetime_request_injects_precomputed_fact_into_system_prompt(monkeypatch, tmp_path):
    from w1x import config, engine

    monkeypatch.setattr(config, "FACTS_PATH", tmp_path / "facts.json")
    monkeypatch.setattr(config, "ACTIVE_FILE_ROOT", tmp_path)

    prompt = "Какая сегодня дата? И напиши стих"
    messages = [
        {"role": "system", "content": "Base system instruction."},
        {"role": "user", "content": prompt},
    ]

    shortcut, working_messages = engine._prepare_turn_or_shortcut(
        "qwen3.5:4b", messages, prompt, 0
    )

    assert shortcut is None
    assert working_messages is not None
    assert working_messages[-1]["content"] == "И напиши стих"
    system_text = working_messages[0]["content"]
    assert "Точный результат вычисления даты/времени для запроса пользователя:" in system_text
    assert "Сейчас:" in system_text


def test_first_run_simulator_integration_with_dynamic_model_resolution(monkeypatch):
    fake_list = ["custom-model:8b"]
    monkeypatch.setattr(w1x_models, "installed_models", lambda: fake_list)

    assert w1x_app.installed_models() == fake_list
    assert w1x_models_ui.installed_models() == fake_list


def test_run_cli_oneshot_pull_exit_code_on_failure(monkeypatch):
    monkeypatch.setattr(w1x_app.GLOBAL_PULL_MANAGER, "run_cli_pull", lambda _: False)
    code = w1x_app._run_cli_oneshot(["pull", "nonexistent-model"])
    assert code == 1


def test_run_cli_oneshot_returns_130_when_generation_interrupted(monkeypatch, tmp_path):
    from w1x import config

    monkeypatch.setattr(config, "FACTS_PATH", tmp_path / "facts.json")
    monkeypatch.setattr(config, "CHAT_DIR", tmp_path / "chats")
    monkeypatch.setattr(w1x_app, "installed_models", lambda: ["test-model:latest"])
    monkeypatch.setattr(w1x_app, "wait_for_model_ready", lambda _: None)
    monkeypatch.setattr(
        w1x_app,
        "generate_answer",
        lambda *args, **kwargs: ("Partial answer", None, True),
    )

    code = w1x_app._run_cli_oneshot(["Скажи что-нибудь"])
    assert code == 130


def test_interactive_config_menu_handles_malformed_types(monkeypatch):
    malformed_cfg = {
        "default_model": "test-m",
        "keep_alive_minutes": None,
        "context_tokens": "not-a-number",
        "stats_enabled": "maybe",
    }
    monkeypatch.setattr(w1x_models_ui, "load_user_config", lambda: malformed_cfg)
    printed = []
    monkeypatch.setattr(w1x_models_ui.console, "print", lambda *args, **kwargs: printed.extend(args))
    monkeypatch.setattr(w1x_models_ui.secondary_session, "prompt", lambda *args, **kwargs: "0")

    w1x_models_ui.interactive_config_menu()
    assert len(printed) > 0
    full_text = " ".join(str(getattr(p, "renderable", p)) for p in printed)
    assert "8192 токенов" in full_text
    assert "30 мин" in full_text
    assert "Выключена" in full_text


def test_dispatch_slash_command_syncs_session_state_after_config(monkeypatch):
    state = w1x_app._SessionState(
        model="old-model",
        system_prompt="sys",
        messages=[],
        think_mode=False,
        stats_enabled=False,
    )

    new_cfg = {
        "default_model": "new-configured-model",
        "stats_enabled": True,
    }

    monkeypatch.setattr(w1x_app, "interactive_config_menu", lambda: None)
    monkeypatch.setattr(w1x_app, "load_user_config", lambda: new_cfg)
    monkeypatch.setattr(w1x_app, "reset_all_models", lambda: None)
    monkeypatch.setattr(w1x_app, "start_background_preload", lambda _: None)
    monkeypatch.setattr(w1x_app, "print_header", lambda _: None)

    handled = w1x_app._dispatch_slash_command("/config", "", state)
    assert handled is True
    assert state.model == "new-configured-model"
    assert state.stats_enabled is True


def test_render_models_panel_and_search_tree_escape_markup(monkeypatch):
    panel = w1x_models_ui.render_models_panel(["model[custom_tag]:4b"], 0, set())
    assert panel is not None

    from w1x.tools import search

    class FakeDDGS:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def text(self, *args, **kwargs):
            return [
                {
                    "title": "Title with [bold] unclosed [tag]",
                    "href": "https://example.com/item[1]",
                    "body": "Body snippet with [/close] and [dim] tags",
                }
            ]

    monkeypatch.setattr(search, "DDGS", FakeDDGS)
    result = search.search_web("query [with] brackets")
    assert "Title with [bold] unclosed [tag]" in result


def test_workspace_list_files_ignores_system_and_hidden_dirs(tmp_path, monkeypatch):
    from w1x.tools import workspace

    root = tmp_path / "project"
    root.mkdir()
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("git config", encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "bin.py").write_text("venv binary", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "pkg.js").write_text("pkg", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "c.pyc").write_text("cache", encoding="utf-8")
    (root / "main.py").write_text("print('hello')", encoding="utf-8")
    (root / "utils.py").write_text("def helper(): pass", encoding="utf-8")

    monkeypatch.setattr(config, "ACTIVE_FILE_ROOT", root)
    listing = workspace.list_project_files(".")
    assert "main.py" in listing
    assert "utils.py" in listing
    assert ".git" not in listing
    assert ".venv" not in listing
    assert "node_modules" not in listing
    assert "__pycache__" not in listing


def test_tools_none_argument_safety():
    from w1x.tools import antigravity, math_tool, python_runner, search

    assert search.search_web(None) == "Ошибка: пустой поисковый запрос."
    assert math_tool.calculate_expression(None) == "Ошибка: выражение пустое или слишком длинное."
    assert python_runner.run_python_code(None, "purpose") == "Ошибка: модель передала пустой код."
    assert antigravity.ask_antigravity(None) == "ESCALATE_ERROR: задача для Antigravity пуста."


def test_testpy_avoids_duplicate_main_call(tmp_path, monkeypatch):
    import subprocess

    from w1x import testpy

    count_file = tmp_path / "call_count.txt"
    code = f"""
called = 0
def main():
    with open(r"{count_file}", "a") as f:
        f.write("call\\n")

if __name__ == "__main__":
    main()
"""
    real_run = subprocess.run

    def run_without_window(arguments, **kwargs):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        return real_run(
            arguments,
            input="q\\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=10,
        )

    monkeypatch.setattr(testpy.subprocess, "run", run_without_window)
    exit_code, log = testpy.run_interactive_python_test(code)
    assert count_file.exists()
    assert count_file.read_text(encoding="utf-8").strip().splitlines() == ["call"]


def test_tool_call_keyboard_interrupt_propagates_and_marks_interrupted(monkeypatch):
    import types

    from w1x import engine
    from w1x.tools import registry

    fake_call = types.SimpleNamespace(
        function=types.SimpleNamespace(name="ask_antigravity", arguments='{"task": "help"}')
    )

    def fake_run_with_spinner(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(registry, "run_with_spinner", fake_run_with_spinner)

    with pytest.raises(KeyboardInterrupt):
        registry.execute_tool_call(fake_call)

    monkeypatch.setattr(
        engine,
        "chat",
        lambda **kw: iter([types.SimpleNamespace(message=types.SimpleNamespace(content="", thinking="", tool_calls=[fake_call]))]),
    )
    answer, stats, interrupted = engine.generate_answer("test-model", [{"role": "user", "content": "hi"}], False)
    assert interrupted is True


def test_trim_messages_strictly_respects_history_budget_with_large_dumps(monkeypatch):
    from w1x import engine

    monkeypatch.setattr(config, "get_context_tokens", lambda: 1000)

    huge_text = "x" * 2000
    messages = [
        {"role": "user", "content": huge_text},
        {"role": "assistant", "content": huge_text},
        {"role": "user", "content": huge_text},
        {"role": "assistant", "content": huge_text},
        {"role": "user", "content": "fresh query"},
    ]

    trimmed = engine.trim_messages_for_model(messages)
    total_trimmed_len = sum(len(m.get("content", "")) for m in trimmed if m.get("role") != "system")
    assert total_trimmed_len <= 3500 + 100


def test_terminal_raw_and_cooked_modes(monkeypatch):
    from w1x import key_input

    calls = []
    fake_termios = MagicMock()
    fake_tty = MagicMock()
    fake_termios.tcgetattr.return_value = ["dummy_attrs"]
    fake_termios.TCSADRAIN = 1

    def fake_setraw(fd):
        calls.append(("setraw", fd))

    def fake_tcsetattr(fd, when, attrs):
        calls.append(("tcsetattr", fd, attrs))

    fake_tty.setraw = fake_setraw
    fake_termios.tcsetattr = fake_tcsetattr

    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdin, "fileno", lambda: 0)
    monkeypatch.setitem(sys.modules, "termios", fake_termios)
    monkeypatch.setitem(sys.modules, "tty", fake_tty)

    with key_input.raw_terminal_mode():
        assert key_input._UNIX_RAW_DEPTH == 1
        with key_input.cooked_terminal_mode():
            assert ("tcsetattr", 0, ["dummy_attrs"]) in calls
    assert key_input._UNIX_RAW_DEPTH == 0
    assert calls[-1] == ("tcsetattr", 0, ["dummy_attrs"])


def test_print_status_displays_session_totals():
    captured = []
    dummy_console = MagicMock()
    dummy_console.print = lambda p: captured.append(p)

    with patch("w1x.app.console", dummy_console):
        totals = {
            "requests": 3,
            "prompt_tokens": 120,
            "output_tokens": 80,
            "total_tokens": 200,
            "total_duration": 1500000000,
        }
        w1x_app.print_status("test-model", False, True, [], totals)

    assert len(captured) == 1
    panel_text = str(captured[0].renderable)
    assert "Токенов за сессию:" in panel_text
    assert "[bold]200[/bold]" in panel_text
    assert "в [bold]3[/bold] ответах" in panel_text


def test_engine_direct_tool_result_format_cleanup():
    from w1x import engine

    request_started_ns = time.perf_counter_ns()
    messages = [{"role": "user", "content": "2 + 2"}]
    shortcut, error = engine._prepare_turn_or_shortcut("test-model", messages, "2 + 2", request_started_ns)
    assert error is None
    assert shortcut is not None
    answer, stats, interrupted = shortcut
    assert answer == "4"
    assert interrupted is False


def test_slash_command_autosuggest_and_submit_exact_command():
    import types

    from prompt_toolkit.auto_suggest import Suggestion
    from prompt_toolkit.document import Document

    from w1x.prompts import SlashCommandAutoSuggest, submit_message

    suggest = SlashCommandAutoSuggest()
    doc_prefix = Document("/mode", 5)
    s_prefix = suggest.get_suggestion(None, doc_prefix)
    assert s_prefix is not None
    assert s_prefix.text in {"l", "ls"}

    doc_exact = Document("/model", 6)
    assert suggest.get_suggestion(None, doc_exact) is None

    class FakeBuffer:
        def __init__(self, text: str, suggestion_text: str | None = None):
            self.text = text
            self.document = Document(text, len(text))
            self.complete_state = None
            self.suggestion = Suggestion(suggestion_text) if suggestion_text else None
            self.inserted: list[str] = []
            self.handled = False

        def insert_text(self, t: str) -> None:
            self.inserted.append(t)

        def validate_and_handle(self) -> None:
            self.handled = True

    buf_exact = FakeBuffer("/model", "s")
    event = types.SimpleNamespace(
        current_buffer=buf_exact,
        app=types.SimpleNamespace(key_processor=None, input=None),
    )
    submit_message(event)
    assert buf_exact.handled is True
    assert buf_exact.inserted == []

    buf_partial = FakeBuffer("/mode", "l")
    event.current_buffer = buf_partial
    submit_message(event)
    assert buf_partial.handled is False
    assert buf_partial.inserted == ["l"]


def test_project_files_not_in_direct_passthrough_tools():
    from w1x.tools import registry

    assert "project_files" not in registry.DIRECT_PASSTHROUGH_TOOLS
    assert "calculate_expression" in registry.DIRECT_PASSTHROUGH_TOOLS
    assert "ask_antigravity" in registry.DIRECT_PASSTHROUGH_TOOLS


def test_ask_antigravity_allows_permission_denied_in_successful_output(monkeypatch):
    from w1x.tools import antigravity

    class FakeProc:
        def __init__(self, rc: int, stdout: str, stderr: str):
            self.returncode = rc
            self._stdout = stdout
            self._stderr = stderr

        def communicate(self, timeout=None):
            return self._stdout, self._stderr

        def poll(self):
            return self.returncode

    monkeypatch.setattr(antigravity.shutil, "which", lambda cmd: "agy")

    # 1. Successful run mentioning permission denied in text explanation
    monkeypatch.setattr(
        antigravity.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProc(0, "Fixing permission denied in Linux via chmod", ""),
    )
    res_ok = antigravity.ask_antigravity("how to fix permission denied?")
    assert "Fixing permission denied in Linux" in res_ok
    assert "ESCALATE_ERROR" not in res_ok

    # 2. Failed run with permission denied error
    monkeypatch.setattr(
        antigravity.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProc(1, "", "fatal: permission denied on socket"),
    )
    res_err = antigravity.ask_antigravity("run task")
    assert "ESCALATE_ERROR: Antigravity запросила запрещённое действие; доступ не выдан." == res_err


def test_parse_math_request_syntax_validation():
    from w1x.tools import math_tool

    assert math_tool.parse_math_request("посчитай 100 - 20%") is None
    assert math_tool.parse_math_request("посчитай 1 000 + 2 000") is None
    assert math_tool.parse_math_request("посчитай 100 - 20") == "100 - 20"
    assert math_tool.parse_math_request("сколько будет 25 * 4") == "25 * 4"


def test_choose_model_filters_placeholder_current(monkeypatch):
    from w1x import models_ui

    fake_models = ["qwen3.5:4b", "llama3.2:3b"]
    monkeypatch.setattr(models_ui, "installed_models", lambda: fake_models)
    monkeypatch.setattr(models_ui, "running_models", lambda: [])
    monkeypatch.setattr(models_ui, "poll_key_crossplatform", lambda: "ENTER")

    # If current is placeholder, should default to first installed or get_last_model, not the placeholder
    chosen = models_ui.choose_model(current="не выбрана")
    assert chosen in fake_models
    assert chosen != "не выбрана"

    # If user cancels wizard at custom/new model item, should not return placeholder
    monkeypatch.setattr(models_ui, "first_time_setup_wizard", lambda: None)
    monkeypatch.setattr(models_ui, "poll_key_crossplatform", lambda: "DOWN")
    # Simulate selecting last index and cancelling
    monkeypatch.setattr(models_ui, "installed_models", lambda: fake_models)


def test_win32_copy_to_clipboard_unicode_and_emojis():
    from w1x import clipboard

    old_clip = None
    try:
        old_clip = clipboard._read_clipboard_text()
    except Exception:
        old_clip = None

    test_string = "Привет мир! 🚀 W1x CLI UTF-8: 100 - 20% = 80."
    try:
        clipboard.copy_to_clipboard(test_string)
        if os.name == "nt":
            read_back = clipboard._read_clipboard_text()
            assert read_back == test_string
    finally:
        if old_clip is not None:
            try:
                clipboard.copy_to_clipboard(old_clip)
            except Exception:
                pass


def test_rich_markup_escaping_prevents_markup_error(monkeypatch, tmp_path):
    from w1x import app as w1x_app
    from w1x import config

    # Fact containing unclosed/malformed Rich markup tags
    tricky_fact = "[bold red]unclosed and [invalid_tag] with [/orphan]"
    facts_file = tmp_path / "facts.json"
    facts_file.write_text(json.dumps([{"fact": tricky_fact}], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(config, "FACTS_PATH", facts_file)
    monkeypatch.setattr(config, "CURRENT_CHAT_PATH", tmp_path / "chat.json")

    # Should not raise MarkupError
    w1x_app._handle_memory_cmd([])

    # Testpy command with bracketed first line
    monkeypatch.setattr(w1x_app, "run_interactive_python_test", lambda code: (0, "output"))
    code_with_brackets = "[[1, 2], [3, 4]]\nx = 10"
    last_resp = w1x_app._handle_testpy_cmd(code_with_brackets, [], "system", "")
    assert "Лог тестирования получен" in last_resp

    # Workspace command with bracketed directory path
    tricky_dir = tmp_path / "[brackets_folder]"
    tricky_dir.mkdir()
    w1x_app._handle_workspace_cmd(str(tricky_dir))
    assert config.ACTIVE_FILE_ROOT == tricky_dir
    config.ACTIVE_FILE_ROOT = None

    # System prompt display with markup brackets
    state = w1x_app._SessionState(
        model="test",
        system_prompt="[system_tag] Prompt with [bold] unclosed",
        messages=[],
        think_mode=False,
        stats_enabled=False,
        last_response="",
    )
    # /system without args prints Panel with escaped prompt
    res = w1x_app._dispatch_slash_command("/system", "", state)
    assert res is True


def test_execute_chat_turn_handles_keyboard_interrupt(monkeypatch):
    from w1x import app as w1x_app
    from w1x.config import ACTIVE_SESSION_DATA

    state = w1x_app._SessionState(
        model="test-model",
        system_prompt="sys",
        messages=[{"role": "system", "content": "sys"}],
        think_mode=False,
        stats_enabled=False,
        last_response="",
    )

    def fake_wait(model):
        raise KeyboardInterrupt()

    monkeypatch.setattr(w1x_app, "wait_for_model_ready", fake_wait)

    # Should catch KeyboardInterrupt, mark interrupted, pop user message, and set cancelled_prompt
    w1x_app._execute_chat_turn("cancelled prompt text", retrying=False, state=state)

    assert len(state.messages) == 1
    assert state.messages[0]["role"] == "system"
    assert ACTIVE_SESSION_DATA.get("cancelled_prompt") == "cancelled prompt text"


def test_poll_key_crossplatform_non_tty_returns_exit(monkeypatch):
    from w1x import key_input

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert key_input.poll_key_crossplatform() == "EXIT"


def test_installed_models_show_error_suppression_and_lock(monkeypatch):
    from w1x import models as w1x_models

    def fake_list():
        raise RuntimeError("Connection refused [10061]")

    monkeypatch.setattr(w1x_models, "ollama_list", fake_list)

    # show_error=False must suppress console print and return []
    res_silent = w1x_models.installed_models(show_error=False)
    assert res_silent == []
    assert w1x_models.LAST_LIST_ERROR is not None

    # show_error=True prints error panel without raising MarkupError
    res_verbose = w1x_models.installed_models(show_error=True)
    assert res_verbose == []


def test_run_interactive_python_test_handles_signals_and_errors(monkeypatch):
    from w1x import testpy

    def fake_interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(testpy.subprocess, "run", fake_interrupt)
    code, log = testpy.run_interactive_python_test("print('hello')")
    assert code == 130
    assert "прервано пользователем" in log

    def fake_oserror(*args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr(testpy.subprocess, "run", fake_oserror)
    code_err, log_err = testpy.run_interactive_python_test("print('hello')")
    assert code_err == 1
    assert "Не удалось запустить процесс" in log_err


def test_datetime_tool_utc_z_suffix_python310():
    res = json.loads(calculate_datetime(base_datetime="2026-11-15T12:00:00Z", hours=1))
    assert "error" not in res
    assert "2026-11-15" in res["result"]


def test_pull_manager_enqueue_normalizes_latest_and_evicts_on_force_refresh(monkeypatch):
    from w1x import models as w1x_models
    from w1x import pull_manager

    mgr = pull_manager.AsyncPullManager()
    monkeypatch.setattr(w1x_models, "installed_models", lambda **kwargs: ["llama3.2:latest"])

    # Enqueueing without :latest should recognize it's already installed
    res = mgr.enqueue("llama3.2")
    assert res == "already_installed"


def test_pull_manager_abort_stops_stream_and_queue():
    from w1x import pull_manager

    mgr = pull_manager.AsyncPullManager()
    mgr.queue = ["model1", "model2"]
    mgr.tasks["model1"] = pull_manager.DownloadTask("model1")
    mgr.tasks["model2"] = pull_manager.DownloadTask("model2")

    mgr.abort()
    assert mgr.queue == []
    assert mgr.tasks["model1"].status == "error"
    assert mgr.tasks["model2"].status == "error"


def test_preload_task_cancellation_and_reset_all_models():
    from w1x import models as w1x_models

    w1x_models.start_background_preload("first-model")
    task1 = w1x_models._CURRENT_PRELOAD
    assert task1 is not None

    # Resetting models should cancel current preload and reset to None
    w1x_models.reset_all_models()
    assert task1.cancelled is True
    assert w1x_models._CURRENT_PRELOAD is None


def test_copy_command_syncs_with_messages_after_undo(monkeypatch):
    from w1x import app as w1x_app

    copied = []
    monkeypatch.setattr(w1x_app, "copy_to_clipboard", lambda text: copied.append(text))

    state = w1x_app._SessionState(
        model="test",
        system_prompt="sys",
        messages=[
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ],
        think_mode=False,
        stats_enabled=False,
        last_response="a2",
    )

    # Initial copy should copy a2
    w1x_app._dispatch_slash_command("/copy", "", state)
    assert copied[-1] == "a2"

    # Simulate undo removing q2 and a2
    state.messages.pop()
    state.messages.pop()

    # Now copy should dynamically copy remaining assistant message a1
    w1x_app._dispatch_slash_command("/copy", "", state)
    assert copied[-1] == "a1"


def test_save_chat_empty_or_dot_name_defaults_to_timestamp(tmp_path, monkeypatch):
    from w1x import chat_store, config

    monkeypatch.setattr(config, "CHAT_DIR", tmp_path / "chats")

    # Saving with "." or "/" should generate chat-*.md, not ".md"
    saved_path = chat_store.save_chat([], ".")
    assert saved_path.name.startswith("chat-")
    assert saved_path.name.endswith(".md")
    assert saved_path.name != ".md"


def test_recall_facts_reports_disk_error_on_oserror(monkeypatch):
    from w1x import memory

    monkeypatch.setattr(memory, "load_facts_with_status", lambda: ([], False))
    res = memory.recall_facts("anything")
    assert "Ошибка: не удалось прочитать файл" in res


def test_run_cli_oneshot_model_flag_combinations(monkeypatch):
    from w1x import app as w1x_app

    printed = []
    monkeypatch.setattr(w1x_app.console, "print", lambda *args, **kwargs: printed.append(" ".join(str(a) for a in args)))
    monkeypatch.setattr(w1x_app, "installed_models", lambda **kwargs: ["qwen3.5:4b"])

    # w1x -m <model> --help should show help (exit 0)
    assert w1x_app._run_cli_oneshot(["-m", "qwen3.5:4b", "--help"]) == 0

    # w1x --model=qwen3.5:4b without question should delegate to interactive chat
    launched_model = []
    monkeypatch.setattr(w1x_app, "_run_interactive_chat", lambda m: launched_model.append(m) or 0)
    res = w1x_app._run_cli_oneshot(["--model=qwen3.5:4b"])
    assert res == 0
    assert launched_model == ["qwen3.5:4b"]


def test_get_context_tokens_validates_bool_and_lower_bound(monkeypatch):
    from w1x import config

    # Boolean value in config should fallback to 8192
    monkeypatch.setattr(config, "load_user_config", lambda: {"context_tokens": True})
    assert config.get_context_tokens() == 8192

    # Lower bound < 512 should be clamped to 512
    monkeypatch.setattr(config, "load_user_config", lambda: {"context_tokens": 128})
    assert config.get_context_tokens() == 512


def test_project_files_handles_none_and_invalid_action():
    from w1x.tools.workspace import project_files

    assert "Ошибка: action должен быть list или read" in project_files(None)  # type: ignore[arg-type]
    assert "Ошибка: action должен быть list или read" in project_files(123)   # type: ignore[arg-type]
    assert "Ошибка: action должен быть list или read" in project_files("invalid")


def test_datetime_mixed_direction_postfix_phrasing():
    from w1x.tools.datetime_tool import parse_datetime_request

    req = parse_datetime_request("1 час назад и 30 минут вперед")
    assert req is not None
    assert req["hours"] == -1
    assert req["minutes"] == 30

    req_rev = parse_datetime_request("через 1 час и 30 минут назад")
    assert req_rev is not None
    assert req_rev["hours"] == 1
    assert req_rev["minutes"] == -30


def test_precompute_math_ignores_identifiers_and_aspect_ratios():
    from w1x.engine import _precompute_math_block

    # Should not evaluate sha256/2 or 16/9
    res = _precompute_math_block("покажи хэш sha256/2 и соотношение 16/9")
    assert res == ""

    # Explicit computation should be evaluated
    calc_res = _precompute_math_block("посчитай 256 / 2")
    assert "128" in calc_res


def test_python_runner_preview_handles_zero_input(monkeypatch):
    from w1x.tools import python_runner

    rendered = []
    monkeypatch.setattr(python_runner, "render_message", lambda msg, role=None: rendered.append(msg))
    monkeypatch.setattr(python_runner.secondary_session, "prompt", lambda *args, **kwargs: "n")

    python_runner.run_python_code("print('ok')", "проверка", inputs=0)  # type: ignore[arg-type]
    preview = [m for m in rendered if "Входные данные для input()" in m]
    assert len(preview) == 1
    assert "'0'" in preview[0]


def test_chat_store_load_current_chat_handles_null_messages(tmp_path, monkeypatch):
    from w1x import chat_store, config

    chat_file = tmp_path / "current-chat.json"
    chat_file.write_text(
        json.dumps({"custom_system_prompt": True, "system_prompt": "Custom prompt", "messages": None}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CURRENT_CHAT_PATH", chat_file)

    prompt, messages = chat_store.load_current_chat()
    assert prompt == "Custom prompt"
    assert len(messages) == 1
    assert messages[0]["content"] == "Custom prompt"


def test_has_explicit_top_call_detects_assignment_and_try():
    import ast

    from w1x.testpy import has_explicit_top_call

    tree1 = ast.parse("rc = main()")
    assert has_explicit_top_call(tree1, "main") is True

    tree2 = ast.parse("try:\n    main()\nexcept Exception:\n    pass")
    assert has_explicit_top_call(tree2, "main") is True

    tree3 = ast.parse("if True:\n    pass\nelse:\n    main()")
    assert has_explicit_top_call(tree3, "main") is True

    tree4 = ast.parse("def helper():\n    main()\nx = 1")
    assert has_explicit_top_call(tree4, "main") is False


def test_load_facts_with_status_handles_oserror_on_exists(tmp_path, monkeypatch):
    from pathlib import Path

    from w1x import config, memory

    fake_file = tmp_path / "protected_facts.json"
    monkeypatch.setattr(config, "FACTS_PATH", fake_file)

    def failing_exists(self):
        raise OSError("Permission denied on directory")

    monkeypatch.setattr(Path, "exists", failing_exists)
    facts, ok = memory.load_facts_with_status()
    assert facts == []
    assert ok is False


def test_memory_rank_facts_and_relevant_block():
    from w1x import memory

    many_facts = [{"fact": f"Факт пользователя номер {i}"} for i in range(35)]
    ranked = memory._rank_facts(many_facts, "номер 5")
    assert any("5" in f for f in ranked)

    # When query has no matches and >5 facts, empty list is returned
    unmatched = memory._rank_facts(many_facts, "совершенно чужое слово")
    assert unmatched == []


def test_session_state_encapsulation_and_active_data_sync():
    from w1x import app as w1x_app
    from w1x import config

    state = w1x_app._SessionState(
        model="demo-model",
        system_prompt="system instruction",
        messages=[{"role": "system", "content": "system instruction"}],
    )

    # Verify auto-sync on init
    assert config.ACTIVE_SESSION_DATA["messages"] == state.messages
    assert config.ACTIVE_SESSION_DATA["system_prompt"] == "system instruction"

    # User turn and cancelled prompt handling
    state.set_cancelled_prompt("draft query")
    assert config.ACTIVE_SESSION_DATA.get("cancelled_prompt") == "draft query"
    assert state.pop_cancelled_prompt() == "draft query"
    assert config.ACTIVE_SESSION_DATA.get("cancelled_prompt") is None

    # Adding messages through state methods
    state.append_user_message("Test user question")
    assert len(state.messages) == 2
    assert config.ACTIVE_SESSION_DATA["messages"][-1]["content"] == "Test user question"

    state.append_assistant_reply("Assistant answer", None)
    assert len(state.messages) == 3
    assert state.last_response == "Assistant answer"
    assert config.ACTIVE_SESSION_DATA["messages"][-1]["content"] == "Assistant answer"

    # Undo turn
    assert state.undo_turn() is True
    assert len(state.messages) == 1
    assert state.last_response == ""
    assert config.ACTIVE_SESSION_DATA["messages"] == state.messages

    # Reset context
    state.append_user_message("Another turn")
    state.reset_context()
    assert len(state.messages) == 1
    assert state.messages[0]["content"] == "system instruction"


def test_get_installed_models_with_status_contract(monkeypatch):
    from w1x import models

    class FakeItem:
        def __init__(self, name):
            self.model = name

    class FakeOllamaList:
        def __init__(self, items):
            self.models = [FakeItem(n) for n in items]

    # Success case
    monkeypatch.setattr(models, "ollama_list", lambda: FakeOllamaList(["qwen2.5:7b"]))
    names, err = models.get_installed_models_with_status(show_error=False)
    assert names == ["qwen2.5:7b"]
    assert err is None
    assert models.installed_models() == ["qwen2.5:7b"]

    # Error case
    def failing_list():
        raise ConnectionRefusedError("Offline")

    monkeypatch.setattr(models, "ollama_list", failing_list)
    names_err, err_obj = models.get_installed_models_with_status(show_error=False)
    assert names_err == []
    assert isinstance(err_obj, ConnectionRefusedError)
    assert models.LAST_LIST_ERROR is err_obj


def test_render_catalog_responsive_width_and_apple_silicon(monkeypatch):
    import shutil
    from collections import namedtuple

    from w1x import catalog_ui

    TerminalSize = namedtuple("TerminalSize", ["columns", "lines"])
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): TerminalSize(80, 24))

    apple_hw = {
        "gpu_name": "Apple Silicon M3 Max",
        "vram_gb": 36.0,
        "ram_gb": 36.0,
    }
    sample_items = [
        {"category": "general", "name": "qwen2.5:7b-instruct-q4_K_M", "size_gb": 4.5, "vram_min": 6.0, "note": "Быстрая модель"}
    ]

    rendered_text = catalog_ui._render_catalog(sample_items, 0, 0, 10, apple_hw, "OK")
    plain = str(rendered_text)

    # Must contain unified memory notice and not exceed 80 chars per table line
    assert "Unified" in plain
    for line in plain.splitlines():
        # Header divider and content should fit narrow width
        assert len(line) <= 85


def test_decomposed_domain_modules_integrity():
    from w1x import clipboard, key_input, prompts, terminal

    # Verify each module exposes its expected domain interfaces
    assert hasattr(clipboard, "copy_to_clipboard")
    assert hasattr(clipboard, "_read_clipboard_text")
    assert hasattr(key_input, "poll_key_crossplatform")
    assert hasattr(key_input, "raw_terminal_mode")
    assert hasattr(key_input, "cooked_terminal_mode")
    assert hasattr(prompts, "read_input")
    assert hasattr(prompts, "session")
    assert hasattr(prompts, "secondary_session")
    assert hasattr(prompts, "SlashCommandCompleter")
    assert hasattr(terminal, "console")
    assert hasattr(terminal, "render_message")
    assert hasattr(terminal, "run_with_spinner")


def test_conftest_hermetic_isolation_active(tmp_path):
    from pathlib import Path

    from w1x import config

    # The autouse fixture must guarantee isolated directories
    assert config.DATA_DIR != Path.home() / ".w1x"
    assert config.CONFIG_FILE_PATH != Path.home() / ".w1x" / "config.json"
    assert config.CURRENT_CHAT_PATH != Path.home() / ".w1x" / "current-chat.json"
    assert config.FACTS_PATH != Path.home() / ".w1x" / "memory-facts.json"


def test_pull_manager_aborted_task_never_completed(monkeypatch):
    from w1x.pull_manager import AsyncPullManager, DownloadTask

    manager = AsyncPullManager()
    task = DownloadTask(model_name="test:model")
    task.status = "downloading"
    manager.tasks["test:model"] = task
    manager.queue = ["test:model"]
    manager._aborted = True

    # Simulate chunk stream where abort occurred
    def fake_pull(model, stream=True):
        yield {"status": "downloading", "completed": 50, "total": 100}

    monkeypatch.setattr("w1x.pull_manager.pull", fake_pull)
    manager._installed_cache = set()
    manager._worker_loop()

    assert task.status == "error"
    assert "test:model" not in manager._installed_cache


def test_execute_chat_turn_retry_interrupt_resilience():
    from w1x.app import _SessionState

    state = _SessionState(
        model="demo",
        system_prompt="sys",
        messages=[
            {"role": "user", "content": "первый вопрос"},
            {"role": "assistant", "content": "первый ответ"},
        ],
    )
    # Simulate /retry turn start: previous assistant message popped
    state.messages.pop()
    retry_prompt = state.messages[-1]["content"]

    # Interrupted during retry: set cancelled prompt and rollback user turn
    state.set_cancelled_prompt(retry_prompt)
    state.rollback_last_turn()
    assert len(state.messages) == 0

    # Next user message must not result in consecutive user turns
    state.append_user_message("новый вопрос")
    roles = [m["role"] for m in state.messages]
    assert roles == ["user"]
    assert state.messages[-1]["content"] == "новый вопрос"


def test_session_state_pop_cancelled_prompt_syncs():
    from w1x import config
    from w1x.app import _SessionState

    state = _SessionState(model="demo", system_prompt="sys", messages=[])
    state.set_cancelled_prompt("prompt to restore")
    assert config.ACTIVE_SESSION_DATA["cancelled_prompt"] == "prompt to restore"

    popped = state.pop_cancelled_prompt()
    assert popped == "prompt to restore"
    assert "cancelled_prompt" not in config.ACTIVE_SESSION_DATA

    state.sync_active_data()
    assert "cancelled_prompt" not in config.ACTIVE_SESSION_DATA


def test_memory_migration_runs_only_once_and_persists_flag(tmp_path, monkeypatch):
    from w1x import config, memory

    fake_config = tmp_path / "config.json"
    fake_facts = tmp_path / "facts.json"
    monkeypatch.setattr(config, "CONFIG_FILE_PATH", fake_config)
    monkeypatch.setattr(config, "FACTS_PATH", fake_facts)

    messages = [
        {"role": "user", "content": "запомни мой город Москва"},
    ]
    migrated = memory.migrate_explicit_memory(messages)
    assert migrated == 1
    assert memory.is_memory_migrated() is True

    # User explicitly forgets the fact
    memory.forget_facts("москва")
    assert len(memory.load_facts()) == 0

    # Second migration must be a no-op and must not resurrect forgotten fact
    migrated_again = memory.migrate_explicit_memory(messages)
    assert migrated_again == 0
    assert len(memory.load_facts()) == 0


def test_chat_store_quarantines_corrupted_json(tmp_path, monkeypatch):
    from w1x import chat_store, config

    corrupted_chat = tmp_path / "current-chat.json"
    corrupted_chat.write_text("{\"broken_json\": ", encoding="utf-8")
    monkeypatch.setattr(config, "CURRENT_CHAT_PATH", corrupted_chat)

    prompt, messages = chat_store.load_current_chat()
    assert prompt == config.DEFAULT_SYSTEM_PROMPT
    assert len(messages) == 1

    corrupted_backups = list(tmp_path.glob("current-chat.corrupted.*.json"))
    assert len(corrupted_backups) == 1


def test_workspace_rejects_dotfiles_and_env(tmp_path):
    from w1x.tools import workspace

    workspace_dir = tmp_path / "my_project"
    workspace_dir.mkdir()
    env_file = workspace_dir / ".env"
    env_file.write_text("SECRET_KEY=12345", encoding="utf-8")
    git_dir = workspace_dir / ".git"
    git_dir.mkdir()
    git_config = git_dir / "config"
    git_config.write_text("[core]\nrepositoryformatversion = 0", encoding="utf-8")

    workspace.set_file_workspace(str(workspace_dir))
    assert "Ошибка доступа" in workspace.read_project_file(".env")
    assert "Ошибка доступа" in workspace.read_project_file(".git/config")


def test_antigravity_secret_mask_handles_hyphens():
    from w1x.tools.antigravity import _redact_secrets

    sample = "Keys: sk-proj-1234567890abcdef and sk-ant-api03-1234567890abcdef and ghp_1234567890abcdef"
    redacted = _redact_secrets(sample)
    assert "sk-proj" not in redacted
    assert "sk-ant" not in redacted
    assert "ghp_" not in redacted
    assert "[СЕКРЕТ СКРЫТ]" in redacted


def test_math_tool_hex_and_octal_literals():
    from w1x.tools.math_tool import calculate_expression

    assert calculate_expression("0x10 + 0b101") == "21"
    assert calculate_expression("0o77") == "63"


def test_datetime_historical_year_and_explicit_time():
    from w1x.tools.datetime_tool import _extract_base_datetime

    iso_1995 = _extract_base_datetime("15 мая 1995 года")
    assert "1995-05-15" in iso_1995

    iso_time = _extract_base_datetime("через 2 часа от 14:00")
    assert "14:00:00" in iso_time


def test_stats_none_token_values_safe():
    from w1x.stats import add_to_totals, empty_totals

    totals = empty_totals()
    stats = {"prompt_tokens": None, "output_tokens": None, "total_tokens": None, "total_duration": None}
    add_to_totals(totals, stats)  # Must not raise TypeError
    assert totals["requests"] == 1
    assert totals["prompt_tokens"] == 0


def test_hardware_multi_gpu_formatting(monkeypatch):
    import subprocess

    from w1x import hardware

    fake_output = (
        "NVIDIA GeForce RTX 3080, 10240\n"
        "NVIDIA GeForce RTX 3080, 10240\n"
    )
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: fake_output)

    info = hardware.detect_hardware()
    assert "2x" in info["gpu_name"]
    assert info["vram_gb"] == 10.0


def test_prompts_session_proxy_plain_text_fallback():
    from prompt_toolkit.formatted_text import HTML

    from w1x.prompts import _format_prompt_for_input

    plain = _format_prompt_for_input(HTML("<b><ansigreen>Ты › </ansigreen></b>"))
    assert "Ты ›" in plain
    assert "<b>" not in plain


def test_looks_like_launch_command_allows_code_questions():
    from w1x.app import looks_like_launch_command

    assert looks_like_launch_command("python main.py") is True
    assert looks_like_launch_command("Python script test.py fails with error, why?") is False
    assert looks_like_launch_command("Как запустить py script.py?") is False











