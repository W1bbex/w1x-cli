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

    from w1x import terminal

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
        monkeypatch.setattr(terminal.os, "read", fake_read)
        fake_stdin = types.SimpleNamespace(isatty=lambda: True, fileno=lambda: 0)
        monkeypatch.setattr(terminal.sys, "stdin", fake_stdin)
        return terminal._poll_key_unix()

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

    from w1x import terminal

    monkeypatch.setattr(
        terminal.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no clipboard cmd")),
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tkinter":
            raise ImportError("no display")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError):
        terminal.copy_to_clipboard("hello")


def test_secondary_session_isolated_and_ctrl_s_keeps_lone_assistant(monkeypatch):
    import types

    from w1x import terminal

    sec = terminal._create_prompt_session(with_repl_bindings=False)
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
    terminal.undo_and_restore_prompt(event)
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

    from w1x import terminal

    saved = []

    def fake_save(system_prompt, messages):
        saved.append((system_prompt, list(messages)))

    terminal.set_undo_persist_callback(fake_save)
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
        terminal.undo_and_restore_prompt(event)
    finally:
        terminal.set_undo_persist_callback(None)

    assert buf.text == "hello"
    assert len(messages) == 1
    assert saved == [("sys", [{"role": "system", "content": "sys"}])]




