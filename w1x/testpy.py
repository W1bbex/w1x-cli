# -*- coding: utf-8 -*-
"""Модуль интерактивного запуска и отладки Python-кода в отдельном окне терминала CMD."""
import os
import re
import sys
import ast
import shutil
import tempfile
import subprocess
from pathlib import Path

from .terminal import console, render_message


def is_valid_python(code_str: str) -> bool:
    """Проверить, является ли строка валидным и осмысленным Python-кодом."""
    cleaned = code_str.strip()
    if not cleaned:
        return False
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
    if len(lines) <= 2 and any(
        lines[0].lower().startswith(cmd)
        for cmd in ("python ", "python3 ", "py ", "pip ", "cd ", "npm ", "git ")
    ):
        return False
    try:
        tree = ast.parse(cleaned)
        has_logic = any(
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.For,
                    ast.While,
                    ast.If,
                    ast.Assign,
                    ast.Expr,
                ),
            )
            for node in ast.walk(tree)
        )
        return has_logic
    except SyntaxError:
        return False


def extract_latest_python_code(messages):
    """Извлечь последний реальный блок Python-кода из ответов ассистента."""
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")

        # 1. Ищем блоки в тройных кавычках ``` ... ```
        blocks = list(re.finditer(r"```(?:[a-zA-Z0-9_-]+)?\r?\n([\s\S]*?)```", content))
        for match in reversed(blocks):
            candidate = match.group(1).strip()
            if is_valid_python(candidate):
                return candidate

        # 2. Если тройных кавычек нет, ищем непрерывный блок кода по ключевым словам
        lines = content.splitlines()
        code_candidates = []
        current_chunk = []
        for line in lines:
            if re.match(
                r"^\s*(?:def\s|import\s|from\s|class\s|if\s|for\s|while\s|try:|#|print\(|input\()",
                line,
            ) or (
                current_chunk
                and (
                    line.startswith("    ")
                    or line.startswith("\t")
                    or line.strip() == ""
                    or line.strip().startswith("#")
                )
            ):
                current_chunk.append(line)
            else:
                if current_chunk:
                    chunk_text = "\n".join(current_chunk).strip()
                    if is_valid_python(chunk_text):
                        code_candidates.append(chunk_text)
                    current_chunk = []
        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if is_valid_python(chunk_text):
                code_candidates.append(chunk_text)

        if code_candidates:
            return max(code_candidates, key=len)

    return ""


def run_interactive_python_test(code: str) -> tuple[int, str]:
    """Запустить интерактивное тестирование Python-кода в отдельном окне CMD."""
    temp_dir = Path(tempfile.mkdtemp(prefix="w1x-testpy-"))
    script_file = temp_dir / "user_script.py"
    runner_file = temp_dir / "runner.py"
    log_file = temp_dir / "session_log.txt"
    exit_code_file = temp_dir / "exit_code.txt"

    script_file.write_text(code, encoding="utf-8")

    runner_code = """# -*- coding: utf-8 -*-
import os
import sys
import traceback
import builtins

base_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(base_dir, "session_log.txt")
exit_code_path = os.path.join(base_dir, "exit_code.txt")
script_file_path = os.path.join(base_dir, "user_script.py")

# Инициализируем файлы по умолчанию
try:
    with open(exit_code_path, "w", encoding="utf-8") as f:
        f.write("1")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("")
except Exception:
    pass


class TeeStream:
    def __init__(self, original, collector):
        self.original = original
        self.collector = collector

    def write(self, data):
        try:
            self.original.write(data)
            self.original.flush()
        except Exception:
            pass
        self.collector.append(data)

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.original, "isatty", lambda: True)()

    def __getattr__(self, name):
        return getattr(self.original, name)


def clear_input_buffer():
    if os.name == "nt":
        try:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getch()
        except Exception:
            pass


def main():
    if os.name == "nt":
        os.system("chcp 65001 >nul 2>&1")
        os.system("title W1x — Интерактивное тестирование Python")

    run_count = 0
    all_sessions_log = []
    final_exit_code = 0

    with open(script_file_path, "r", encoding="utf-8", errors="replace") as f:
        code_text = f.read()

    orig_input = builtins.input

    while True:
        run_count += 1
        if run_count == 1:
            if os.name == "nt":
                os.system("cls")
            else:
                os.system("clear")

        print("=" * 64)
        print(f"  [W1x Chat] Интерактивное тестирование (Запуск #{run_count})")
        print("=" * 64 + "\\n")

        session_log = []
        inputs_called = 0

        def logged_input(prompt=""):
            nonlocal inputs_called
            inputs_called += 1
            val = orig_input(prompt)
            session_log.append(str(val) + "\\n")
            return val

        builtins.input = logged_input
        tee_out = TeeStream(sys.__stdout__, session_log)
        tee_err = TeeStream(sys.__stderr__, session_log)

        sys.stdout = tee_out
        sys.stderr = tee_err
        exit_code = 0

        try:
            global_scope = {
                "__name__": "__main__",
                "__file__": script_file_path,
                "__doc__": None,
            }
            exec(compile(code_text, script_file_path, "exec"), global_scope)

            # Если модель только объявила def main() / calculator() и забыла вызвать внизу
            if not session_log and not inputs_called:
                entry_points = [
                    "main",
                    "calculator",
                    "simple_calculator",
                    "calc",
                    "run",
                    "app",
                    "start",
                    "cli",
                ]
                for ep in entry_points:
                    if ep in global_scope and callable(global_scope[ep]):
                        global_scope[ep]()
                        break
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else (1 if exc.code is not None else 0)
        except KeyboardInterrupt:
            exit_code = 130
            print("\\n[Выполнение остановлено пользователем (Ctrl+C)]")
        except Exception:
            exit_code = 1
            traceback.print_exc()
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            builtins.input = orig_input

        final_exit_code = exit_code
        run_text = "".join(session_log)
        all_sessions_log.append(f"--- Запуск #{run_count} (код {exit_code}) ---\\n{run_text}")

        # Сохраняем общий лог после каждого запуска
        with open(log_file_path, "w", encoding="utf-8", errors="replace") as f:
            f.write("\\n\\n".join(all_sessions_log))
        with open(exit_code_path, "w", encoding="utf-8") as f:
            f.write(str(final_exit_code))

        print("\\n" + "=" * 64)
        if exit_code == 0:
            print(f"  ✓ Запуск #{run_count} успешно завершён (код 0).")
        else:
            print(f"  ⚠ Запуск #{run_count} завершился с ошибкой (код {exit_code}).")

        if inputs_called == 0:
            print("  ℹ В этом коде нет функции input(): он выполнился со своими данными.")

        print("  ─────────────────────────────────────────────────────────────")
        print("  [Enter] или [R] — Запустить этот код ещё раз")
        print("  [Q]             — Завершить тестирование и передать лог в чат")
        print("=" * 64)

        while True:
            clear_input_buffer()
            try:
                choice = orig_input("  Ваш выбор (Enter/R/Q) › ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                choice = "q"

            if choice in {"q", "quit", "exit", "й", "выход"}:
                return exit_code
            elif choice in {"", "r", "к", "run", "restart"}:
                break
            else:
                print(f"  [Неизвестная команда '{choice}'. Введите Q для выхода или Enter для перезапуска]")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\\n[Критическая ошибка раннера: {exc}]")
        traceback.print_exc()
        try:
            input("\\n[Нажмите Enter для выхода]...")
        except Exception:
            pass
"""
    runner_file.write_text(runner_code, encoding="utf-8")

    console.print()
    render_message("Открываю отдельное окно CMD для тестирования…", role="tool")
    console.print("[dim]Окно открыто. Протестируйте программу и завершите её для возврата в чат.[/dim]")

    creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0
    exit_code = 0
    log_content = ""
    try:
        subprocess.run(
            [sys.executable, str(runner_file)],
            creationflags=creation_flags,
            check=False,
        )

        if exit_code_file.exists():
            try:
                exit_code = int(exit_code_file.read_text(encoding="utf-8").strip())
            except Exception:
                exit_code = 0

        if log_file.exists():
            try:
                log_content = log_file.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                log_content = ""
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    return exit_code, log_content
