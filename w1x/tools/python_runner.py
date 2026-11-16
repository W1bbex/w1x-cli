"""Локальный запуск Python-кода с обязательным подтверждением пользователя (без системной песочницы)."""
import subprocess
import sys
import tempfile

from prompt_toolkit.formatted_text import HTML
from rich.panel import Panel
from rich.syntax import Syntax

from ..prompts import secondary_session
from ..terminal import console, render_message


def run_python_code(code: str, purpose: str, inputs: str = "") -> str:
    r"""Проверить небольшой фрагмент Python-кода на компьютере пользователя.

    Args:
        code: Полный Python-код для проверки.
        purpose: Коротко: что проверяет код и зачем это нужно для ответа.
        inputs: Необязательно: текстовые строки для input(), через \n (например: "2\n*\n5\n").

    Returns:
        Код завершения, stdout и stderr программы.
    """
    code = str("" if code is None else code).strip()
    if not code:
        return "Ошибка: модель передала пустой код."
    if len(code) > 8000:
        return "Ошибка: код длиннее 8000 символов и не будет запущен."

    purpose = " ".join(str(purpose or "").split())
    if not purpose:
        return "Ошибка: модель не объяснила, зачем ей запускать Python-код."

    console.print()
    render_message("Зачем нужен Python ›", role="assistant_label")
    render_message(purpose[:600], role="assistant")
    has_inputs = inputs is not None and str(inputs) != ""
    if has_inputs:
        input_lines = str(inputs).splitlines() or [""]
        formatted_inputs = ", ".join(repr(line) for line in input_lines)
        render_message(f"Входные данные для input() › {formatted_inputs}", role="assistant_label")
    console.print()
    console.print(
        Panel(
            Syntax(code, "python", theme="monokai", word_wrap=True, padding=0),
            title="Python просит разрешение на запуск",
            border_style="yellow",
        )
    )
    console.print(
        "[yellow]Для подтверждения введите [bold]y[/bold] или [bold]да[/bold]. "
        "Нажатие Enter или любой другой ввод отклонит запуск.[/yellow]"
    )
    try:
        confirmation = secondary_session.prompt(
            HTML("<b><ansiyellow>Разрешить запуск кода? [y/N] › </ansiyellow></b>"),
            multiline=False,
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        confirmation = "n"
    if confirmation not in {"y", "yes", "д", "да"}:
        return "Пользователь отказался запускать код."

    with tempfile.TemporaryDirectory(prefix="ollama-python-", ignore_cleanup_errors=True) as temp_dir:
        try:
            has_inputs = inputs is not None and str(inputs) != ""
            completed = subprocess.run(
                [sys.executable, "-I", "-X", "utf8", "-c", code],
                input=(str(inputs) + "\n").encode("utf-8") if has_inputs else None,
                stdin=None if has_inputs else subprocess.DEVNULL,
                cwd=temp_dir,
                capture_output=True,
                timeout=10,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            stdout = completed.stdout.decode("utf-8", errors="replace")[-12000:].rstrip()
            stderr = completed.stderr.decode("utf-8", errors="replace")[-12000:].rstrip()
            return_code = completed.returncode
        except subprocess.TimeoutExpired as error:
            stdout_err = (
                (error.stdout.decode("utf-8", errors="replace") if error.stdout else "")
                + (error.stderr.decode("utf-8", errors="replace") if error.stderr else "")
            )[-12000:]
            return f"Выполнение остановлено через 10 секунд.\n{stdout_err}"
        except Exception as error:
            return f"Ошибка запуска: {type(error).__name__}: {error}"

    parts = [f"Код завершения: {return_code}"]
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    if not stdout and not stderr:
        if return_code == 0:
            parts.append(
                "Программа завершилась успешно (код 0), но в stdout ничего не выведено "
                "(код выполнился без ошибок, но без явного вызова print())."
            )
        else:
            parts.append("Программа завершилась с ошибкой (код != 0) без вывода в stdout.")
    return "\n\n".join(parts)
