"""Реестр и диспетчер вызова встроенных инструментов W1x."""
import json
from collections.abc import Callable
from typing import Any

from ..memory import memory
from ..terminal import console, render_message, run_with_spinner
from .antigravity import ask_antigravity, cancel_active_agy
from .datetime_tool import calculate_datetime
from .math_tool import calculate_expression
from .python_runner import run_python_code
from .search import search_web
from .workspace import project_files

AVAILABLE_TOOLS: dict[str, Callable[..., Any]] = {
    "search_web": search_web,
    "calculate_datetime": calculate_datetime,
    "calculate_expression": calculate_expression,
    "run_python_code": run_python_code,
    "memory": memory,
    "project_files": project_files,
    "ask_antigravity": ask_antigravity,
}
TOOL_FUNCTIONS: list[Callable[..., Any]] = list(AVAILABLE_TOOLS.values())

DIRECT_PASSTHROUGH_TOOLS = {
    "calculate_expression",
    "ask_antigravity",
}


def execute_tool_call(call: Any) -> str:
    """Выполнить один tool-call модели с единой обработкой ошибок и отмены."""
    name = call.function.name
    arguments = call.function.arguments or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return f"Инструмент {name} получил неправильные аргументы: {arguments}"
    if not isinstance(arguments, dict):
        return f"Инструмент {name} получил аргументы неправильного типа."
    function = AVAILABLE_TOOLS.get(name)
    if function is None:
        return f"Неизвестный инструмент: {name}"

    labels = {
        "search_web": f"Ищу в интернете: {arguments.get('query', '')}",
        "calculate_datetime": "Вычисляю точные дату и время",
        "calculate_expression": f"Считаю: {arguments.get('expression', '')}",
        "run_python_code": "Модель хочет проверить Python-код",
        "memory": f"Постоянная память: {arguments.get('action', '')}",
        "project_files": f"Файлы проекта, только чтение: {arguments.get('action', '')}",
        "ask_antigravity": "Передаю сложную задачу Antigravity",
    }
    console.print()
    render_message(labels.get(name, name), role="tool")
    try:
        if name == "ask_antigravity":
            return str(
                run_with_spinner(
                    "Antigravity думает...",
                    lambda: function(**arguments),
                    on_interrupt=cancel_active_agy,
                )
            )
        return str(function(**arguments))
    except KeyboardInterrupt:
        raise
    except Exception as error:
        return f"Ошибка инструмента {name}: {type(error).__name__}: {error}"
