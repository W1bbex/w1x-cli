"""Безопасный AST-вычислитель и распознавание математических запросов."""
import ast
import decimal
import operator
import re
from collections.abc import Callable, Sequence
from decimal import Decimal

from ..types import ChatMessage


def _decimal_floordiv(a: Decimal, b: Decimal) -> Decimal:
    q = a // b
    r = a % b
    if r != 0 and ((a < 0) != (b < 0)):
        q -= 1
    return q


def _decimal_mod(a: Decimal, b: Decimal) -> Decimal:
    r = a % b
    if r != 0 and ((a < 0) != (b < 0)):
        r += b
    return r


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Decimal, Decimal], Decimal]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: _decimal_floordiv,
    ast.Mod: _decimal_mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Decimal], Decimal]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_math_node(node: ast.AST, source: str = "") -> Decimal:
    if isinstance(node, ast.Expression):
        return _evaluate_math_node(node.body, source)
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        if isinstance(node.value, int):
            return Decimal(node.value)
        if source:
            segment = ast.get_source_segment(source, node)
            if segment:
                return Decimal(segment.strip().replace("_", ""))
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_math_node(node.operand, source))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_math_node(node.left, source)
        right = _evaluate_math_node(node.right, source)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 1000:
                raise ValueError("Слишком большая степень")
            if right == int(right):
                return left ** int(right)
            return left ** right
        return _BINARY_OPERATORS[type(node.op)](left, right)
    raise ValueError("Разрешены только числа, скобки и операции + - * / // % **")


def _format_math_result(val: Decimal | object) -> str:
    if isinstance(val, Decimal):
        if not val.is_finite():
            return str(val)
        if val == 0:
            return "0"
        val_norm = val.normalize()
        if val_norm == 0:
            return "0"
        _sign, _digits, exponent = val_norm.as_tuple()
        if isinstance(exponent, int) and exponent > 0:
            return str(int(val_norm))
        formatted = f"{val_norm:f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return "0" if formatted in {"", "-0"} else formatted
    return str(val)


def calculate_expression(expression: str) -> str:
    """Точно вычислить арифметическое выражение без приблизительного счёта.

    Args:
        expression: Выражение с числами, скобками и операциями + - * / // % **.

    Returns:
        Точный результат вычисления или описание ошибки.
    """
    expression = str("" if expression is None else expression).strip()
    if not expression or len(expression) > 300:
        return "Ошибка: выражение пустое или слишком длинное."
    try:
        tree = ast.parse(expression, mode="eval")
        with decimal.localcontext() as ctx:
            ctx.prec = max(ctx.prec, 100)
            result = _evaluate_math_node(tree, expression)
            return _format_math_result(result)
    except Exception as error:
        return f"Ошибка вычисления: {type(error).__name__}: {error}"


_MATH_ALLOWED_WORDS = {
    "а", "будет", "вычисли", "вычислить", "калькулятором", "математикой",
    "пожалуйста", "посчитай", "проверь", "результат", "сколько", "точно",
    "чему", "равно", "равен", "теперь",
}


def _is_non_math_literal(candidate: str) -> bool:
    if re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", candidate):
        return True
    if re.search(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b", candidate):
        return True
    if re.search(r"\b\d{1,3}(?:,\d{3})+\b", candidate):
        return True
    if re.search(r"^\+?\d[\d\s()-]{6,}\d$", candidate) and not re.search(r"[+*/%]|(?:\s-\s)", candidate.lstrip("+")):
        return True
    return False


def parse_math_request(text: str) -> str | None:
    """Извлечь чистое арифметическое выражение из текста пользователя."""
    normalized = text.lower().replace("×", "*").replace("÷", "/")
    if re.search(r"не\s+(?:используй|вызывай).*калькулятор", normalized):
        return None
    candidates = re.findall(r"(?<![\w.])[\d\s().,+\-*/%]{3,}(?![\w.])", normalized)
    candidates = [candidate.strip(" ,.") for candidate in candidates]
    candidates = [
        candidate
        for candidate in candidates
        if re.search(r"[+\-*/%]", candidate)
        and re.search(r"\d", candidate)
        and not _is_non_math_literal(candidate)
    ]
    if not candidates:
        return None
    raw_best = str(max(candidates, key=len))
    expression = raw_best.replace(",", ".").strip()
    remaining = normalized.replace(raw_best, " ")
    words = set(re.findall(r"[a-zа-яё]+", remaining))
    if words and not words <= _MATH_ALLOWED_WORDS:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
        _evaluate_math_node(tree, expression)
    except Exception:
        return None
    return expression


def previous_math_expression(messages: Sequence[ChatMessage]) -> str | None:
    """Найти последнее математическое выражение в истории сообщений пользователя."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        expression = parse_math_request(message.get("content", ""))
        if expression:
            return expression
    return None


def needs_clarification(text: str) -> bool:
    """Определить короткие неопределённые запросы, требующие уточнения."""
    normalized = " ".join(text.casefold().strip(" .!?,-").split())
    vague_patterns = [
        r"(?:сделай|повтори|реши|запусти|удали)(?:\s+(?:это|там|так))?(?:\s+как\s+(?:вчера|раньше))?",
        r"(?:я\s+же\s+)?(?:сказал|писал)\s+уже",
        r"сделай\s+как\s+(?:вчера|раньше)",
    ]
    return len(normalized) < 80 and any(
        re.fullmatch(pattern, normalized) for pattern in vague_patterns
    )
