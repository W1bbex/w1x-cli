import json
import pytest
from w1x.tools import calculate_datetime, calculate_expression, _format_offset
from datetime import timedelta


def test_calculate_datetime_mixed_offsets():
    # days=1, hours=-2 -> net offset should be +22 hours forward, NOT "1 день 2 часа вперёд"
    res_raw = calculate_datetime(days=1, hours=-2, base_datetime="2026-01-01T12:00:00")
    data = json.loads(res_raw)
    answer = data["answer"]
    assert "22 часа вперёд" in answer
    assert "2026-01-02T10:00:00" in data["result"]


def test_calculate_datetime_negative_offset():
    # days=-1, hours=-2 -> net offset should be 1 day 2 hours backward
    res_raw = calculate_datetime(days=-1, hours=-2, base_datetime="2026-01-05T12:00:00")
    data = json.loads(res_raw)
    answer = data["answer"]
    assert "1 день 2 часа назад" in answer
    assert "2026-01-04T10:00:00" in data["result"]


def test_calculate_expression_exact_float_precision():
    # 0.1 + 0.2 must be strictly "0.3" without IEEE 754 float drift
    res = calculate_expression("0.1 + 0.2")
    assert res == "0.3"

    res = calculate_expression("100 - 0.01")
    assert res == "99.99"

    res = calculate_expression("0.1 * 3")
    assert res == "0.3"


def test_calculate_expression_integer_and_power():
    assert calculate_expression("10 / 2") == "5"
    assert calculate_expression("2 ** 10") == "1024"
    assert calculate_expression("10 // 3") == "3"
    assert calculate_expression("10 % 3") == "1"
