"""Общие структурные контракты типов для модулей W1x."""
from typing import Any, TypedDict


class _BaseChatMessage(TypedDict):
    role: str
    content: str


class ChatMessage(_BaseChatMessage, total=False):
    thinking: str
    tool_calls: list[Any]
    tool_name: str


class UserConfig(TypedDict, total=False):
    default_model: str
    keep_alive_minutes: int
    context_tokens: int
    stats_enabled: bool


class StatsRecord(TypedDict, total=False):
    source: str
    tool_name: str
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    total_duration: int | None
    load_duration: int | None
    prompt_duration: int | None
    eval_duration: int
    speed: float
    tool_answers: int


class SessionTotals(TypedDict):
    requests: int
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    total_duration: int


class HardwareInfo(TypedDict):
    gpu_name: str
    vram_gb: float
    ram_gb: float


class MemoryFact(TypedDict):
    id: str
    fact: str
    created_at: str


class PullTaskSnapshot(TypedDict):
    status: str
    percentage: float
    speed: str
    status_msg: str
    error_msg: str
    completed_bytes: int
    total_bytes: int
