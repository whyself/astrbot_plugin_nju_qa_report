"""Best-effort accounting for provider-reported chat-model token usage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    reported_calls: int = 0

    @property
    def unreported_calls(self) -> int:
        return max(0, self.calls - self.reported_calls)

    def since(self, earlier: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=max(0, self.input_tokens - earlier.input_tokens),
            output_tokens=max(0, self.output_tokens - earlier.output_tokens),
            total_tokens=max(0, self.total_tokens - earlier.total_tokens),
            calls=max(0, self.calls - earlier.calls),
            reported_calls=max(0, self.reported_calls - earlier.reported_calls),
        )


class TokenUsageTracker:
    """Accumulate actual usage fields exposed by AstrBot provider responses."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._usage = TokenUsage()

    def record(self, response: Any) -> None:
        extracted = _extract_usage(response)
        with self._lock:
            current = self._usage
            if extracted is None:
                self._usage = TokenUsage(
                    current.input_tokens,
                    current.output_tokens,
                    current.total_tokens,
                    current.calls + 1,
                    current.reported_calls,
                )
                return
            input_tokens, output_tokens, total_tokens = extracted
            self._usage = TokenUsage(
                current.input_tokens + input_tokens,
                current.output_tokens + output_tokens,
                current.total_tokens + total_tokens,
                current.calls + 1,
                current.reported_calls + 1,
            )

    def snapshot(self) -> TokenUsage:
        with self._lock:
            return self._usage


def _extract_usage(response: Any) -> tuple[int, int, int] | None:
    raw_completion = _value(response, "raw_completion")
    new_record = _value(response, "_new_record")
    for source in (response, raw_completion, new_record):
        usage = _value(source, "usage")
        if usage is None:
            continue
        input_tokens = _token_value(usage, "prompt_tokens", "input_tokens")
        output_tokens = _token_value(usage, "completion_tokens", "output_tokens")
        total_tokens = _token_value(usage, "total_tokens")
        if input_tokens is None and output_tokens is None and total_tokens is None:
            continue
        normalized_input = input_tokens or 0
        normalized_output = output_tokens or 0
        normalized_total = (
            total_tokens
            if total_tokens is not None
            else normalized_input + normalized_output
        )
        return normalized_input, normalized_output, normalized_total
    return None


def _token_value(source: Any, *keys: str) -> int | None:
    for key in keys:
        value = _value(source, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, float) and value >= 0 and value.is_integer():
            return int(value)
    return None


def _value(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)
