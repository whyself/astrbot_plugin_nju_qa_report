"""Short-lived fingerprints for suppressing this plugin's outgoing file echoes."""

from __future__ import annotations

import time
from pathlib import PurePath


class OutgoingFileEchoGuard:
    """Remember recently sent filenames without blocking unrelated self messages."""

    def __init__(self, *, ttl_seconds: float = 60.0) -> None:
        self._ttl_seconds = max(1.0, float(ttl_seconds))
        self._next_token = 1
        self._entries: dict[int, tuple[str, float]] = {}

    def remember(self, filename: str, *, now: float | None = None) -> int:
        current = time.monotonic() if now is None else now
        self._prune(current)
        token = self._next_token
        self._next_token += 1
        self._entries[token] = (
            _normalize_filename(filename),
            current + self._ttl_seconds,
        )
        return token

    def cancel(self, token: int) -> None:
        self._entries.pop(token, None)

    def matches(self, filenames: list[str], *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        self._prune(current)
        candidates = {_normalize_filename(item) for item in filenames}
        candidates.discard("")
        return bool(candidates) and any(
            remembered in candidates for remembered, _ in self._entries.values()
        )

    def _prune(self, now: float) -> None:
        expired = [token for token, (_, expiry) in self._entries.items() if expiry < now]
        for token in expired:
            self._entries.pop(token, None)


def _normalize_filename(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    return PurePath(text).name.casefold() if text else ""
