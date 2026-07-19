"""Structured merge markers carried through existing screening audit reasons."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

_FINAL_MERGE_RE = re.compile(r"^\[FINAL_MERGE:([0-9a-f]{16})\]\s*")


def with_final_merge_marker(reason: str, member_keys: Iterable[str]) -> str:
    members = tuple(sorted({str(item).strip() for item in member_keys if str(item).strip()}))
    digest = hashlib.sha256("\0".join(members).encode("utf-8")).hexdigest()[:16]
    clean_reason = _FINAL_MERGE_RE.sub("", reason.strip())
    return f"[FINAL_MERGE:{digest}] {clean_reason}".strip()


def final_merge_marker(reason: str) -> str:
    match = _FINAL_MERGE_RE.match(reason.strip())
    return match.group(1) if match else ""


def without_final_merge_marker(reason: str) -> str:
    return _FINAL_MERGE_RE.sub("", reason.strip()).strip()
