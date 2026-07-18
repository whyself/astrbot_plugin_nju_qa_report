"""Minimal data minimization before chat text is sent to an external LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass

_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{7,16}(?!\d)")
_URL_RE = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
_QQ_LABEL_RE = re.compile(r"(?i)(QQ|企鹅号)(\s*[：:]?\s*)\d{5,12}")
_AT_WITH_ID_RE = re.compile(r"@[^\r\n()]{1,64}\(\d{5,12}\)")
_REDACTED_AT_WITH_LABEL_RE = re.compile(
    r"\[提及用户\][^\r\n()]{0,64}\(\[编号\]\)"
)
_AT_RE = re.compile(r"@[\w\-\u4e00-\u9fff]{1,32}")
_NAME_RE = re.compile(r"(我叫|姓名\s*[是为：:]?\s*)[\u4e00-\u9fff]{2,4}")
_ROOM_RE = re.compile(
    r"(宿舍|寝室|房间)(\s*[号：:]?\s*)[A-Za-z0-9\-]{2,16}",
    re.IGNORECASE,
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True, slots=True)
class PreparedScopeInput:
    message: str
    context: str


def prepare_scope_input(
    message: str,
    context: str,
    *,
    max_message_chars: int = 2000,
    max_context_chars: int = 8000,
) -> PreparedScopeInput:
    """Redact common identifiers and bound input before provider transmission."""

    return PreparedScopeInput(
        message=_truncate(_redact(str(message)), max_message_chars),
        context=_truncate(_redact(str(context)), max_context_chars),
    )


def redact_for_report(text: str, *, max_chars: int = 2000) -> str:
    """Return a bounded, identifier-redacted copy safe for reports and exports."""

    return _truncate(_redact(str(text)), max_chars)


def _redact(text: str) -> str:
    redacted = _CONTROL_RE.sub("", text)
    redacted = _URL_RE.sub("[链接]", redacted)
    redacted = _EMAIL_RE.sub("[邮箱]", redacted)
    redacted = _ID_CARD_RE.sub("[身份证号]", redacted)
    redacted = _PHONE_RE.sub("[手机号]", redacted)
    redacted = _QQ_LABEL_RE.sub(r"\1\2[账号]", redacted)
    redacted = _AT_WITH_ID_RE.sub("[提及用户]", redacted)
    redacted = _LONG_NUMBER_RE.sub("[编号]", redacted)
    redacted = _REDACTED_AT_WITH_LABEL_RE.sub("[提及用户]", redacted)
    redacted = _AT_RE.sub("[提及用户]", redacted)
    redacted = _NAME_RE.sub(r"\1[姓名]", redacted)
    return _ROOM_RE.sub(r"\1\2[房间]", redacted)


def _truncate(text: str, limit: int) -> str:
    if limit < 1:
        raise ValueError("文本长度限制必须大于 0")
    if len(text) <= limit:
        return text
    marker = "\n[内容已截断]"
    return text[: max(0, limit - len(marker))] + marker
