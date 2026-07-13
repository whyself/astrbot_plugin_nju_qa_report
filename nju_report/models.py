"""Domain models shared by collection and reporting services."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CaptureOutcome(str, Enum):
    CAPTURED = "CAPTURED"
    DUPLICATE = "DUPLICATE"
    DISABLED = "DISABLED"
    PRIVATE_MESSAGE = "PRIVATE_MESSAGE"
    OUT_OF_SCOPE_GROUP = "OUT_OF_SCOPE_GROUP"
    BOT_MESSAGE = "BOT_MESSAGE"
    SYSTEM_MESSAGE = "SYSTEM_MESSAGE"
    COMMAND_MESSAGE = "COMMAND_MESSAGE"
    EMPTY_MESSAGE = "EMPTY_MESSAGE"


class ScopeDecision(str, Enum):
    INCLUDE = "INCLUDE"
    AUTO_REVIEW = "AUTO_REVIEW"
    DROP = "DROP"
    DROP_LOW_CONFIDENCE = "DROP_LOW_CONFIDENCE"
    AUTO_REVIEW_ERROR = "AUTO_REVIEW_ERROR"


class Clarity(str, Enum):
    CLEAR = "CLEAR"
    UNCERTAIN = "UNCERTAIN"


class KnowledgeValue(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True, slots=True)
class MessageEnvelope:
    """Platform-neutral message received from AstrBot."""

    platform_id: str
    bot_self_id: str
    external_message_id: str
    session_id: str
    group_id: str
    sender_id: str
    sender_name: str
    sent_at_utc: int
    text: str
    outline: str = ""
    reply_to_message_id: str = ""
    is_group_message: bool = True
    is_self_message: bool = False
    is_system_message: bool = False


@dataclass(frozen=True, slots=True)
class StoredMessage:
    """Message data persisted for later daily processing."""

    platform_id: str
    bot_self_id: str
    external_message_id: str
    message_fingerprint: str
    session_id: str
    group_id: str
    group_alias: str
    sender_id: str
    sender_name: str
    sent_at_utc: int
    text: str
    outline: str
    reply_to_message_id: str
    analyzable: bool


@dataclass(frozen=True, slots=True)
class ScopeAssessment:
    """Validated AI decision for one candidate question."""

    decision: ScopeDecision
    reason: str
    confidence: float
    canonical_question: str = ""
    category: str = ""
    clarity: Clarity = Clarity.UNCERTAIN
    knowledge_value: KnowledgeValue = KnowledgeValue.LOW
    time_sensitive: bool = False


@dataclass(frozen=True, slots=True)
class ScopeResolution:
    """Final result after optional automatic AI review."""

    assessment: ScopeAssessment
    review_rounds: int
    initial_assessment: ScopeAssessment | None = None
    review_attempts: tuple[ScopeAssessment, ...] = ()
    retryable: bool = False
    error_summary: str = ""
