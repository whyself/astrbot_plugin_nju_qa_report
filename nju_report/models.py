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


@dataclass(frozen=True, slots=True)
class QuestionCandidate:
    """One locally retained scope-screening result."""

    question_code: str
    source_key: str
    report_date: str
    original_question: str
    canonical_question: str
    category: str
    initial_decision: str
    final_decision: str
    reason: str
    confidence: float
    status: str
    group_alias: str
    sent_at_utc: int
    created_at_utc: int
    updated_at_utc: int


@dataclass(frozen=True, slots=True)
class ProcessingWindowRecord:
    """Persistent state and counts for one idempotent daily run."""

    report_date: str
    timezone: str
    start_utc: int
    end_utc: int
    status: str
    run_id: str
    messages_scanned: int
    candidates_saved: int
    included_count: int
    dropped_count: int
    error_count: int
    error_summary: str
    created_at_utc: int
    updated_at_utc: int


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    """One approved Yuque document stored locally for report investigation."""

    namespace: str
    yuque_id: str
    title: str
    slug: str
    url: str
    updated_at: str
    body: str
    body_hash: str


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    """Stable searchable fragment of one knowledge document."""

    chunk_id: str
    namespace: str
    document_id: str
    title: str
    source_url: str
    updated_at: str
    chunk_index: int
    content: str
    content_hash: str
    embedding: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class KnowledgeSearchHit:
    """One hybrid retrieval result backed by a locally stored chunk."""

    chunk: KnowledgeChunk
    score: float
    keyword_score: float
    vector_score: float
    methods: tuple[str, ...]


class CoverageStatus(str, Enum):
    ANSWERABLE = "ANSWERABLE"
    PARTIAL = "PARTIAL"
    NO_USABLE_EVIDENCE = "NO_USABLE_EVIDENCE"
    INCOMPLETE = "INCOMPLETE"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class CommunityAnswer:
    external_message_id: str
    redacted_text: str
    sent_at_utc: int
    confidence: float
    direct_reply: bool


@dataclass(frozen=True, slots=True)
class QuestionCluster:
    question_code: str
    report_date: str
    canonical_question: str
    category: str
    candidate_source_keys: tuple[str, ...]
    representative_questions: tuple[str, ...]
    group_aliases: tuple[str, ...]
    first_sent_at_utc: int
    last_sent_at_utc: int
    answers: tuple[CommunityAnswer, ...] = ()

    @property
    def occurrence_count(self) -> int:
        return len(self.candidate_source_keys)


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    namespace: str
    document_id: str
    title: str
    source_url: str
    updated_at: str
    excerpt: str


@dataclass(frozen=True, slots=True)
class InvestigationResult:
    question_code: str
    status: CoverageStatus
    summary: str
    missing_information: str
    recommendation: str
    evidence: tuple[EvidenceItem, ...] = ()
    flags: tuple[str, ...] = ()
    queries: tuple[str, ...] = ()
    error_summary: str = ""


@dataclass(frozen=True, slots=True)
class ReportArtifact:
    """Frozen, locally rendered report version."""

    report_id: int
    report_date: str
    version: int
    status: str
    subject: str
    html_path: str
    summary_json: str
    created_at_utc: int


@dataclass(frozen=True, slots=True)
class MailDelivery:
    """Per-recipient delivery state for one frozen report."""

    report_id: int
    recipient_hash: str
    status: str
    attempts: int
    error_summary: str
    sent_at_utc: int | None
