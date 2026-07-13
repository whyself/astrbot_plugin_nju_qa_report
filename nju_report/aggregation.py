"""Conservative question aggregation and community-answer association."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher

from .models import CommunityAnswer, QuestionCandidate, QuestionCluster, StoredMessage
from .privacy import redact_for_report
from .storage import ReportStorage
from .time_windows import natural_day_window

_ANSWER_WINDOW_SECONDS = 20 * 60


@dataclass(slots=True)
class _ClusterBuilder:
    candidates: list[QuestionCandidate] = field(default_factory=list)

    @property
    def representative(self) -> QuestionCandidate:
        return max(
            self.candidates, key=lambda item: (len(item.canonical_question), -item.sent_at_utc)
        )


class QuestionAggregationService:
    """Group equivalent canonical questions without hiding ambiguous differences."""

    def __init__(self, storage: ReportStorage, *, timezone_name: str) -> None:
        self._storage = storage
        self._timezone_name = timezone_name

    async def aggregate_date(self, report_date: date) -> list[QuestionCluster]:
        candidates, _ = await asyncio.to_thread(
            self._storage.list_question_candidates,
            report_date=report_date.isoformat(),
            limit=None,
        )
        included = [item for item in candidates if item.final_decision == "INCLUDE"]
        window = natural_day_window(report_date, self._timezone_name)
        messages = await asyncio.to_thread(self._storage.messages_in_window, window)
        clusters = _aggregate(included, messages)
        await asyncio.to_thread(
            self._storage.save_question_clusters,
            report_date.isoformat(),
            clusters,
        )
        return clusters


def _aggregate(
    candidates: list[QuestionCandidate],
    messages: list[StoredMessage],
) -> list[QuestionCluster]:
    builders: list[_ClusterBuilder] = []
    for candidate in sorted(candidates, key=lambda item: (item.sent_at_utc, item.question_code)):
        match = next(
            (builder for builder in builders if _same_question(candidate, builder.representative)),
            None,
        )
        if match is None:
            builders.append(_ClusterBuilder([candidate]))
        else:
            match.candidates.append(candidate)

    message_by_external = {item.external_message_id: item for item in messages}
    candidate_message_ids = {
        external
        for item in candidates
        if (external := _external_id_from_source_key(item.source_key))
    }
    result: list[QuestionCluster] = []
    for builder in builders:
        ordered = sorted(
            builder.candidates, key=lambda item: (item.sent_at_utc, item.question_code)
        )
        representative = builder.representative
        question_messages = [
            message_by_external[external]
            for item in ordered
            if (external := _external_id_from_source_key(item.source_key)) in message_by_external
        ]
        answers = _answers_for_cluster(
            question_messages,
            messages,
            excluded_message_ids=candidate_message_ids,
        )
        result.append(
            QuestionCluster(
                question_code=min(item.question_code for item in ordered),
                report_date=representative.report_date,
                canonical_question=representative.canonical_question,
                category=representative.category,
                candidate_source_keys=tuple(item.source_key for item in ordered),
                representative_questions=tuple(
                    dict.fromkeys(
                        item.original_question for item in ordered if item.original_question
                    )
                )[:5],
                group_aliases=tuple(
                    sorted({item.group_alias for item in ordered if item.group_alias})
                ),
                first_sent_at_utc=min(item.sent_at_utc for item in ordered),
                last_sent_at_utc=max(item.sent_at_utc for item in ordered),
                answers=tuple(answers),
            )
        )
    return sorted(result, key=lambda item: item.question_code)


def _same_question(left: QuestionCandidate, right: QuestionCandidate) -> bool:
    if left.category and right.category and left.category != right.category:
        return False
    left_text = _normalize(left.canonical_question)
    right_text = _normalize(right.canonical_question)
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    ratio = SequenceMatcher(None, left_text, right_text).ratio()
    if ratio >= 0.88:
        return True
    left_terms = _bigrams(left_text)
    right_terms = _bigrams(right_text)
    union = left_terms | right_terms
    return bool(union) and len(left_terms & right_terms) / len(union) >= 0.72


def _answers_for_cluster(
    question_messages: list[StoredMessage],
    messages: list[StoredMessage],
    *,
    excluded_message_ids: set[str],
) -> list[CommunityAnswer]:
    if not question_messages:
        return []
    question_ids = {item.external_message_id for item in question_messages}
    group_ids = {item.group_id for item in question_messages}
    first_time = min(item.sent_at_utc for item in question_messages)
    last_time = max(item.sent_at_utc for item in question_messages)
    question_senders = {item.sender_id for item in question_messages}
    selected: dict[str, CommunityAnswer] = {}
    for message in messages:
        if message.external_message_id in question_ids | excluded_message_ids:
            continue
        if message.group_id not in group_ids or not message.text.strip():
            continue
        direct = message.reply_to_message_id in question_ids
        temporal = (
            last_time <= message.sent_at_utc <= last_time + _ANSWER_WINDOW_SECONDS
            and message.sender_id not in question_senders
            and not _looks_like_question(message.text)
        )
        if not direct and not temporal:
            continue
        if message.sent_at_utc < first_time:
            continue
        confidence = 0.98 if direct else 0.58
        selected[message.external_message_id] = CommunityAnswer(
            external_message_id=message.external_message_id,
            redacted_text=redact_for_report(message.text, max_chars=800),
            sent_at_utc=message.sent_at_utc,
            confidence=confidence,
            direct_reply=direct,
        )
    ordered = sorted(
        selected.values(),
        key=lambda item: (-int(item.direct_reply), -item.confidence, item.sent_at_utc),
    )
    return ordered[:8]


def _external_id_from_source_key(source_key: str) -> str:
    parts = source_key.split(":", 3)
    return parts[3] if len(parts) == 4 and parts[0] == "message" else ""


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]", "", value.casefold())


def _bigrams(value: str) -> set[str]:
    if len(value) < 2:
        return {value} if value else set()
    return {value[index : index + 2] for index in range(len(value) - 1)}


def _looks_like_question(value: str) -> bool:
    text = value.strip()
    return bool(
        "?" in text
        or "？" in text
        or re.search(r"(吗|么|嘛|如何|怎么|哪里|何时|多少|能不能|可不可以)[啊呀呢吗]?$", text)
    )
