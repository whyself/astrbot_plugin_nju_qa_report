"""Conservative question aggregation and community-answer association."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import date
from difflib import SequenceMatcher

from .answer_agent import CommunityAnswerAgent
from .models import QuestionCandidate, QuestionCluster, StoredMessage
from .storage import ReportStorage
from .time_windows import natural_day_window

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        storage: ReportStorage,
        answer_agent: CommunityAnswerAgent,
        *,
        timezone_name: str,
        concurrency: int = 3,
    ) -> None:
        self._storage = storage
        self._answer_agent = answer_agent
        self._timezone_name = timezone_name
        self._semaphore = asyncio.Semaphore(max(1, concurrency))

    async def aggregate_date(self, report_date: date) -> list[QuestionCluster]:
        candidates, _ = await asyncio.to_thread(
            self._storage.list_question_candidates,
            report_date=report_date.isoformat(),
            limit=None,
        )
        included = [item for item in candidates if item.final_decision == "INCLUDE"]
        window = natural_day_window(report_date, self._timezone_name)
        messages = await asyncio.to_thread(self._storage.messages_in_window, window)
        clusters = _aggregate(included)
        excluded_message_ids = {
            external_id
            for item in included
            if (external_id := _external_id_from_source_key(item.source_key))
        }

        async def attach(cluster: QuestionCluster) -> QuestionCluster:
            async with self._semaphore:
                try:
                    answers = await self._answer_agent.collect(
                        cluster,
                        messages,
                        excluded_message_ids=excluded_message_ids,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "NJU community-answer context agent failed for %s",
                        cluster.question_code,
                    )
                    answers = ()
                return replace(cluster, answers=answers)

        clusters = list(await asyncio.gather(*(attach(item) for item in clusters)))
        await asyncio.to_thread(
            self._storage.save_question_clusters,
            report_date.isoformat(),
            clusters,
        )
        return clusters


def _aggregate(
    candidates: list[QuestionCandidate],
    messages: list[StoredMessage] | None = None,
) -> list[QuestionCluster]:
    del messages  # Kept for compatibility; answer discovery is now Agent-driven.
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

    result: list[QuestionCluster] = []
    for builder in builders:
        ordered = sorted(
            builder.candidates, key=lambda item: (item.sent_at_utc, item.question_code)
        )
        representative = builder.representative
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


def _external_id_from_source_key(source_key: str) -> str:
    parts = source_key.split(":", 3)
    return parts[3] if len(parts) == 4 and parts[0] == "message" else ""


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]", "", value.casefold())


def _bigrams(value: str) -> set[str]:
    if len(value) < 2:
        return {value} if value else set()
    return {value[index : index + 2] for index in range(len(value) - 1)}
