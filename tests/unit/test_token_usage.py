from __future__ import annotations

from types import SimpleNamespace

from nju_report.token_usage import TokenUsageTracker


def test_tracker_reads_openai_usage_from_raw_completion() -> None:
    tracker = TokenUsageTracker()
    response = SimpleNamespace(
        raw_completion=SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=120,
                completion_tokens=30,
                total_tokens=150,
            )
        )
    )

    tracker.record(response)

    usage = tracker.snapshot()
    assert usage.input_tokens == 120
    assert usage.output_tokens == 30
    assert usage.total_tokens == 150
    assert usage.calls == usage.reported_calls == 1


def test_tracker_accepts_input_output_aliases_and_marks_missing_usage() -> None:
    tracker = TokenUsageTracker()
    before = tracker.snapshot()
    tracker.record({"usage": {"input_tokens": 10, "output_tokens": 4}})
    tracker.record(SimpleNamespace(completion_text="no raw usage"))

    usage = tracker.snapshot().since(before)

    assert usage.input_tokens == 10
    assert usage.output_tokens == 4
    assert usage.total_tokens == 14
    assert usage.calls == 2
    assert usage.reported_calls == 1
    assert usage.unreported_calls == 1
