from __future__ import annotations

import pytest

from nju_report.models import CoverageStatus
from nju_report.report_query import parse_list_arguments


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ("2026-07-12", ("2026-07-12", None, 1)),
        ("2026-07-12 2", ("2026-07-12", None, 2)),
        ("2026-07-12 missing", ("2026-07-12", CoverageStatus.NO_USABLE_EVIDENCE, 1)),
        ("all error 2", (None, CoverageStatus.ERROR, 2)),
        ("全部 部分覆盖", (None, CoverageStatus.PARTIAL, 1)),
    ],
)
def test_parse_list_arguments(tail: str, expected: tuple[object, ...]) -> None:
    assert parse_list_arguments(tail) == expected


@pytest.mark.parametrize("tail", ["yesterday", "all unknown", "all error 0", "all 2 3"])
def test_parse_list_arguments_rejects_invalid_values(tail: str) -> None:
    with pytest.raises(ValueError):
        parse_list_arguments(tail)
