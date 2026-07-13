from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from nju_report.time_windows import natural_day_window, previous_local_date


def test_shanghai_natural_day_is_a_utc_half_open_interval() -> None:
    window = natural_day_window(date(2026, 7, 12), "Asia/Shanghai")

    assert window.start_utc == datetime(2026, 7, 11, 16, 0, tzinfo=timezone.utc)
    assert window.end_utc == datetime(2026, 7, 12, 16, 0, tzinfo=timezone.utc)


def test_previous_date_uses_local_calendar_not_last_24_hours() -> None:
    now = datetime(2026, 7, 13, 3, 30, tzinfo=timezone.utc)
    assert previous_local_date(now, "Asia/Shanghai") == date(2026, 7, 12)


def test_previous_date_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="时区"):
        previous_local_date(datetime(2026, 7, 13, 3, 30), "Asia/Shanghai")


def test_dst_timezone_uses_local_midnights() -> None:
    window = natural_day_window(date(2026, 3, 8), "America/New_York")
    assert (window.end_utc - window.start_utc).total_seconds() == 23 * 3600
