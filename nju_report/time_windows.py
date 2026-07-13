"""Natural-day reporting windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """UTC half-open interval for one local reporting day."""

    report_date: date
    timezone_name: str
    start_utc: datetime
    end_utc: datetime

    @property
    def start_timestamp(self) -> int:
        return int(self.start_utc.timestamp())

    @property
    def end_timestamp(self) -> int:
        return int(self.end_utc.timestamp())


def natural_day_window(report_date: date, timezone_name: str) -> TimeWindow:
    """Return the local natural day converted to a UTC half-open interval."""

    try:
        local_timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("timezone_name 不是有效的 IANA 时区") from exc
    local_start = datetime.combine(report_date, time.min, tzinfo=local_timezone)
    local_end = datetime.combine(report_date + timedelta(days=1), time.min, tzinfo=local_timezone)
    return TimeWindow(
        report_date=report_date,
        timezone_name=timezone_name,
        start_utc=local_start.astimezone(timezone.utc),
        end_utc=local_end.astimezone(timezone.utc),
    )


def previous_local_date(now: datetime, timezone_name: str) -> date:
    """Return the previous calendar date in the configured timezone."""

    if now.tzinfo is None:
        raise ValueError("now 必须包含时区")
    try:
        local_timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("timezone_name 不是有效的 IANA 时区") from exc
    return now.astimezone(local_timezone).date() - timedelta(days=1)
