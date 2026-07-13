from __future__ import annotations

from nju_report.file_echo import OutgoingFileEchoGuard


def test_guard_matches_only_recent_normalized_filename() -> None:
    guard = OutgoingFileEchoGuard(ttl_seconds=60)
    guard.remember(r"C:\exports\全部问题简表.CSV", now=100)

    assert guard.matches(["全部问题简表.csv"], now=159)
    assert not guard.matches(["其他插件.csv"], now=159)
    assert not guard.matches(["全部问题简表.csv"], now=161)


def test_cancelled_upload_does_not_suppress_future_file() -> None:
    guard = OutgoingFileEchoGuard(ttl_seconds=60)
    token = guard.remember("日报.html", now=100)

    guard.cancel(token)

    assert not guard.matches(["日报.html"], now=101)


def test_repeated_echoes_are_suppressed_for_the_full_ttl() -> None:
    guard = OutgoingFileEchoGuard(ttl_seconds=60)
    guard.remember("日报.html", now=100)

    assert guard.matches(["日报.html"], now=101)
    assert guard.matches(["日报.html"], now=150)
