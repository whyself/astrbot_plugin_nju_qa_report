from __future__ import annotations

from nju_report.help_text import (
    COVERAGE_STATUS_HELP,
    available_help_topics,
    detailed_help,
    normalize_help_topic,
)


def test_coverage_status_help_distinguishes_gap_from_execution_error() -> None:
    assert "answerable：知识库已有资料，可明确回答" in COVERAGE_STATUS_HELP
    assert "missing：知识库没有足以回答问题的可用信息" in COVERAGE_STATUS_HELP
    assert "error：程序或模型执行异常" in COVERAGE_STATUS_HELP
    assert "不能据此判断知识库是否缺失" in COVERAGE_STATUS_HELP


def test_operator_can_get_nested_command_help() -> None:
    text = detailed_help("report rerun", include_operator=True)

    assert text is not None
    assert "/nju_collect report rerun <YYYY-MM-DD|all> confirm" in text
    assert "强制重跑" in text
    assert "/nju_collect report rerun all confirm" in text
    assert "report status" in text


def test_help_accepts_full_command_and_public_aliases() -> None:
    assert normalize_help_topic("/nju_collect help repo sync") == "repo sync"
    assert normalize_help_topic("/nju_collect repo search") == "repo search"
    assert normalize_help_topic("/南哪日报 列表") == "南哪日报 列表"
    assert normalize_help_topic("列表") == "南哪日报 列表"


def test_viewer_can_only_get_public_command_help() -> None:
    assert detailed_help("南哪日报 查看", include_operator=False) is not None
    assert detailed_help("report send", include_operator=False) is None
    topics = available_help_topics(include_operator=False)
    assert "南哪日报 查看" in topics
    assert "report send" not in topics


def test_every_advertised_operator_topic_has_detail() -> None:
    topics = available_help_topics(include_operator=True)

    for topic in (
        "status",
        "help",
        "import inspect",
        "import run",
        "repo status",
        "repo sync",
        "repo search",
        "report run",
        "report rerun",
        "report status",
        "report preview",
        "report send",
        "test startup",
        "test scope",
        "investigate",
        "export questions",
    ):
        assert topic in topics
        assert detailed_help(topic, include_operator=True) is not None
