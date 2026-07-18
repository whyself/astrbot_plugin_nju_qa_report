from nju_report.privacy import redact_for_report


def test_redaction_removes_full_mention_label_before_parenthesized_id() -> None:
    rendered = redact_for_report(
        "@26 化学 希露菲(123456789) 浦口宿舍是双人间",
    )

    assert rendered == "[提及用户] 浦口宿舍是双人间"
    assert "希露菲" not in rendered
    assert "123456789" not in rendered


def test_redaction_repairs_previously_partial_mention_redaction() -> None:
    rendered = redact_for_report(
        "[提及用户] 马院 高成([编号]) 还没出结果呢",
    )

    assert rendered == "[提及用户] 还没出结果呢"
    assert "高成" not in rendered
