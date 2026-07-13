from __future__ import annotations

import ast
from pathlib import Path


def test_every_command_stops_default_llm_fallthrough() -> None:
    """Command results, especially HTML files, must not fall through to chat LLM."""

    source = (Path(__file__).parents[2] / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    commands = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and any(_is_command_decorator(item) for item in node.decorator_list)
    ]
    assert commands
    for command in commands:
        assert command.body, command.name
        first = command.body[0]
        assert isinstance(first, ast.Expr), command.name
        call = first.value
        assert isinstance(call, ast.Call), command.name
        assert isinstance(call.func, ast.Attribute), command.name
        assert call.func.attr == "stop_event", command.name


def _is_command_decorator(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "command"
    )
