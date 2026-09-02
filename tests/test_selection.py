"""Tests for the builder registry and selection."""

import pytest

from vera_engine.builders.claude_code import ClaudeCodeBuilder
from vera_engine.builders.codex import CodexBuilder
from vera_engine.builders.grok import GrokBuilder
from vera_engine.builders.opencode import OpenCodeBuilder
from vera_engine.selection import get_builder, list_engines


def test_list_engines_sorted():
    assert list_engines() == ["claude-code", "codex", "grok", "opencode"]


def test_get_builder_claude_code():
    builder = get_builder("claude-code")
    assert isinstance(builder, ClaudeCodeBuilder)
    assert builder.engine_name == "claude-code"


def test_get_builder_opencode():
    builder = get_builder("opencode")
    assert isinstance(builder, OpenCodeBuilder)
    assert builder.engine_name == "opencode"


def test_get_builder_codex():
    builder = get_builder("codex")
    assert isinstance(builder, CodexBuilder)
    assert builder.engine_name == "codex"


def test_get_builder_grok():
    builder = get_builder("grok")
    assert isinstance(builder, GrokBuilder)
    assert builder.engine_name == "grok"


def test_get_builder_unknown_engine_raises():
    with pytest.raises(ValueError, match="unknown engine 'aider'"):
        get_builder("aider")


def test_get_builder_unknown_engine_lists_available():
    with pytest.raises(
        ValueError, match=r"available:.*claude-code.*codex.*grok.*opencode"
    ):
        get_builder("nonexistent")


def test_get_builder_returns_same_instance_each_call():
    assert get_builder("codex") is get_builder("codex")
