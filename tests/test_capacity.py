"""Tests for provider capacity checks."""

import os
from unittest.mock import patch

import pytest

from vera_engine.capacity import (
    ProviderCapacity,
    _parse_claude_usage,
    check_anthropic,
    check_openai,
    check_openrouter,
)

CLAUDE_USAGE_OUTPUT = """\
You are currently using your subscription to power your Claude Code usage

Current session: 4% used · resets Aug 12, 3:10pm (UTC)
Current week (all models): 24% used · resets Aug 17, 6am (UTC)
Current week (Fable): 20% used · resets Aug 17, 6am (UTC)
"""

CLAUDE_USAGE_HIGH = """\
You are currently using your subscription to power your Claude Code usage

Current session: 98% used · resets Aug 12, 3:10pm (UTC)
Current week (all models): 99% used · resets Aug 17, 6am (UTC)
"""


def test_parse_claude_usage_normal():
    cap = _parse_claude_usage(CLAUDE_USAGE_OUTPUT)
    assert cap.available is True
    assert cap.remaining_pct == 76.0
    assert "session 96%" in cap.detail
    assert "week 76%" in cap.detail


def test_parse_claude_usage_nearly_exhausted():
    cap = _parse_claude_usage(CLAUDE_USAGE_HIGH)
    assert cap.available is False
    assert cap.remaining_pct == 1.0


def test_parse_claude_usage_not_subscription():
    cap = _parse_claude_usage("You are using API key mode\n")
    assert cap.available is False
    assert "not on subscription" in cap.detail


def test_check_openai_with_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
        cap = check_openai()
    assert cap.available is True
    assert cap.provider == "openai"


def test_check_openai_without_key():
    with patch.dict(os.environ, {}, clear=True):
        cap = check_openai()
    assert cap.available is False


def test_check_anthropic_with_api_key():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
        cap = check_anthropic()
    assert cap.available is True
    assert "API key" in cap.detail


def test_check_anthropic_falls_back_to_claude_cli():
    with patch.dict(os.environ, {}, clear=True):
        with patch("vera_engine.capacity.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = CLAUDE_USAGE_OUTPUT
            cap = check_anthropic()
    assert cap.available is True
    assert cap.remaining_pct == 76.0


def test_check_anthropic_no_key_no_cli():
    with patch.dict(os.environ, {}, clear=True):
        with patch("vera_engine.capacity.subprocess.run", side_effect=FileNotFoundError):
            cap = check_anthropic()
    assert cap.available is False
    assert "not found" in cap.detail
