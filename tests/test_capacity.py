"""Tests for provider capacity checks."""

import os
from unittest.mock import patch

import pytest

from vera_engine.capacity import (
    ProviderCapacity,
    _parse_claude_usage,
    _parse_codex_status,
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


CODEX_STATUS_OUTPUT = """\
╭─────────────────────────────────────────────────────────────────────────────────╮
│  >_ OpenAI Codex (v0.144.6)                                                     │
│                                                                                 │
│ Visit https://chatgpt.com/codex/settings/usage for up-to-date                   │
│ information on rate limits and credits                                          │
│                                                                                 │
│  Model:                gpt-5.6-terra (reasoning medium, summaries auto)         │
│  Directory:            /workspace/vera-engine                                   │
│  Permissions:          Workspace (Ask for approval)                             │
│  Agents.md:            <none>                                                   │
│  Account:              user@example.com (Plus)                                  │
│  Collaboration mode:   Default                                                  │
│  Session:              019ff6b8-0c25-7b80-8ce0-2693c9355a1e                     │
│                                                                                 │
│  Weekly limit:         [████████████████████] 99% left (resets 14:31 on 18 Aug) │
╰─────────────────────────────────────────────────────────────────────────────────╯
"""

CODEX_STATUS_LOW = """\
│  Account:              user@example.com (Pro)                                   │
│  Weekly limit:         [█                   ] 2% left (resets 14:31 on 18 Aug)  │
"""

CODEX_STATUS_PARTIAL = """\
╭────────────────────────────────────────────────────╮
│ >_ OpenAI Codex (v0.144.6)                         │
│                                                    │
│ model:     gpt-5.6-terra medium   /model to change │
│ directory: /workspace/vera-engine                  │
╰────────────────────────────────────────────────────╯
"""


def test_parse_codex_status_normal():
    cap = _parse_codex_status(CODEX_STATUS_OUTPUT)
    assert cap is not None
    assert cap.available is True
    assert cap.remaining_pct == 99.0
    assert "Plus" in cap.detail
    assert cap.auth_method == "oauth"


def test_parse_codex_status_low():
    cap = _parse_codex_status(CODEX_STATUS_LOW)
    assert cap is not None
    assert cap.available is False
    assert cap.remaining_pct == 2.0
    assert "Pro" in cap.detail


def test_parse_codex_status_not_ready():
    cap = _parse_codex_status(CODEX_STATUS_PARTIAL)
    assert cap is None


def test_parse_codex_status_empty():
    assert _parse_codex_status("") is None


def test_check_openai_with_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
        cap = check_openai()
    assert cap.available is True
    assert cap.provider == "openai"


def test_check_openai_without_key_falls_back_to_codex():
    with patch.dict(os.environ, {}, clear=True):
        with patch("vera_engine.capacity._check_codex_status") as mock_cs:
            mock_cs.return_value = ProviderCapacity(
                "openai", available=True, remaining_pct=99.0,
                detail="codex Plus: 99% weekly left", auth_method="oauth",
            )
            cap = check_openai()
    assert cap.available is True
    assert cap.remaining_pct == 99.0
    assert cap.auth_method == "oauth"


def test_check_openai_without_key_codex_unavailable():
    with patch.dict(os.environ, {}, clear=True):
        with patch("vera_engine.capacity._check_codex_status", return_value=None):
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
