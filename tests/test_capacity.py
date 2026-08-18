"""Tests for provider capacity checks."""

import os
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from vera_engine.capacity import (
    ProviderCapacity,
    _check_grok_usage,
    _parse_claude_usage,
    _parse_codex_status,
    _parse_grok_models,
    _parse_grok_usage,
    check_all,
    check_anthropic,
    check_openai,
    check_openrouter,
    check_xai,
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


def test_claude_usage_reports_weekly_token_time_and_pressure():
    cap = _parse_claude_usage(
        CLAUDE_USAGE_OUTPUT,
        now=datetime(2026, 8, 16, 6, 0, tzinfo=UTC),
    )
    assert cap.token_used_pct == 24.0
    assert cap.window_name == "week"
    assert cap.window_used_pct == pytest.approx(85.7, abs=0.1)
    assert cap.usage_pressure == pytest.approx(0.282, abs=0.001)


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


def test_codex_status_reports_token_time_and_pressure():
    cap = _parse_codex_status(
        CODEX_STATUS_OUTPUT,
        now=datetime(2026, 8, 14, 14, 31, tzinfo=UTC),
    )
    assert cap is not None
    assert cap.token_used_pct == 1.0
    assert cap.window_name == "week"
    assert cap.window_used_pct == pytest.approx(42.9, abs=0.1)
    assert cap.usage_pressure == pytest.approx(0.0182, abs=0.001)


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


def test_check_anthropic_prefers_subscription_when_key_also_set():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
        with patch("vera_engine.capacity.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = CLAUDE_USAGE_OUTPUT
            cap = check_anthropic()
    assert cap.available is True
    assert cap.auth_method == "oauth"
    assert cap.remaining_pct == 76.0
    mock_run.assert_called()


def test_check_anthropic_falls_through_to_api_key_when_subscription_exhausted():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
        with patch("vera_engine.capacity.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = CLAUDE_USAGE_HIGH
            cap = check_anthropic()
    assert cap.available is True
    assert cap.auth_method == "api-key"
    assert "API key" in cap.detail


def test_check_anthropic_falls_through_to_api_key_when_cli_missing():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
        with patch("vera_engine.capacity.subprocess.run", side_effect=FileNotFoundError):
            cap = check_anthropic()
    assert cap.available is True
    assert cap.auth_method == "api-key"
    assert "API key" in cap.detail


def test_check_anthropic_falls_back_to_claude_cli():
    with patch.dict(os.environ, {}, clear=True):
        with patch("vera_engine.capacity.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = CLAUDE_USAGE_OUTPUT
            cap = check_anthropic()
    assert cap.available is True
    assert cap.remaining_pct == 76.0
    assert cap.auth_method == "oauth"


def test_check_anthropic_no_key_no_cli():
    with patch.dict(os.environ, {}, clear=True):
        with patch("vera_engine.capacity.subprocess.run", side_effect=FileNotFoundError):
            cap = check_anthropic()
    assert cap.available is False
    assert "not found" in cap.detail


GROK_MODELS_LOGGED_IN = """\
You are logged in with grok.com.

Default model: grok-4.6

Available models:
  * grok-4.6 (default)
  - grok-4.5
"""

GROK_MODELS_UNAUTH = """\
You are not authenticated.

Default model: grok-4.6
"""


# Captured from grok TUI /usage (tmux capture-pane), SuperGrok weekly bar.
GROK_USAGE_OUTPUT = """\
                                           │  Context usage  Usage limit  Session info                                                        │
                                           │──────────────────────────────────────────────────────────────────────────────────────────────────│
                                           │  Weekly limit (SuperGrok)                                                                        │
                                           │                                                                                                  │
                                           │  ████████████████░░░░░░░░░░░░░░  53%                                                             │
                                           │  Resets: August 23, 13:51                                                                        │
                                           │                                                                                                  │
                                           │  Session usage: no model calls yet in this session.                                              │
"""

GROK_USAGE_LOW = """\
                                           │  Weekly limit (SuperGrok)                                                                        │
                                           │                                                                                                  │
                                           │  █████████████████████████████░  98%                                                             │
                                           │  Resets: August 23, 13:51                                                                        │
"""

GROK_USAGE_SPLASH = """\
                                  │  ⠀⠀⠀⠀⠀⠀⣀⣀⡀⠀⠀⠀⢀⠄   Grok Build  1.0.4                                                                                  │
                                  │  ⠀⠀⣼⡟⠁⠀⠀⠀⢀⡴⠻⣿⡀⠀   Grok 4.6 is here!                                                                                  │
  │ ❯                                                                                                                                                                                   │
  ╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── Grok 4.6 (high) · always-approve ─╯
"""

GROK_USAGE_USD_ONLY = """\
                                           │  Weekly limit (SuperGrok)                                                                        │
                                           │  Credits: $12.40 / $50.00 per month                                                              │
"""


def test_parse_grok_models_logged_in():
    cap = _parse_grok_models(GROK_MODELS_LOGGED_IN)
    assert cap.available is True
    assert cap.auth_method == "oauth"
    assert cap.remaining_pct is None
    assert "usage unknown" in cap.detail


def test_parse_grok_models_not_authenticated():
    cap = _parse_grok_models(GROK_MODELS_UNAUTH)
    assert cap.available is False
    assert "not on subscription" in cap.detail


def test_parse_grok_usage_healthy():
    cap = _parse_grok_usage(GROK_USAGE_OUTPUT)
    assert cap is not None
    assert cap.available is True
    assert cap.remaining_pct == 47.0
    assert cap.auth_method == "oauth"
    assert "SuperGrok" in cap.detail
    assert "47%" in cap.detail


def test_grok_usage_reports_token_time_and_pressure():
    cap = _parse_grok_usage(
        GROK_USAGE_OUTPUT,
        now=datetime(2026, 8, 19, 13, 51, tzinfo=UTC),
    )
    assert cap is not None
    assert cap.token_used_pct == 53.0
    assert cap.window_name == "week"
    assert cap.window_used_pct == pytest.approx(42.9, abs=0.1)
    assert cap.usage_pressure == pytest.approx(0.964, abs=0.001)


def test_parse_grok_usage_nearly_exhausted():
    cap = _parse_grok_usage(GROK_USAGE_LOW)
    assert cap is not None
    assert cap.available is False
    assert cap.remaining_pct == 2.0
    assert cap.auth_method == "oauth"


def test_parse_grok_usage_unparseable_splash():
    assert _parse_grok_usage(GROK_USAGE_SPLASH) is None


def test_parse_grok_usage_not_logged_in():
    cap = _parse_grok_usage(GROK_MODELS_UNAUTH)
    assert cap is not None
    assert cap.available is False
    assert "not on subscription" in cap.detail


def test_parse_grok_usage_usd_only():
    cap = _parse_grok_usage(GROK_USAGE_USD_ONLY)
    assert cap is not None
    assert cap.remaining_pct is None
    assert cap.remaining_usd == 37.6
    assert cap.available is True
    assert cap.auth_method == "oauth"


def test_check_xai_prefers_usage_subscription_when_key_also_set():
    usage = ProviderCapacity(
        "xai",
        available=True,
        remaining_pct=47.0,
        detail="grok SuperGrok: 47% weekly remaining",
        auth_method="oauth",
    )
    with patch.dict(os.environ, {"XAI_API_KEY": "xai-test"}, clear=True):
        with patch("vera_engine.capacity.shutil.which", return_value="/usr/bin/grok"):
            with patch("vera_engine.capacity._check_grok_usage", return_value=usage):
                cap = check_xai()
    assert cap.available is True
    assert cap.auth_method == "oauth"
    assert cap.remaining_pct == 47.0


def test_check_xai_prefers_subscription_when_key_also_set():
    with patch.dict(os.environ, {"XAI_API_KEY": "xai-test"}, clear=True):
        with patch("vera_engine.capacity.shutil.which", return_value="/usr/bin/grok"):
            with patch("vera_engine.capacity._check_grok_usage", return_value=None):
                with patch("vera_engine.capacity.subprocess.run") as mock_run:
                    mock_run.return_value.returncode = 0
                    mock_run.return_value.stdout = GROK_MODELS_LOGGED_IN
                    mock_run.return_value.stderr = ""
                    cap = check_xai()
    assert cap.available is True
    assert cap.auth_method == "oauth"
    assert cap.remaining_pct is None
    called_env = mock_run.call_args.kwargs.get("env")
    assert called_env is not None
    assert "XAI_API_KEY" not in called_env
    assert mock_run.call_args.args[0][:2] == ["grok", "models"]


def test_check_xai_tmux_missing_falls_back_to_grok_models():
    def which(name: str) -> str | None:
        return "/usr/bin/grok" if name == "grok" else None

    with patch.dict(os.environ, {}, clear=True):
        with patch("vera_engine.capacity.shutil.which", side_effect=which):
            with patch("vera_engine.capacity.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = GROK_MODELS_LOGGED_IN
                mock_run.return_value.stderr = ""
                cap = check_xai()
    assert cap.available is True
    assert cap.auth_method == "oauth"
    assert cap.remaining_pct is None
    assert "usage unknown" in cap.detail
    assert mock_run.call_args.args[0][:2] == ["grok", "models"]


def test_check_xai_falls_through_to_api_key():
    with patch.dict(os.environ, {"XAI_API_KEY": "xai-test"}, clear=True):
        with patch("vera_engine.capacity.shutil.which", return_value="/usr/bin/grok"):
            with patch("vera_engine.capacity._check_grok_usage", return_value=None):
                with patch("vera_engine.capacity.subprocess.run") as mock_run:
                    mock_run.return_value.returncode = 0
                    mock_run.return_value.stdout = GROK_MODELS_UNAUTH
                    mock_run.return_value.stderr = ""
                    cap = check_xai()
    assert cap.available is True
    assert cap.auth_method == "api-key"
    assert "API key" in cap.detail


def test_check_xai_api_key_requires_cli():
    with patch.dict(os.environ, {"XAI_API_KEY": "xai-test"}, clear=True):
        with patch("vera_engine.capacity.shutil.which", return_value=None):
            cap = check_xai()
    assert cap.available is False
    assert "not found" in cap.detail


def test_check_xai_no_key_no_cli():
    with patch.dict(os.environ, {}, clear=True):
        with patch("vera_engine.capacity.shutil.which", return_value=None):
            cap = check_xai()
    assert cap.available is False
    assert "not found" in cap.detail


def test_check_grok_usage_strips_api_key_and_uses_dedicated_session():
    with patch.dict(os.environ, {"XAI_API_KEY": "xai-test"}, clear=False):
        with patch("vera_engine.capacity.shutil.which", return_value="/usr/bin/bin"):
            with patch("vera_engine.capacity.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 1
                mock_run.return_value.stdout = ""
                mock_run.return_value.stderr = ""
                assert _check_grok_usage() is None
    new_sess = next(
        c for c in mock_run.call_args_list if c.args and c.args[0][:2] == ["tmux", "new-session"]
    )
    argv = new_sess.args[0]
    assert "vera-engine-grok-usage" in argv
    assert "vera-engine-codex-status" not in argv
    assert argv[-4:] == ["env", "-u", "XAI_API_KEY", "grok"]
    env = new_sess.kwargs.get("env")
    assert env is not None
    assert "XAI_API_KEY" not in env


def test_check_all_includes_xai(tmp_path, monkeypatch):
    monkeypatch.setattr("vera_engine.capacity._CAPACITY_CACHE_PATH", tmp_path / "capacity.json")
    fake = ProviderCapacity("xai", available=False, detail="stub")
    with patch("vera_engine.capacity.check_anthropic", return_value=fake):
        with patch("vera_engine.capacity.check_openrouter", return_value=fake):
            with patch("vera_engine.capacity.check_openai", return_value=fake):
                with patch("vera_engine.capacity.check_xai", return_value=fake):
                    caps = check_all()
    assert set(caps) == {"anthropic", "openrouter", "openai", "xai"}


def test_check_all_reuses_fresh_disk_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "capacity.json"
    monkeypatch.setattr("vera_engine.capacity._CAPACITY_CACHE_PATH", cache_path)
    fake = ProviderCapacity("xai", available=True, detail="cached")

    with (
        patch("vera_engine.capacity.check_anthropic", return_value=fake) as anthropic,
        patch("vera_engine.capacity.check_openrouter", return_value=fake),
        patch("vera_engine.capacity.check_openai", return_value=fake),
        patch("vera_engine.capacity.check_xai", return_value=fake),
    ):
        assert check_all(force=True)["anthropic"] == fake
        assert check_all()["anthropic"] == fake

    anthropic.assert_called_once()
