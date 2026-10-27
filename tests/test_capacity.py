"""Tests for provider capacity checks."""

import json
import os
from datetime import UTC, datetime, timedelta
from email.message import Message
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from vera_engine.capacity import (
    ProviderCapacity,
    _ANTHROPIC_MESSAGES_URL,
    _HAIKU_PROBE_MODEL,
    _check_claude_code_subscription,
    _check_codex_status,
    _check_codex_subscription,
    _check_codex_usage,
    _check_grok_usage,
    _codex_chatgpt_auth,
    _parse_anthropic_rate_headers,
    _parse_codex_status,
    _parse_codex_usage,
    _parse_grok_models,
    _parse_grok_usage,
    _usage_pressure,
    check_all,
    check_anthropic,
    check_openai,
    check_xai,
)

_NOW = datetime(2026, 9, 1, 20, 45, tzinfo=UTC)
# 5h window 50% elapsed: 2.5h remaining. 7d window 50% elapsed: 3.5d remaining.
_RESET_5H = int((_NOW + timedelta(hours=2.5)).timestamp())
_RESET_7D = int((_NOW + timedelta(days=3.5)).timestamp())
# 5h window 20% elapsed: 4h remaining.
_RESET_5H_EARLY = int((_NOW + timedelta(hours=4)).timestamp())


class _FakeResponse:
    def __init__(self, headers: dict[str, str], status: int = 200, body: bytes = b"{}"):
        self.headers = headers
        self.status = status
        self._body = body

    def read(self, amt: int | None = None) -> bytes:
        return self._body if amt is None else self._body[:amt]

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _unified_headers(
    *,
    util_5h: str = "0.07",
    util_7d: str = "0.02",
    status_5h: str = "allowed",
    status_7d: str = "allowed",
    reset_5h: int = _RESET_5H,
    reset_7d: int = _RESET_7D,
) -> dict[str, str]:
    return {
        "anthropic-ratelimit-unified-5h-utilization": util_5h,
        "anthropic-ratelimit-unified-5h-reset": str(reset_5h),
        "anthropic-ratelimit-unified-5h-status": status_5h,
        "anthropic-ratelimit-unified-7d-utilization": util_7d,
        "anthropic-ratelimit-unified-7d-reset": str(reset_7d),
        "anthropic-ratelimit-unified-7d-status": status_7d,
    }


def test_usage_pressure_is_used_over_elapsed_window():
    cap = ProviderCapacity(
        "xai",
        available=True,
        remaining_pct=47.0,
        token_used_pct=53.0,
        window_used_pct=40.0,
        window_name="week",
    )
    assert cap.usage_pressure == pytest.approx(0.53 / (0.40 * 0.9 + 0.05))


def test_usage_pressure_falls_back_to_remaining_when_token_used_absent():
    cap = ProviderCapacity(
        "anthropic",
        available=True,
        remaining_pct=55.0,
        window_used_pct=30.0,
    )
    assert cap.usage_pressure == pytest.approx(0.45 / (0.30 * 0.9 + 0.05))


def test_parse_anthropic_headers_picks_session_when_its_pressure_is_larger():
    cap = _parse_anthropic_rate_headers(_unified_headers(), now=_NOW)
    assert cap.available is True
    assert cap.auth_method == "oauth"
    assert cap.remaining_pct == pytest.approx(93.0)
    assert cap.token_used_pct == pytest.approx(7.0)
    assert cap.window_name == "session"
    assert cap.window_used_pct == pytest.approx(50.0, abs=0.1)
    assert cap.usage_pressure == pytest.approx(0.07 / (0.50 * 0.9 + 0.05), abs=0.01)
    assert "session 93%" in cap.detail
    assert "week 98%" in cap.detail


def test_parse_anthropic_headers_picks_week_when_its_pressure_is_larger():
    headers = _unified_headers(
        util_5h="0.10",
        util_7d="0.50",
        reset_5h=_RESET_5H_EARLY,
    )
    cap = _parse_anthropic_rate_headers(headers, now=_NOW)
    assert cap.remaining_pct == pytest.approx(50.0)
    assert cap.token_used_pct == pytest.approx(50.0)
    assert cap.window_name == "week"
    assert cap.window_used_pct == pytest.approx(50.0, abs=0.1)
    assert cap.usage_pressure == pytest.approx(0.50 / (0.50 * 0.9 + 0.05), abs=0.01)


def test_parse_anthropic_headers_reports_tighter_window_even_if_other_is_healthy():
    headers = _unified_headers(util_5h="0.99", util_7d="0.10")
    cap = _parse_anthropic_rate_headers(headers, now=_NOW)
    assert cap.available is False
    assert cap.remaining_pct == pytest.approx(1.0)
    assert cap.window_name == "session"


def test_parse_anthropic_headers_blocked_window_is_unavailable():
    headers = _unified_headers(util_5h="0.99", status_5h="blocked", util_7d="0.05")
    cap = _parse_anthropic_rate_headers(headers, now=_NOW)
    assert cap.window_name == "session"
    assert cap.available is False


def test_parse_anthropic_headers_nearly_exhausted():
    headers = _unified_headers(util_5h="0.99", util_7d="0.99")
    cap = _parse_anthropic_rate_headers(headers, now=_NOW)
    assert cap.available is False
    assert cap.remaining_pct == pytest.approx(1.0)


def test_parse_anthropic_headers_missing_windows_is_usage_unknown():
    cap = _parse_anthropic_rate_headers({}, now=_NOW)
    assert cap.available is True
    assert cap.remaining_pct is None
    assert cap.auth_method == "oauth"
    assert "usage unknown" in cap.detail


def _patch_probe(headers: dict[str, str] | None = None, *, error: BaseException | None = None):
    def fake_urlopen(req: object, timeout: object = None) -> _FakeResponse:
        if error is not None:
            raise error
        return _FakeResponse(headers or _unified_headers())

    return patch("vera_engine.capacity._urlopen", side_effect=fake_urlopen)


def test_check_anthropic_prefers_subscription_when_key_also_set():
    env = {
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test",
    }
    with patch.dict(os.environ, env, clear=True):
        with _patch_probe():
            cap = check_anthropic()
    assert cap.available is True
    assert cap.auth_method == "oauth"
    assert cap.remaining_pct == pytest.approx(93.0)


def test_check_anthropic_falls_through_to_api_key_when_subscription_exhausted():
    env = {
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test",
    }
    with patch.dict(os.environ, env, clear=True):
        with _patch_probe(_unified_headers(util_5h="0.99", util_7d="0.99")):
            cap = check_anthropic()
    assert cap.available is True
    assert cap.auth_method == "api-key"
    assert "API key" in cap.detail


def test_check_anthropic_falls_through_to_api_key_when_oauth_token_missing():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
        with patch(
            "vera_engine.capacity._CLAUDE_CREDENTIALS_PATH"
        ) as creds_path:
            creds_path.read_text.side_effect = FileNotFoundError
            cap = check_anthropic()
    assert cap.available is True
    assert cap.auth_method == "api-key"
    assert "API key" in cap.detail


def test_check_anthropic_posts_haiku_to_api_anthropic_with_oauth_token():
    captured: list[tuple[object, object]] = []

    def fake_urlopen(req: object, timeout: object = None) -> _FakeResponse:
        captured.append((req, timeout))
        return _FakeResponse(_unified_headers())

    env = {
        "CLAUDE_CODE_OAUTH_TOKEN": "  sk-ant-oat01-test  ",
        "ANTHROPIC_API_KEY": "sk-ant-api-should-not-be-used",
        "ANTHROPIC_AUTH_TOKEN": "auth-should-not-be-used",
        "ANTHROPIC_BASE_URL": "https://proxy.example.invalid",
    }
    with patch.dict(os.environ, env, clear=True):
        with patch("vera_engine.capacity._urlopen", side_effect=fake_urlopen):
            cap = check_anthropic()
    assert cap.available is True
    assert cap.auth_method == "oauth"
    req, _timeout = captured[0]
    assert req.full_url == _ANTHROPIC_MESSAGES_URL
    assert "proxy.example.invalid" not in req.full_url
    assert req.get_header("Authorization") == "Bearer sk-ant-oat01-test"
    body = json.loads(req.data.decode())
    assert body["model"] == _HAIKU_PROBE_MODEL
    assert body["max_tokens"] == 1


def test_check_anthropic_reads_credentials_file_when_env_token_absent(tmp_path):
    creds = tmp_path / ".credentials.json"
    creds.write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-file\n"}}),
        encoding="utf-8",
    )
    captured: list[object] = []

    def fake_urlopen(req: object, timeout: object = None) -> _FakeResponse:
        captured.append(req)
        return _FakeResponse(_unified_headers())

    with patch.dict(os.environ, {}, clear=True):
        with patch("vera_engine.capacity._CLAUDE_CREDENTIALS_PATH", creds):
            with patch("vera_engine.capacity._urlopen", side_effect=fake_urlopen):
                cap = check_anthropic()
    assert cap.available is True
    assert cap.auth_method == "oauth"
    assert captured[0].get_header("Authorization") == "Bearer sk-ant-oat01-file"


def test_check_anthropic_401_is_not_on_subscription():
    error = HTTPError(
        _ANTHROPIC_MESSAGES_URL,
        401,
        "Unauthorized",
        Message(),
        BytesIO(b""),
    )
    with patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-bad"}, clear=True):
        with _patch_probe(error=error):
            cap = check_anthropic()
    assert cap.available is False
    assert "not on subscription" in cap.detail


def test_check_anthropic_429_still_parses_rate_headers():
    hdrs = Message()
    for key, value in _unified_headers().items():
        hdrs[key] = value
    error = HTTPError(
        _ANTHROPIC_MESSAGES_URL,
        429,
        "Too Many Requests",
        hdrs,
        BytesIO(b""),
    )
    with patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test"}, clear=True):
        with _patch_probe(error=error):
            cap = check_anthropic()
    assert cap.available is True
    assert cap.remaining_pct == pytest.approx(93.0)
    assert cap.auth_method == "oauth"


def test_check_anthropic_no_token_no_key(tmp_path):
    missing = tmp_path / "missing.json"
    with patch.dict(os.environ, {}, clear=True):
        with patch("vera_engine.capacity._CLAUDE_CREDENTIALS_PATH", missing):
            cap = check_anthropic()
    assert cap.available is False


def test_usage_pressure_formula():
    assert _usage_pressure(9.0, 50.0) == pytest.approx(
        0.09 / (0.5 * 0.9 + 0.05), abs=0.001
    )


def test_usage_pressure_property_from_remaining():
    cap = ProviderCapacity(
        "anthropic",
        available=True,
        remaining_pct=91.0,
        token_used_pct=9.0,
        window_used_pct=50.0,
    )
    assert cap.usage_pressure == pytest.approx(_usage_pressure(9.0, 50.0), abs=0.001)


def test_usage_pressure_property_from_token_used_when_no_remaining():
    cap = ProviderCapacity(
        "anthropic",
        available=True,
        token_used_pct=9.0,
        window_used_pct=50.0,
    )
    assert cap.usage_pressure == pytest.approx(_usage_pressure(9.0, 50.0), abs=0.001)


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
    assert cap.usage_pressure == pytest.approx(_usage_pressure(1.0, 42.9), abs=0.01)


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


CODEX_USAGE_WEEKLY = {
    "plan_type": "prolite",
    "rate_limit": {
        "allowed": True,
        "limit_reached": False,
        "primary_window": {
            "used_percent": 2,
            "limit_window_seconds": 604800,
            "reset_after_seconds": 585130,
            "reset_at": 1_800_000_000,
        },
        "secondary_window": None,
    },
}

CODEX_USAGE_BOTH = {
    "plan_type": "plus",
    "rate_limit": {
        "allowed": True,
        "limit_reached": False,
        "primary_window": {
            "used_percent": 90,
            "limit_window_seconds": 18000,
            "reset_at": 1_800_000_000,
        },
        "secondary_window": {
            "used_percent": 10,
            "limit_window_seconds": 604800,
            "reset_at": 1_800_000_000,
        },
    },
}


def test_parse_codex_usage_weekly_primary():
    now = datetime.fromtimestamp(1_800_000_000, tz=UTC) - timedelta(
        seconds=int(604800 * 0.7)
    )
    cap = _parse_codex_usage(CODEX_USAGE_WEEKLY, now=now)
    assert cap is not None
    assert cap.available is True
    assert cap.remaining_pct == 98.0
    assert cap.token_used_pct == 2.0
    assert cap.window_name == "week"
    assert cap.window_used_pct == pytest.approx(30.0, abs=0.1)
    assert cap.auth_method == "oauth"
    assert "prolite" in cap.detail
    assert "week 98%" in cap.detail
    assert cap.usage_pressure == pytest.approx(_usage_pressure(2.0, 30.0), abs=0.01)


def test_parse_codex_usage_picks_tighter_window():
    now = datetime.fromtimestamp(1_800_000_000 - 60, tz=UTC)
    cap = _parse_codex_usage(CODEX_USAGE_BOTH, now=now)
    assert cap is not None
    assert cap.window_name == "session"
    assert cap.remaining_pct == 10.0
    assert "session 10%" in cap.detail
    assert "week 90%" in cap.detail


def test_parse_codex_usage_limit_reached():
    payload = json.loads(json.dumps(CODEX_USAGE_WEEKLY))
    payload["rate_limit"]["limit_reached"] = True
    cap = _parse_codex_usage(payload)
    assert cap is not None
    assert cap.available is False
    assert cap.remaining_pct == 98.0


def test_parse_codex_usage_low_remaining():
    payload = json.loads(json.dumps(CODEX_USAGE_WEEKLY))
    payload["rate_limit"]["primary_window"]["used_percent"] = 99
    cap = _parse_codex_usage(payload)
    assert cap is not None
    assert cap.available is False
    assert cap.remaining_pct == 1.0


def test_parse_codex_usage_plan_without_windows():
    cap = _parse_codex_usage({"plan_type": "plus", "rate_limit": {}})
    assert cap is not None
    assert cap.available is True
    assert cap.remaining_pct is None
    assert cap.auth_method == "oauth"
    assert "usage unknown" in cap.detail


def test_parse_codex_usage_empty():
    assert _parse_codex_usage({}) is None
    assert _parse_codex_usage("nope") is None


def test_codex_chatgpt_auth_reads_login(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": None,
                "tokens": {"access_token": "tok", "account_id": "acc"},
            }
        ),
        encoding="utf-8",
    )
    assert _codex_chatgpt_auth() == ("tok", "acc")


def test_codex_chatgpt_auth_ignores_api_key_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "apikey",
                "tokens": {"access_token": "tok", "account_id": "acc"},
            }
        ),
        encoding="utf-8",
    )
    assert _codex_chatgpt_auth() is None


def test_check_codex_usage_sends_chatgpt_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {"access_token": "tok", "account_id": "acc"},
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class _Resp:
        def read(self) -> bytes:
            return json.dumps(CODEX_USAGE_WEEKLY).encode()

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    def fake_urlopen(req: object, timeout: float) -> _Resp:
        captured["timeout"] = timeout
        captured["url"] = req.full_url  # type: ignore[attr-defined]
        captured["headers"] = {
            key.lower(): value for key, value in req.header_items()  # type: ignore[attr-defined]
        }
        return _Resp()

    with patch("vera_engine.capacity._urlopen", side_effect=fake_urlopen):
        cap = _check_codex_usage()
    assert captured["url"] == "https://chatgpt.com/backend-api/wham/usage"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["authorization"] == "Bearer tok"
    assert headers["chatgpt-account-id"] == "acc"
    assert cap is not None
    assert cap.remaining_pct == 98.0
    assert cap.auth_method == "oauth"


def test_check_codex_usage_unauthorized(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {"access_token": "tok", "account_id": "acc"},
            }
        ),
        encoding="utf-8",
    )
    error = HTTPError(
        "https://chatgpt.com/backend-api/wham/usage",
        401,
        "unauthorized",
        Message(),
        BytesIO(b""),
    )
    with patch("vera_engine.capacity._urlopen", side_effect=error):
        cap = _check_codex_usage()
    assert cap is not None
    assert cap.available is False
    assert "not on subscription" in cap.detail


def test_codex_subscription_prefers_usage_api_over_tui():
    usage = ProviderCapacity(
        "openai",
        available=True,
        remaining_pct=98.0,
        detail="codex prolite: week 98% remaining",
        auth_method="oauth",
    )
    with (
        patch("vera_engine.capacity._check_codex_usage", return_value=usage),
        patch("vera_engine.capacity._check_codex_status") as tui,
    ):
        cap = _check_codex_subscription()
    assert cap.remaining_pct == 98.0
    tui.assert_not_called()


def test_codex_subscription_falls_back_to_tui():
    tui_cap = ProviderCapacity(
        "openai",
        available=True,
        remaining_pct=99.0,
        detail="codex Plus: 99% weekly left",
        auth_method="oauth",
    )
    with (
        patch("vera_engine.capacity._check_codex_usage", return_value=None),
        patch("vera_engine.capacity._check_codex_status", return_value=tui_cap),
    ):
        cap = _check_codex_subscription()
    assert cap.remaining_pct == 99.0
    assert cap.auth_method == "oauth"


def test_check_codex_status_strips_api_key():
    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False),
        patch("vera_engine.capacity.shutil.which", return_value="/usr/bin/bin"),
        patch("vera_engine.capacity.subprocess.run") as mock_run,
    ):
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        assert _check_codex_status() is None
    new_sess = next(
        call
        for call in mock_run.call_args_list
        if call.args and call.args[0][:2] == ["tmux", "new-session"]
    )
    argv = new_sess.args[0]
    assert argv[-4:] == ["env", "-u", "OPENAI_API_KEY", "codex"]
    env = new_sess.kwargs.get("env")
    assert env is not None
    assert "OPENAI_API_KEY" not in env


def test_check_openai_prefers_subscription_when_key_also_set():
    sub = ProviderCapacity(
        "openai",
        available=True,
        remaining_pct=98.0,
        detail="codex prolite: week 98% remaining",
        auth_method="oauth",
    )
    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
        patch("vera_engine.capacity._check_codex_subscription", return_value=sub),
    ):
        cap = check_openai()
    assert cap.available is True
    assert cap.auth_method == "oauth"
    assert cap.remaining_pct == 98.0


def test_check_openai_falls_through_to_api_key_when_subscription_exhausted():
    sub = ProviderCapacity(
        "openai",
        available=False,
        remaining_pct=1.0,
        detail="codex prolite: week 1% remaining",
        auth_method="oauth",
    )
    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
        patch("vera_engine.capacity._check_codex_subscription", return_value=sub),
    ):
        cap = check_openai()
    assert cap.available is True
    assert cap.auth_method == "api-key"
    assert "API key" in cap.detail


def test_check_openai_falls_through_to_api_key_when_not_subscribed():
    sub = ProviderCapacity("openai", available=False, detail="not on subscription")
    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
        patch("vera_engine.capacity._check_codex_subscription", return_value=sub),
    ):
        cap = check_openai()
    assert cap.available is True
    assert cap.auth_method == "api-key"


def test_check_openai_without_key_uses_subscription():
    sub = ProviderCapacity(
        "openai",
        available=True,
        remaining_pct=98.0,
        detail="codex prolite: week 98% remaining",
        auth_method="oauth",
    )
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("vera_engine.capacity._check_codex_subscription", return_value=sub),
    ):
        cap = check_openai()
    assert cap.available is True
    assert cap.remaining_pct == 98.0
    assert cap.auth_method == "oauth"


def test_check_openai_without_key_codex_unavailable():
    sub = ProviderCapacity("openai", available=False, detail="not on subscription")
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("vera_engine.capacity._check_codex_subscription", return_value=sub),
    ):
        cap = check_openai()
    assert cap.available is False
    assert "not on subscription" in cap.detail


def test_subscription_probe_logs_raw_body_on_unauthorized(caplog):
    """A revoked/broken token returns 401 and must not be a silent "not on
    subscription" -- the raw body is logged so it's distinguishable from a
    genuine non-subscription response."""
    with (
        patch("vera_engine.capacity._anthropic_oauth_token", return_value="tok"),
        patch(
            "vera_engine.capacity._post_anthropic_messages",
            return_value=(
                401,
                Message(),
                b'{"error": {"message": "invalid x-api-key"}}',
            ),
        ),
        caplog.at_level("WARNING", logger="vera_engine.capacity"),
    ):
        cap = _check_claude_code_subscription()
    assert cap.available is False
    assert cap.detail == "not on subscription"
    assert any("invalid x-api-key" in record.getMessage() for record in caplog.records)


def test_subscription_probe_logs_raw_body_on_unexpected_status(caplog):
    with (
        patch("vera_engine.capacity._anthropic_oauth_token", return_value="tok"),
        patch(
            "vera_engine.capacity._post_anthropic_messages",
            return_value=(500, Message(), b"internal server error"),
        ),
        caplog.at_level("WARNING", logger="vera_engine.capacity"),
    ):
        cap = _check_claude_code_subscription()
    assert cap.available is False
    assert "HTTP 500" in cap.detail
    assert any(
        "internal server error" in record.getMessage() for record in caplog.records
    )


def test_subscription_probe_logs_raw_body_on_429_non_available(caplog):
    """A 429 whose headers resolve to a non-available verdict (e.g. a
    blocked window) must still surface the raw body -- otherwise it's the
    same silent dead end this PR fixes for 401/403 and unexpected statuses."""
    headers = _unified_headers(status_5h="blocked")
    with (
        patch("vera_engine.capacity._anthropic_oauth_token", return_value="tok"),
        patch(
            "vera_engine.capacity._post_anthropic_messages",
            return_value=(429, headers, b'{"error": {"message": "rate limited"}}'),
        ),
        caplog.at_level("WARNING", logger="vera_engine.capacity"),
    ):
        cap = _check_claude_code_subscription()
    assert cap.available is False
    assert any("rate limited" in record.getMessage() for record in caplog.records)


def test_subscription_probe_no_warning_on_success(caplog):
    headers = _unified_headers()
    with (
        patch("vera_engine.capacity._anthropic_oauth_token", return_value="tok"),
        patch(
            "vera_engine.capacity._post_anthropic_messages",
            return_value=(200, headers, b""),
        ),
        caplog.at_level("WARNING", logger="vera_engine.capacity"),
    ):
        cap = _check_claude_code_subscription()
    assert cap.available is True
    assert caplog.records == []


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
  ╰────────────────────────────────────────────────────────────────────────────────── Grok 4.6 (high) · always-approve ─╯
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
    assert cap.usage_pressure == pytest.approx(_usage_pressure(53.0, 42.9), abs=0.01)


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
    with (
        patch.dict(os.environ, {"XAI_API_KEY": "xai-test"}, clear=True),
        patch("vera_engine.capacity.shutil.which", return_value="/usr/bin/grok"),
        patch("vera_engine.capacity._check_grok_usage", return_value=usage),
    ):
        cap = check_xai()
    assert cap.available is True
    assert cap.auth_method == "oauth"
    assert cap.remaining_pct == 47.0


def test_check_xai_prefers_subscription_when_key_also_set():
    with (
        patch.dict(os.environ, {"XAI_API_KEY": "xai-test"}, clear=True),
        patch("vera_engine.capacity.shutil.which", return_value="/usr/bin/grok"),
        patch("vera_engine.capacity._check_grok_usage", return_value=None),
        patch("vera_engine.capacity.subprocess.run") as mock_run,
    ):
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

    with (
        patch.dict(os.environ, {}, clear=True),
        patch("vera_engine.capacity.shutil.which", side_effect=which),
        patch("vera_engine.capacity.subprocess.run") as mock_run,
    ):
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
    with (
        patch.dict(os.environ, {"XAI_API_KEY": "xai-test"}, clear=True),
        patch("vera_engine.capacity.shutil.which", return_value="/usr/bin/grok"),
        patch("vera_engine.capacity._check_grok_usage", return_value=None),
        patch("vera_engine.capacity.subprocess.run") as mock_run,
    ):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = GROK_MODELS_UNAUTH
        mock_run.return_value.stderr = ""
        cap = check_xai()
    assert cap.available is True
    assert cap.auth_method == "api-key"
    assert "API key" in cap.detail


def test_check_xai_api_key_requires_cli():
    with (
        patch.dict(os.environ, {"XAI_API_KEY": "xai-test"}, clear=True),
        patch("vera_engine.capacity.shutil.which", return_value=None),
    ):
        cap = check_xai()
    assert cap.available is False
    assert "not found" in cap.detail


def test_check_xai_no_key_no_cli():
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("vera_engine.capacity.shutil.which", return_value=None),
    ):
        cap = check_xai()
    assert cap.available is False
    assert "not found" in cap.detail


def test_check_grok_usage_strips_api_key_and_uses_dedicated_session():
    with (
        patch.dict(os.environ, {"XAI_API_KEY": "xai-test"}, clear=False),
        patch("vera_engine.capacity.shutil.which", return_value="/usr/bin/bin"),
        patch("vera_engine.capacity.subprocess.run") as mock_run,
    ):
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        assert _check_grok_usage() is None
    new_sess = next(
        c
        for c in mock_run.call_args_list
        if c.args and c.args[0][:2] == ["tmux", "new-session"]
    )
    argv = new_sess.args[0]
    assert "vera-engine-grok-usage" in argv
    assert "vera-engine-codex-status" not in argv
    assert argv[-4:] == ["env", "-u", "XAI_API_KEY", "grok"]
    env = new_sess.kwargs.get("env")
    assert env is not None
    assert "XAI_API_KEY" not in env


def test_check_all_includes_xai(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vera_engine.capacity._CAPACITY_CACHE_PATH", tmp_path / "capacity.json"
    )
    fake = ProviderCapacity("xai", available=False, detail="stub")
    with (
        patch("vera_engine.capacity.check_anthropic", return_value=fake),
        patch("vera_engine.capacity.check_openrouter", return_value=fake),
        patch("vera_engine.capacity.check_openai", return_value=fake),
        patch("vera_engine.capacity.check_xai", return_value=fake),
    ):
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
