"""Collect remaining capacity from each provider.

OpenRouter: REST API for credit balance.
Anthropic (Claude Code): parse ``claude -p "/usage"`` output.
xAI (Grok): parse ``grok models`` for a live grok.com login.
OpenAI (Codex): launch interactive TUI in tmux, send ``/status``, parse output.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"


@dataclass(frozen=True)
class ProviderCapacity:
    provider: str
    available: bool
    remaining_pct: float | None = None
    remaining_usd: float | None = None
    detail: str = ""
    auth_method: str = "api-key"


def check_openrouter() -> ProviderCapacity:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return ProviderCapacity("openrouter", available=False, detail="no key")
    try:
        req = urllib.request.Request(
            OPENROUTER_CREDITS_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
                "User-Agent": "vera-engine",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read()).get("data", {})
        total = float(data.get("total_credits", 0))
        used = float(data.get("total_usage", 0))
        remaining = total - used
        return ProviderCapacity(
            "openrouter",
            available=remaining > 0.01,
            remaining_usd=remaining,
            detail=f"${remaining:.2f} remaining",
        )
    except Exception as exc:
        logger.warning("OpenRouter credits check failed: %s", exc)
        return ProviderCapacity("openrouter", available=bool(key), detail=f"check failed: {exc}")


_SUBSCRIPTION_MIN_PCT = 5.0


def _prefer_subscription(cap: ProviderCapacity) -> bool:
    """True when a live subscription should beat an API key."""
    if not cap.available or cap.auth_method != "oauth":
        return False
    if cap.remaining_pct is not None and cap.remaining_pct < _SUBSCRIPTION_MIN_PCT:
        return False
    return True


def _api_key_capacity(provider: str) -> ProviderCapacity:
    return ProviderCapacity(provider, available=True, detail="API key set", auth_method="api-key")


def check_anthropic() -> ProviderCapacity:
    sub = _check_claude_code_subscription()
    if _prefer_subscription(sub):
        return sub
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _api_key_capacity("anthropic")
    return sub


def _check_claude_code_subscription() -> ProviderCapacity:
    """Parse `claude -p "/usage"` for subscription remaining capacity."""
    try:
        result = subprocess.run(
            ["claude", "-p", "/usage"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return ProviderCapacity("anthropic", available=False, detail="claude not available")
        return _parse_claude_usage(result.stdout)
    except FileNotFoundError:
        return ProviderCapacity("anthropic", available=False, detail="claude CLI not found")
    except subprocess.TimeoutExpired:
        return ProviderCapacity("anthropic", available=False, detail="claude usage check timed out")
    except Exception as exc:
        logger.warning("Claude usage check failed: %s", exc)
        return ProviderCapacity("anthropic", available=False, detail=f"check failed: {exc}")


_SESSION_PCT_RE = re.compile(r"Current session:\s*(\d+)%\s*used")
_WEEK_PCT_RE = re.compile(r"Current week \(all models\):\s*(\d+)%\s*used")


def _parse_claude_usage(output: str) -> ProviderCapacity:
    if "subscription" not in output.lower():
        return ProviderCapacity("anthropic", available=False, detail="not on subscription")

    session_match = _SESSION_PCT_RE.search(output)
    week_match = _WEEK_PCT_RE.search(output)

    session_remaining = 100 - int(session_match.group(1)) if session_match else None
    week_remaining = 100 - int(week_match.group(1)) if week_match else None

    if session_remaining is not None and week_remaining is not None:
        remaining = min(session_remaining, week_remaining)
    elif week_remaining is not None:
        remaining = week_remaining
    elif session_remaining is not None:
        remaining = session_remaining
    else:
        return ProviderCapacity("anthropic", available=True, detail="subscription active, usage unknown", auth_method="oauth")

    parts = []
    if session_remaining is not None:
        parts.append(f"session {session_remaining}%")
    if week_remaining is not None:
        parts.append(f"week {week_remaining}%")

    return ProviderCapacity(
        "anthropic",
        available=remaining > 2,
        remaining_pct=float(remaining),
        detail=f"{', '.join(parts)} remaining",
        auth_method="oauth",
    )


def check_openai() -> ProviderCapacity:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return ProviderCapacity("openai", available=True, detail="API key set")
    cap = _check_codex_status()
    if cap is not None:
        return cap
    return ProviderCapacity("openai", available=False, detail="no key and codex status unavailable")


_TMUX_SESSION = "vera-engine-codex-status"
_CODEX_POLL_INTERVAL = 0.5
_CODEX_STARTUP_TIMEOUT = 15.0
_CODEX_STATUS_TIMEOUT = 10.0
_WEEKLY_PCT_RE = re.compile(r"Weekly limit:\s*\[.*?\]\s*(\d+)%\s*left")
_ACCOUNT_RE = re.compile(r"Account:\s*\S+\s*\((\w+)\)")


def _tmux_capture(session: str) -> str:
    try:
        r = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p", "-S", "-80"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def _tmux_send(session: str, *keys: str) -> None:
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", session, *keys],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass


def _tmux_send_literal(session: str, text: str) -> None:
    """Send literal text (bypasses tmux key lookup)."""
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", session, "-l", text],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass


def _tmux_kill(session: str) -> None:
    try:
        subprocess.run(
            ["tmux", "kill-session", "-t", session],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass


def _check_codex_status() -> ProviderCapacity | None:
    """Launch codex in tmux, send /status, parse weekly limit."""
    if not shutil.which("codex"):
        return None
    if not shutil.which("tmux"):
        logger.debug("tmux not available for codex status check")
        return None

    _tmux_kill(_TMUX_SESSION)
    try:
        r = subprocess.run(
            ["tmux", "new-session", "-d", "-s", _TMUX_SESSION, "-x", "120", "-y", "40"],
            capture_output=True, timeout=5,
        )
        if r.returncode != 0:
            return None

        _tmux_send(_TMUX_SESSION, "codex", "Enter")

        # Wait for codex to finish starting (handle update prompt).
        # The update prompt is a TUI select menu — arrow-key Down to
        # "Skip", then Enter.  May need multiple attempts if the menu
        # hasn't rendered yet.
        update_dismissed = False
        deadline = time.monotonic() + _CODEX_STARTUP_TIMEOUT
        ready = False
        while time.monotonic() < deadline:
            time.sleep(_CODEX_POLL_INTERVAL)
            buf = _tmux_capture(_TMUX_SESSION)
            if not update_dismissed and "Update available" in buf and "Press enter" in buf:
                _tmux_send(_TMUX_SESSION, "Down", "Enter")
                update_dismissed = True
                continue
            if "OpenAI Codex" in buf and "/model to change" in buf:
                ready = True
                break

        if not ready:
            logger.debug("codex TUI did not reach ready state")
            return None

        time.sleep(0.5)
        _tmux_send_literal(_TMUX_SESSION, "/status")
        time.sleep(0.2)
        _tmux_send(_TMUX_SESSION, "Enter")

        # /status renders as a modal overlay. Poll for the weekly-limit
        # line; send Enter to dismiss the overlay so it lands in the
        # scrollback. Retry the dismiss a few times — the first Enter
        # may arrive before the overlay has fully rendered.
        dismiss_count = 0
        deadline = time.monotonic() + _CODEX_STATUS_TIMEOUT
        while time.monotonic() < deadline:
            time.sleep(_CODEX_POLL_INTERVAL)
            buf = _tmux_capture(_TMUX_SESSION)
            parsed = _parse_codex_status(buf)
            if parsed is not None:
                return parsed
            if "/status" in buf and dismiss_count < 3:
                _tmux_send(_TMUX_SESSION, "Enter")
                dismiss_count += 1

        logger.debug("codex /status did not render in time")
        return None
    except Exception as exc:
        logger.warning("codex status check failed: %s", exc)
        return None
    finally:
        _tmux_kill(_TMUX_SESSION)


def _parse_codex_status(output: str) -> ProviderCapacity | None:
    """Extract weekly limit from codex /status output.

    Returns None if the weekly-limit line hasn't rendered yet.
    """
    m = _WEEKLY_PCT_RE.search(output)
    if m is None:
        return None

    remaining = int(m.group(1))

    account_match = _ACCOUNT_RE.search(output)
    plan = account_match.group(1) if account_match else "unknown"

    return ProviderCapacity(
        "openai",
        available=remaining > 2,
        remaining_pct=float(remaining),
        detail=f"codex {plan}: {remaining}% weekly left",
        auth_method="oauth",
    )


def check_xai() -> ProviderCapacity:
    sub = _check_grok_subscription()
    if _prefer_subscription(sub):
        return sub
    if os.environ.get("XAI_API_KEY") and shutil.which("grok"):
        return _api_key_capacity("xai")
    return sub


def _check_grok_subscription() -> ProviderCapacity:
    """Parse ``grok models`` for a live grok.com login.

    ``/usage`` is a TUI slash command. ``grok -p /usage`` starts an agent.
    Remaining percent is not exposed headless. A live grok.com session
    matches Claude's "subscription active, usage unknown".
    """
    if not shutil.which("grok"):
        return ProviderCapacity("xai", available=False, detail="grok CLI not found")
    try:
        env = os.environ.copy()
        env.pop("XAI_API_KEY", None)
        result = subprocess.run(
            ["grok", "models"],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        return _parse_grok_models(f"{result.stdout}\n{result.stderr}")
    except FileNotFoundError:
        return ProviderCapacity("xai", available=False, detail="grok CLI not found")
    except subprocess.TimeoutExpired:
        return ProviderCapacity("xai", available=False, detail="grok models check timed out")
    except Exception as exc:
        logger.warning("Grok models check failed: %s", exc)
        return ProviderCapacity("xai", available=False, detail=f"check failed: {exc}")


def _parse_grok_models(output: str) -> ProviderCapacity:
    if "You are logged in with grok.com" in output:
        return ProviderCapacity(
            "xai",
            available=True,
            detail="subscription active, usage unknown",
            auth_method="oauth",
        )
    return ProviderCapacity("xai", available=False, detail="not on subscription")


def check_all() -> dict[str, ProviderCapacity]:
    return {
        "anthropic": check_anthropic(),
        "openrouter": check_openrouter(),
        "openai": check_openai(),
        "xai": check_xai(),
    }
