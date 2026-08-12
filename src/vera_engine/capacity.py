"""Collect remaining capacity from each provider.

OpenRouter: REST API for credit balance.
Anthropic (Claude Code): parse `claude -p "/usage"` output.
OpenAI: key presence only (no API available).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
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


def check_anthropic() -> ProviderCapacity:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return _check_claude_code_subscription()
    return ProviderCapacity("anthropic", available=True, detail="API key set", auth_method="api-key")


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
    if not key:
        return ProviderCapacity("openai", available=False, detail="no key")
    return ProviderCapacity("openai", available=True, detail="API key set")


def check_all() -> dict[str, ProviderCapacity]:
    return {
        "anthropic": check_anthropic(),
        "openrouter": check_openrouter(),
        "openai": check_openai(),
    }
