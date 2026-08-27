"""Collect remaining capacity from each provider.

OpenRouter: REST API for credit balance.
Anthropic (Claude Code): parse ``claude -p "/usage"`` output.
xAI (Grok): launch interactive TUI in tmux, send ``/usage``, parse weekly limit.
``grok models`` is the fallback when tmux or ``/usage`` cannot produce a percent.
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
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"
_CAPACITY_CACHE_TTL_SECONDS = 30 * 60
_CAPACITY_CACHE_PATH = Path.home() / ".cache" / "vera-engine" / "capacity.json"


@dataclass(frozen=True)
class ProviderCapacity:
    provider: str
    available: bool
    remaining_pct: float | None = None
    remaining_usd: float | None = None
    detail: str = ""
    auth_method: str = "api-key"
    token_used_pct: float | None = None
    window_used_pct: float | None = None
    window_name: str | None = None

    @property
    def usage_pressure(self) -> float | None:
        """Return quota-use pressure relative to elapsed subscription time."""
        if self.token_used_pct is None or self.window_used_pct is None:
            return None
        token_used = self.token_used_pct / 100
        time_used = self.window_used_pct / 100
        return token_used / (time_used * 0.7 + 0.25)


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
        return ProviderCapacity(
            "openrouter", available=bool(key), detail=f"check failed: {exc}"
        )


_SUBSCRIPTION_MIN_PCT = 5.0


def _prefer_subscription(cap: ProviderCapacity) -> bool:
    """True when a live subscription should beat an API key."""
    if not cap.available or cap.auth_method != "oauth":
        return False
    if cap.remaining_pct is not None and cap.remaining_pct < _SUBSCRIPTION_MIN_PCT:
        return False
    return True


def _api_key_capacity(provider: str) -> ProviderCapacity:
    return ProviderCapacity(
        provider, available=True, detail="API key set", auth_method="api-key"
    )


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
            logger.info(
                "claude -p /usage exited %d; stderr=%r",
                result.returncode,
                (result.stderr or "")[:200],
            )
            return ProviderCapacity(
                "anthropic", available=False, detail="claude not available"
            )
        logger.info("claude -p /usage stdout=%r", result.stdout[:500])
        return _parse_claude_usage(result.stdout)
    except FileNotFoundError:
        return ProviderCapacity(
            "anthropic", available=False, detail="claude CLI not found"
        )
    except subprocess.TimeoutExpired:
        return ProviderCapacity(
            "anthropic", available=False, detail="claude usage check timed out"
        )
    except Exception as exc:
        logger.warning("Claude usage check failed: %s", exc)
        return ProviderCapacity(
            "anthropic", available=False, detail=f"check failed: {exc}"
        )


_SESSION_PCT_RE = re.compile(r"Current session:\s*(\d+)%\s*used")
_WEEK_PCT_RE = re.compile(r"Current week \(all models\):\s*(\d+)%\s*used")
_CLAUDE_SESSION_RESET_RE = re.compile(
    r"Current session:.*?resets\s+([A-Z][a-z]{2}\s+\d{1,2},\s+\d{1,2}(?::\d{2})?[ap]m)\s+\(UTC\)"
)
_CLAUDE_WEEK_RESET_RE = re.compile(
    r"Current week \(all models\):.*?resets\s+([A-Z][a-z]{2}\s+\d{1,2},\s+\d{1,2}(?::\d{2})?[ap]m)\s+\(UTC\)"
)


def _window_used_pct(reset_at: datetime, duration: timedelta, now: datetime) -> float:
    """Return elapsed-window percentage, bounded to the active window."""
    start_at = reset_at - duration
    elapsed = (now - start_at).total_seconds() / duration.total_seconds()
    return max(0.0, min(100.0, elapsed * 100))


def _parse_reset(
    value: str, fmt: str, *, now: datetime, duration: timedelta
) -> float | None:
    """Parse a provider reset timestamp and return its elapsed-window percentage."""
    value = re.sub(r"(?<!:)\b(\d{1,2})([ap]m)\b", r"\1:00\2", value)
    try:
        reset_at = datetime.strptime(value, fmt).replace(year=now.year, tzinfo=UTC)
    except ValueError:
        return None
    if reset_at < now - duration:
        reset_at = reset_at.replace(year=reset_at.year + 1)
    return _window_used_pct(reset_at, duration, now)


def _parse_claude_usage(
    output: str, *, now: datetime | None = None
) -> ProviderCapacity:
    if "subscription" not in output.lower():
        return ProviderCapacity(
            "anthropic", available=False, detail="not on subscription"
        )

    session_match = _SESSION_PCT_RE.search(output)
    week_match = _WEEK_PCT_RE.search(output)

    current_time = now or datetime.now(UTC)
    session_used = float(session_match.group(1)) if session_match else None
    week_used = float(week_match.group(1)) if week_match else None
    session_remaining = 100 - session_used if session_used is not None else None
    week_remaining = 100 - week_used if week_used is not None else None

    if session_remaining is not None and week_remaining is not None:
        remaining = min(session_remaining, week_remaining)
    elif week_remaining is not None:
        remaining = week_remaining
    elif session_remaining is not None:
        remaining = session_remaining
    else:
        return ProviderCapacity(
            "anthropic",
            available=True,
            detail="subscription active, usage unknown",
            auth_method="oauth",
        )

    parts = []
    if session_remaining is not None:
        parts.append(f"session {session_remaining:.0f}%")
    if week_remaining is not None:
        parts.append(f"week {week_remaining:.0f}%")

    limiting_window = max(
        (
            (session_used, "session", _CLAUDE_SESSION_RESET_RE, timedelta(hours=5)),
            (week_used, "week", _CLAUDE_WEEK_RESET_RE, timedelta(days=7)),
        ),
        key=lambda value: value[0] if value[0] is not None else -1,
    )
    token_used, window_name, reset_re, duration = limiting_window
    reset_match = reset_re.search(output)
    window_used = (
        _parse_reset(
            reset_match.group(1), "%b %d, %I:%M%p", now=current_time, duration=duration
        )
        if reset_match is not None
        else None
    )

    return ProviderCapacity(
        "anthropic",
        available=remaining > 2,
        remaining_pct=float(remaining),
        detail=f"{', '.join(parts)} remaining",
        auth_method="oauth",
        token_used_pct=token_used,
        window_used_pct=window_used,
        window_name=window_name,
    )


def check_openai() -> ProviderCapacity:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return ProviderCapacity("openai", available=True, detail="API key set")
    cap = _check_codex_status()
    if cap is not None:
        return cap
    return ProviderCapacity(
        "openai", available=False, detail="no key and codex status unavailable"
    )


_TMUX_SESSION = "vera-engine-codex-status"
_CODEX_POLL_INTERVAL = 0.5
_CODEX_STARTUP_TIMEOUT = 15.0
_CODEX_STATUS_TIMEOUT = 10.0
_WEEKLY_PCT_RE = re.compile(r"Weekly limit:\s*\[.*?\]\s*(\d+)%\s*left")
_CODEX_RESET_RE = re.compile(
    r"Weekly limit:.*?\(resets\s+(\d{1,2}:\d{2})\s+on\s+(\d{1,2}\s+[A-Z][a-z]{2})\)"
)
_ACCOUNT_RE = re.compile(r"Account:\s*\S+\s*\((\w+)\)")


def _tmux_capture(session: str) -> str:
    try:
        r = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p", "-S", "-80"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def _tmux_send(session: str, *keys: str) -> None:
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", session, *keys],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass


def _tmux_send_literal(session: str, text: str) -> None:
    """Send literal text (bypasses tmux key lookup)."""
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", session, "-l", text],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass


def _tmux_kill(session: str) -> None:
    try:
        subprocess.run(
            ["tmux", "kill-session", "-t", session],
            capture_output=True,
            timeout=5,
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
            capture_output=True,
            timeout=5,
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
            if (
                not update_dismissed
                and "Update available" in buf
                and "Press enter" in buf
            ):
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


def _parse_codex_status(
    output: str, *, now: datetime | None = None
) -> ProviderCapacity | None:
    """Extract weekly limit from codex /status output.

    Returns None if the weekly-limit line hasn't rendered yet.
    """
    m = _WEEKLY_PCT_RE.search(output)
    if m is None:
        return None

    remaining = int(m.group(1))
    reset_match = _CODEX_RESET_RE.search(output)
    current_time = now or datetime.now(UTC)
    window_used = (
        _parse_reset(
            f"{reset_match.group(2)} {reset_match.group(1)}",
            "%d %b %H:%M",
            now=current_time,
            duration=timedelta(days=7),
        )
        if reset_match is not None
        else None
    )

    account_match = _ACCOUNT_RE.search(output)
    plan = account_match.group(1) if account_match else "unknown"

    return ProviderCapacity(
        "openai",
        available=remaining > 2,
        remaining_pct=float(remaining),
        detail=f"codex {plan}: {remaining}% weekly left",
        auth_method="oauth",
        token_used_pct=float(100 - remaining),
        window_used_pct=window_used,
        window_name="week",
    )


def check_xai() -> ProviderCapacity:
    sub = _check_grok_subscription()
    if _prefer_subscription(sub):
        return sub
    if os.environ.get("XAI_API_KEY") and shutil.which("grok"):
        return _api_key_capacity("xai")
    return sub


_GROK_TMUX_SESSION = "vera-engine-grok-usage"
_GROK_POLL_INTERVAL = 0.5
_GROK_STARTUP_TIMEOUT = 15.0
_GROK_USAGE_TIMEOUT = 10.0
# TUI bar is creditUsagePercent (used), not remaining. Same conversion as Claude.
_GROK_WEEKLY_BAR_RE = re.compile(
    r"Weekly limit\s*\(([^)]+)\)(?:[^\n]*\n){1,3}[^\n]*?(\d+)\s*%",
)
_GROK_USD_RE = re.compile(
    r"\$(\d+(?:\.\d+)?)\s*/\s*\$(\d+(?:\.\d+)?)",
)
_GROK_RESET_RE = re.compile(r"Resets:\s*([A-Z][a-z]+\s+\d{1,2},\s+\d{1,2}:\d{2})")


def _check_grok_subscription() -> ProviderCapacity:
    """Probe grok.com remaining capacity.

    Primary: tmux TUI ``/usage``. ``grok -p /usage`` starts an agent.
    Fallback: ``grok models`` login line, remaining percent unknown.
    """
    usage = _check_grok_usage()
    if usage is not None:
        return usage
    return _check_grok_models()


def _check_grok_usage() -> ProviderCapacity | None:
    """Launch grok in tmux, send /usage, parse the weekly-limit bar."""
    if not shutil.which("grok"):
        return None
    if not shutil.which("tmux"):
        logger.debug("tmux not available for grok usage check")
        return None

    _tmux_kill(_GROK_TMUX_SESSION)
    try:
        env = os.environ.copy()
        env.pop("XAI_API_KEY", None)
        r = subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                _GROK_TMUX_SESSION,
                "-x",
                "120",
                "-y",
                "40",
                "--",
                "env",
                "-u",
                "XAI_API_KEY",
                "grok",
            ],
            capture_output=True,
            timeout=5,
            env=env,
        )
        if r.returncode != 0:
            return None

        deadline = time.monotonic() + _GROK_STARTUP_TIMEOUT
        ready = False
        while time.monotonic() < deadline:
            time.sleep(_GROK_POLL_INTERVAL)
            buf = _tmux_capture(_GROK_TMUX_SESSION)
            if "Do you trust" in buf:
                _tmux_send(_GROK_TMUX_SESSION, "Enter")
                continue
            if "Grok Build" in buf or "❯" in buf or "always-approve" in buf:
                ready = True
                break

        if not ready:
            logger.debug("grok TUI did not reach ready state")
            return None

        time.sleep(0.5)
        _tmux_send_literal(_GROK_TMUX_SESSION, "/usage")
        time.sleep(0.2)
        _tmux_send(_GROK_TMUX_SESSION, "Enter")

        dismiss_count = 0
        deadline = time.monotonic() + _GROK_USAGE_TIMEOUT
        while time.monotonic() < deadline:
            time.sleep(_GROK_POLL_INTERVAL)
            buf = _tmux_capture(_GROK_TMUX_SESSION)
            parsed = _parse_grok_usage(buf)
            if parsed is not None:
                return parsed
            if "/usage" in buf and dismiss_count < 3:
                _tmux_send(_GROK_TMUX_SESSION, "Enter")
                dismiss_count += 1

        logger.debug("grok /usage did not render in time")
        return None
    except Exception as exc:
        logger.warning("grok usage check failed: %s", exc)
        return None
    finally:
        _tmux_kill(_GROK_TMUX_SESSION)


def _parse_grok_usage(
    output: str, *, now: datetime | None = None
) -> ProviderCapacity | None:
    """Extract weekly remaining from grok TUI ``/usage``.

    Returns None if the weekly-limit line has not rendered yet.
    """
    if "You are not authenticated" in output:
        return ProviderCapacity("xai", available=False, detail="not on subscription")

    bar = _GROK_WEEKLY_BAR_RE.search(output)
    usd = _GROK_USD_RE.search(output)
    if bar is None and usd is None:
        return None

    plan = bar.group(1).strip() if bar else "unknown"
    remaining_pct = float(100 - int(bar.group(2))) if bar is not None else None
    remaining_usd = (
        float(usd.group(2)) - float(usd.group(1)) if usd is not None else None
    )
    reset_match = _GROK_RESET_RE.search(output)
    current_time = now or datetime.now(UTC)
    window_used = (
        _parse_reset(
            reset_match.group(1),
            "%B %d, %H:%M",
            now=current_time,
            duration=timedelta(days=7),
        )
        if reset_match is not None
        else None
    )

    if remaining_pct is not None:
        available = remaining_pct > 2
        detail = f"grok {plan}: {remaining_pct:.0f}% weekly remaining"
    elif remaining_usd is not None:
        available = remaining_usd > 0
        detail = f"grok {plan}: ${remaining_usd:.2f} remaining"
    else:
        return None

    return ProviderCapacity(
        "xai",
        available=available,
        remaining_pct=remaining_pct,
        remaining_usd=remaining_usd,
        detail=detail,
        auth_method="oauth",
        token_used_pct=float(100 - remaining_pct)
        if remaining_pct is not None
        else None,
        window_used_pct=window_used,
        window_name="week" if remaining_pct is not None else None,
    )


def _check_grok_models() -> ProviderCapacity:
    """Parse ``grok models`` for a live grok.com login."""
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
        return ProviderCapacity(
            "xai", available=False, detail="grok models check timed out"
        )
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


def _load_cached_capacities() -> dict[str, ProviderCapacity] | None:
    """Load a fresh cross-process capacity snapshot, if one exists."""
    try:
        payload = json.loads(_CAPACITY_CACHE_PATH.read_text(encoding="utf-8"))
        if time.time() >= payload["expires_at"]:
            return None
        capacities = {
            provider: ProviderCapacity(**value)
            for provider, value in payload["capacities"].items()
        }
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return capacities


def _save_cached_capacities(capacities: dict[str, ProviderCapacity]) -> None:
    """Persist a short-lived capacity snapshot without storing credentials."""
    try:
        _CAPACITY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "expires_at": time.time() + _CAPACITY_CACHE_TTL_SECONDS,
            "capacities": {
                provider: {
                    "provider": cap.provider,
                    "available": cap.available,
                    "remaining_pct": cap.remaining_pct,
                    "remaining_usd": cap.remaining_usd,
                    "detail": cap.detail,
                    "auth_method": cap.auth_method,
                    "token_used_pct": cap.token_used_pct,
                    "window_used_pct": cap.window_used_pct,
                    "window_name": cap.window_name,
                }
                for provider, cap in capacities.items()
            },
        }
        _CAPACITY_CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        logger.debug("capacity cache write failed: %s", exc)


def check_all(*, force: bool = False) -> dict[str, ProviderCapacity]:
    """Return provider capacities, reusing a 30-minute local snapshot by default."""
    if not force:
        cached = _load_cached_capacities()
        if cached is not None:
            return cached

    capacities = {
        "anthropic": check_anthropic(),
        "openrouter": check_openrouter(),
        "openai": check_openai(),
        "xai": check_xai(),
    }
    _save_cached_capacities(capacities)
    return capacities
