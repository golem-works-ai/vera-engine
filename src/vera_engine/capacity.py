"""Collect remaining capacity from each provider.

OpenRouter: ``GET /credits`` at ``OPENROUTER_BASE_URL`` (default openrouter.ai).
A gateway base without that endpoint still counts the key as usable.
Anthropic: 1-token Haiku ``POST /v1/messages`` and parse
``anthropic-ratelimit-unified-{5h,7d}-*`` response headers.
xAI (Grok): launch interactive TUI in tmux, send ``/usage``, parse weekly limit.
``grok models`` is the fallback when tmux or ``/usage`` cannot produce a percent.
OpenAI (Codex): ChatGPT ``GET /backend-api/wham/usage`` for the login in
``~/.codex/auth.json``. The ``/status`` TUI is the fallback.
Subscription is probed before ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_PROBE_TIMEOUT_SECONDS = 5
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
    weekly_remaining_pct: float | None = None
    reset_credit_id: str | None = None
    reset_credit_expires_at: float | None = None

    @property
    def usage_pressure(self) -> float | None:
        """Return quota used relative to elapsed window time. Below 1 is healthy."""
        if self.window_used_pct is None:
            return None
        if self.token_used_pct is not None:
            used_pct = self.token_used_pct
        elif self.remaining_pct is not None:
            used_pct = 100.0 - self.remaining_pct
        else:
            return None
        # An expiring full reset adds another complete allowance.
        if self.reset_credit_id is not None:
            used_pct /= 2
        return _usage_pressure(used_pct, self.window_used_pct)


def openrouter_base_url() -> str:
    """Where OpenRouter is reached: ``OPENROUTER_BASE_URL`` or openrouter.ai.

    A caller that routes OpenRouter traffic through a gateway sets this to the
    gateway's OpenRouter base. The capacity probe then sends the key only to
    that gateway, never to openrouter.ai, because a gateway key is not an
    OpenRouter key.
    """
    base = os.environ.get("OPENROUTER_BASE_URL") or OPENROUTER_DEFAULT_BASE_URL
    return base.rstrip("/")


def check_openrouter() -> ProviderCapacity:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
        "PROXY_OPENROUTER_API_KEY"
    )
    if not key:
        return ProviderCapacity("openrouter", available=False, detail="no key")
    base = openrouter_base_url()
    try:
        req = urllib.request.Request(
            f"{base}/credits",
            headers={
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
                "User-Agent": "vera-engine",
            },
        )
        with _urlopen(req, timeout=_OPENROUTER_PROBE_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read()).get("data", {})
        remaining = float(data.get("total_credits", 0)) - float(
            data.get("total_usage", 0)
        )
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return ProviderCapacity(
                "openrouter",
                available=False,
                detail=f"key rejected by {base} ({exc.code})",
            )
        # A gateway that forwards completions need not serve /credits. The key
        # was accepted as far as the gateway is concerned, so the rows stay
        # usable; only the balance is unknown.
        logger.info("OpenRouter credits endpoint missing at %s: %s", base, exc)
        return ProviderCapacity(
            "openrouter",
            available=True,
            detail=f"key present; no credits endpoint at {base} ({exc.code})",
        )
    except Exception as exc:  # noqa: BLE001 - probe failure is reported or ignored, never raised
        logger.warning("OpenRouter credits check failed: %s", exc)
        return ProviderCapacity(
            "openrouter", available=True, detail=f"check failed: {exc}"
        )
    return ProviderCapacity(
        "openrouter",
        available=remaining > 0.01,
        remaining_usd=remaining,
        detail=f"${remaining:.2f} remaining",
    )


_SUBSCRIPTION_MIN_PCT = 5.0


def _prefer_subscription(cap: ProviderCapacity) -> bool:
    """True when a live subscription should beat an API key."""
    if not cap.available or cap.auth_method != "oauth":
        return False
    return not (
        cap.remaining_pct is not None and cap.remaining_pct < _SUBSCRIPTION_MIN_PCT
    )


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


# Vera runners set ANTHROPIC_BASE_URL to the internal proxy, which 401s
# setup-tokens. This probe must hit Anthropic directly.
_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
_CODEX_RESET_CREDITS_URL = (
    "https://chatgpt.com/backend-api/wham/rate-limit-reset-credits"
)
_CODEX_USAGE_TIMEOUT_SECONDS = 15
_HAIKU_PROBE_MODEL = "claude-haiku-4-5-20251001"
_ANTHROPIC_PROBE_TIMEOUT_SECONDS = 15
_CLAUDE_CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"
_PRESSURE_SLOPE = 0.9
_PRESSURE_FLOOR = 0.05
_ALLOWED_UNIFIED_STATUSES = frozenset({"allowed", "allowed_warning"})
_UNIFIED_WINDOWS = (
    ("5h", "session", timedelta(hours=5)),
    ("7d", "week", timedelta(days=7)),
)


def _usage_pressure(used_pct: float, window_used_pct: float) -> float:
    """used_frac / (elapsed_frac * 0.9 + 0.05). Lower is healthier."""
    used = used_pct / 100.0
    time_used = window_used_pct / 100.0
    return used / (time_used * _PRESSURE_SLOPE + _PRESSURE_FLOOR)


def _anthropic_oauth_token() -> str | None:
    # API keys do not return unified 5h/7d headers. The pool selector ranks
    # CLAUDE_CODE_OAUTH_TOKEN, so that env var is the credential, not a shadow.
    env = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if env:
        return env
    try:
        data = json.loads(_CLAUDE_CREDENTIALS_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    oauth = data.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        return None
    token = oauth.get("accessToken")
    if isinstance(token, str) and token.strip():
        return token.strip()
    return None


def _urlopen(req: urllib.request.Request, timeout: float) -> object:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req, timeout=timeout)


_RAW_BODY_LOG_BYTES = 2000


def _post_anthropic_messages(token: str) -> tuple[int, object, bytes]:
    body = json.dumps(
        {
            "model": _HAIKU_PROBE_MODEL,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "x"}],
        }
    ).encode()
    req = urllib.request.Request(
        _ANTHROPIC_MESSAGES_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "User-Agent": "vera-engine",
        },
    )
    # Capped at the same size _log_raw_probe_failure truncates to -- the body
    # is only ever used for that log line, so there's no reason to pull more
    # of it into memory on every probe call, success included.
    try:
        with _urlopen(req, timeout=_ANTHROPIC_PROBE_TIMEOUT_SECONDS) as resp:
            return int(resp.status), resp.headers, resp.read(_RAW_BODY_LOG_BYTES)
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.headers, exc.read(_RAW_BODY_LOG_BYTES)


def _log_raw_probe_failure(reason: str, status: int, body: bytes) -> None:
    """Log the truncated response body so a non-available verdict isn't a mystery.

    A 401/403 from a broken or revoked credential looks identical to a
    genuine "no subscription" response in ``detail`` alone. Logging the raw
    body here keeps that distinction visible without changing caller-facing
    behavior.
    """
    logger.warning(
        "anthropic usage probe non-available (%s): status=%d body=%r",
        reason,
        status,
        body[:_RAW_BODY_LOG_BYTES],
    )


def _check_claude_code_subscription() -> ProviderCapacity:
    """Probe Anthropic subscription remaining via unified rate-limit headers."""
    token = _anthropic_oauth_token()
    if not token:
        return ProviderCapacity(
            "anthropic", available=False, detail="not on subscription"
        )
    try:
        status, headers, body = _post_anthropic_messages(token)
    except TimeoutError:
        return ProviderCapacity(
            "anthropic", available=False, detail="anthropic usage check timed out"
        )
    except Exception as exc:  # noqa: BLE001 - probe failure is reported or ignored, never raised
        logger.warning("Anthropic usage check failed: %s", exc)
        return ProviderCapacity(
            "anthropic", available=False, detail=f"check failed: {exc}"
        )
    if status in (401, 403):
        _log_raw_probe_failure("unauthorized", status, body)
        return ProviderCapacity(
            "anthropic", available=False, detail="not on subscription"
        )
    if status not in (200, 429):
        _log_raw_probe_failure("unexpected status", status, body)
        return ProviderCapacity(
            "anthropic",
            available=False,
            detail=f"anthropic usage check HTTP {status}",
        )
    cap = _parse_anthropic_rate_headers(headers)
    if not cap.available:
        _log_raw_probe_failure("non-available verdict", status, body)
    logger.info(
        "anthropic header probe HTTP %s remaining_pct=%s window=%s",
        status,
        cap.remaining_pct,
        cap.window_name,
    )
    return cap


def _header_map(headers: object) -> dict[str, str]:
    if headers is None:
        return {}
    try:
        items = headers.items()
    except AttributeError:
        return {}
    return {str(key).lower(): str(value) for key, value in items}


@dataclass(frozen=True)
class _UnifiedWindow:
    window_name: str
    remaining_pct: float
    token_used_pct: float
    window_used_pct: float
    status: str
    pressure: float


def _parse_anthropic_rate_headers(
    headers: object, *, now: datetime | None = None
) -> ProviderCapacity:
    """Pick the 5h/7d window with larger used/time pressure."""
    mapped = _header_map(headers)
    current_time = now or datetime.now(UTC)
    readings: list[_UnifiedWindow] = []
    for key, window_name, duration in _UNIFIED_WINDOWS:
        prefix = f"anthropic-ratelimit-unified-{key}"
        util_raw = mapped.get(f"{prefix}-utilization")
        reset_raw = mapped.get(f"{prefix}-reset")
        if util_raw is None or reset_raw is None:
            continue
        try:
            utilization = float(util_raw)
            reset_epoch = int(float(reset_raw))
        except ValueError:
            continue
        remaining_pct = max(0.0, min(100.0, (1.0 - utilization) * 100.0))
        reset_at = datetime.fromtimestamp(reset_epoch, tz=UTC)
        window_used = _window_used_pct(reset_at, duration, current_time)
        readings.append(
            _UnifiedWindow(
                window_name=window_name,
                remaining_pct=remaining_pct,
                token_used_pct=utilization * 100.0,
                window_used_pct=window_used,
                status=mapped.get(f"{prefix}-status", ""),
                pressure=_usage_pressure(utilization * 100.0, window_used),
            )
        )

    if not readings:
        return ProviderCapacity(
            "anthropic",
            available=True,
            detail="subscription active, usage unknown",
            auth_method="oauth",
        )

    chosen = max(readings, key=lambda reading: reading.pressure)
    parts = [
        f"{reading.window_name} {reading.remaining_pct:.0f}%" for reading in readings
    ]
    status_ok = all(
        (not reading.status) or reading.status in _ALLOWED_UNIFIED_STATUSES
        for reading in readings
    )
    return ProviderCapacity(
        "anthropic",
        available=chosen.remaining_pct > 2 and status_ok,
        remaining_pct=chosen.remaining_pct,
        detail=f"{', '.join(parts)} remaining",
        auth_method="oauth",
        token_used_pct=chosen.token_used_pct,
        window_used_pct=chosen.window_used_pct,
        window_name=chosen.window_name,
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


def check_openai() -> ProviderCapacity:
    sub = _check_codex_subscription()
    if _prefer_subscription(sub):
        return sub
    if os.environ.get("OPENAI_API_KEY"):
        return _api_key_capacity("openai")
    return sub


def _codex_home(env: dict[str, str] | None = None) -> Path:
    """Resolve the Codex home directory, honoring an explicit env mapping."""
    env = os.environ if env is None else env
    raw = env.get("CODEX_HOME", "").strip()
    if raw:
        return Path(raw)
    return Path(env.get("HOME", str(Path.home()))) / ".codex"


def _codex_chatgpt_auth(home: Path | None = None) -> tuple[str, str] | None:
    """Return the ChatGPT access token and account id, when both are stored."""
    try:
        data = json.loads(
            ((home or _codex_home()) / "auth.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("auth_mode") not in (None, "chatgpt"):
        return None
    tokens = data.get("tokens")
    if not isinstance(tokens, dict):
        return None
    token = tokens.get("access_token")
    account_id = tokens.get("account_id")
    if not isinstance(token, str) or not token.strip():
        return None
    if not isinstance(account_id, str) or not account_id.strip():
        return None
    return token.strip(), account_id.strip()


def _check_codex_subscription() -> ProviderCapacity:
    """Probe the ChatGPT usage API, then the Codex TUI."""
    usage = _check_codex_usage()
    if usage is not None:
        return usage
    tui = _check_codex_status()
    if tui is not None:
        return tui
    return ProviderCapacity("openai", available=False, detail="not on subscription")


def _check_codex_usage(
    *, auth: tuple[str, str] | None = None
) -> ProviderCapacity | None:
    """Read remaining Codex quota from ``GET /backend-api/wham/usage``."""
    auth = auth or _codex_chatgpt_auth()
    if auth is None:
        return None
    token, account_id = auth
    req = urllib.request.Request(
        _CODEX_USAGE_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "ChatGPT-Account-Id": account_id,
            "Accept": "application/json",
            "User-Agent": "codex-cli",
        },
    )
    try:
        with _urlopen(req, timeout=_CODEX_USAGE_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return ProviderCapacity(
                "openai", available=False, detail="not on subscription"
            )
        logger.warning("codex usage check HTTP %s", exc.code)
        return None
    except TimeoutError:
        logger.warning("codex usage check timed out")
        return None
    except Exception as exc:  # noqa: BLE001 - probe failure is reported or ignored, never raised
        logger.warning("codex usage check failed: %s", exc)
        return None
    if not isinstance(payload, dict):
        return None
    summary = payload.get("rate_limit_reset_credits")
    if (
        isinstance(summary, dict)
        and (_as_float(summary.get("available_count")) or 0) > 0
    ):
        try:
            request = urllib.request.Request(
                _CODEX_RESET_CREDITS_URL, headers=dict(req.header_items())
            )
            with _urlopen(request, timeout=_CODEX_USAGE_TIMEOUT_SECONDS) as resp:
                payload["rate_limit_reset_credits"] = json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001 - keep the ordinary usage reading
            logger.warning("codex reset-credit check failed: %s", exc)
    return _parse_codex_usage(payload)


def _expiring_codex_reset(payload: dict, now: datetime) -> tuple[str, float] | None:
    summary = payload.get("rate_limit_reset_credits")
    credits = summary.get("credits") if isinstance(summary, dict) else None
    eligible = []
    for credit in credits if isinstance(credits, list) else []:
        if not isinstance(credit, dict) or credit.get("status") != "available":
            continue
        if (
            credit.get("reset_type") != "codex_rate_limits"
            or credit.get("is_supported_by_plan") is not True
        ):
            continue
        credit_id = credit.get("id")
        expiry = credit.get("expires_at")
        try:
            expires_at = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
            remaining = (expires_at - now).total_seconds()
        except (AttributeError, TypeError, ValueError):
            continue
        if isinstance(credit_id, str) and credit_id and 0 < remaining < 48 * 3600:
            eligible.append((credit_id, expires_at.timestamp()))
    return min(eligible, key=lambda item: item[1]) if eligible else None


def redeem_codex_reset_before_start(env: dict[str, str]) -> None:
    """Redeem one expiring full reset below 5% weekly quota, using live usage."""
    home = _codex_home(env)
    auth = _codex_chatgpt_auth(home)
    if auth is None:
        return
    # Serialize fresh-check + redemption across local session starts.
    import fcntl

    lock_path = home / "vera-engine-reset.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cap = _check_codex_usage(auth=auth)
        if (
            cap is None
            or cap.reset_credit_id is None
            or cap.weekly_remaining_pct is None
            or cap.weekly_remaining_pct >= 5
        ):
            return
        req = urllib.request.Request(
            _CODEX_RESET_CREDITS_URL + "/consume",
            data=json.dumps(
                {
                    "credit_id": cap.reset_credit_id,
                    "redeem_request_id": str(uuid.uuid4()),
                }
            ).encode(),
            headers={
                "Authorization": f"Bearer {auth[0]}",
                "ChatGPT-Account-Id": auth[1],
                "Content-Type": "application/json",
                "User-Agent": "codex-cli",
            },
        )
        try:
            with _urlopen(req, timeout=_CODEX_USAGE_TIMEOUT_SECONDS) as resp:
                result = json.loads(resp.read())
            # The consume endpoint is a private ChatGPT backend API whose
            # response body is not contractually documented. Tolerate any
            # unrecognized 2xx shape instead of failing startup; the request
            # was accepted, so treat a non-allowlisted outcome as best effort.
            if not isinstance(result, dict) or result.get("outcome") not in (
                "reset",
                "already_redeemed",
            ):
                outcome = result.get("outcome") if isinstance(result, dict) else result
                logger.warning(
                    "codex reset consume returned unrecognized outcome %r; "
                    "assuming best effort",
                    outcome,
                )
        finally:
            # A timeout can also mean the backend redeemed it; discard stale quota.
            _CAPACITY_CACHE_PATH.unlink(missing_ok=True)


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass(frozen=True)
class _CodexWindow:
    window_name: str
    remaining_pct: float
    token_used_pct: float
    window_used_pct: float | None
    pressure: float


def _codex_window_name(seconds: float | None) -> str:
    if seconds is not None and seconds >= 2 * 24 * 3600:
        return "week"
    return "session"


def _parse_codex_window(window: object, *, now: datetime) -> _CodexWindow | None:
    if not isinstance(window, dict):
        return None
    used = _as_float(window.get("used_percent"))
    if used is None:
        return None
    used = max(0.0, min(100.0, used))
    remaining = 100.0 - used
    seconds = _as_float(window.get("limit_window_seconds"))
    duration = timedelta(seconds=seconds) if seconds is not None and seconds > 0 else None
    reset_at = _as_float(window.get("reset_at"))
    window_used = None
    if duration is not None and reset_at is not None:
        window_used = _window_used_pct(
            datetime.fromtimestamp(reset_at, tz=UTC), duration, now
        )
    pressure_time = window_used if window_used is not None else 0.0
    return _CodexWindow(
        window_name=_codex_window_name(seconds),
        remaining_pct=remaining,
        token_used_pct=used,
        window_used_pct=window_used,
        pressure=_usage_pressure(used, pressure_time),
    )


def _parse_codex_usage(
    payload: object, *, now: datetime | None = None
) -> ProviderCapacity | None:
    """Extract the tighter Codex window from a ``/wham/usage`` payload.

    Returns None when the payload has no plan and no window.
    """
    if not isinstance(payload, dict):
        return None
    plan = payload.get("plan_type")
    plan_name = plan.strip() if isinstance(plan, str) and plan.strip() else "unknown"
    rate = payload.get("rate_limit")
    if not isinstance(rate, dict):
        if plan_name == "unknown":
            return None
        return ProviderCapacity(
            "openai",
            available=True,
            detail=f"codex {plan_name}: subscription active, usage unknown",
            auth_method="oauth",
        )

    current = now or datetime.now(UTC)
    readings = [
        reading
        for reading in (
            _parse_codex_window(rate.get("primary_window"), now=current),
            _parse_codex_window(rate.get("secondary_window"), now=current),
        )
        if reading is not None
    ]
    if not readings:
        return ProviderCapacity(
            "openai",
            available=True,
            detail=f"codex {plan_name}: subscription active, usage unknown",
            auth_method="oauth",
        )

    chosen = max(readings, key=lambda reading: reading.pressure)
    reset = _expiring_codex_reset(payload, current)
    weekly = next(
        (reading for reading in readings if reading.window_name == "week"), None
    )
    # A full reset restores both windows. Count one imminent reset, never stack them.
    bonus = 100.0 if reset is not None and weekly is not None else 0.0
    parts = [
        f"{reading.window_name} {reading.remaining_pct:.0f}%" for reading in readings
    ]
    if bonus:
        parts.append("+100% expiring full reset")
    allowed = rate.get("allowed", True)
    limit_reached = rate.get("limit_reached") is True
    return ProviderCapacity(
        "openai",
        available=bool(bonus)
        or (chosen.remaining_pct > 2 and allowed is not False and not limit_reached),
        remaining_pct=chosen.remaining_pct + bonus,
        detail=f"codex {plan_name}: {', '.join(parts)} remaining",
        auth_method="oauth",
        token_used_pct=chosen.token_used_pct,
        window_used_pct=chosen.window_used_pct,
        window_name=chosen.window_name,
        weekly_remaining_pct=weekly.remaining_pct if weekly else None,
        reset_credit_id=reset[0] if bonus else None,
        reset_credit_expires_at=reset[1] if bonus else None,
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
            check=False,
        )
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _tmux_send(session: str, *keys: str) -> None:
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", session, *keys],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("tmux command failed for session %s", session, exc_info=True)


def _tmux_send_literal(session: str, text: str) -> None:
    """Send literal text (bypasses tmux key lookup)."""
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", session, "-l", text],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("tmux command failed for session %s", session, exc_info=True)


def _tmux_kill(session: str) -> None:
    try:
        subprocess.run(
            ["tmux", "kill-session", "-t", session],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("tmux command failed for session %s", session, exc_info=True)


def _check_codex_status() -> ProviderCapacity | None:
    """Launch codex in tmux, send /status, parse weekly limit."""
    if not shutil.which("codex"):
        return None
    if not shutil.which("tmux"):
        logger.debug("tmux not available for codex status check")
        return None

    _tmux_kill(_TMUX_SESSION)
    try:
        # Drop OPENAI_API_KEY so /status reports the ChatGPT weekly limit.
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        r = subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                _TMUX_SESSION,
                "-x",
                "120",
                "-y",
                "40",
                "--",
                "env",
                "-u",
                "OPENAI_API_KEY",
                "codex",
            ],
            capture_output=True,
            timeout=5,
            env=env,
            check=False,
        )
        if r.returncode != 0:
            return None

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
    except Exception as exc:  # noqa: BLE001 - probe failure is reported or ignored, never raised
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
            check=False,
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
    except Exception as exc:  # noqa: BLE001 - probe failure is reported or ignored, never raised
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
            check=False,
        )
        return _parse_grok_models(f"{result.stdout}\n{result.stderr}")
    except FileNotFoundError:
        return ProviderCapacity("xai", available=False, detail="grok CLI not found")
    except subprocess.TimeoutExpired:
        return ProviderCapacity(
            "xai", available=False, detail="grok models check timed out"
        )
    except Exception as exc:  # noqa: BLE001 - probe failure is reported or ignored, never raised
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
        if payload.get("version") != 2:
            return None
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
            "version": 2,
            "expires_at": min(
                [time.time() + _CAPACITY_CACHE_TTL_SECONDS]
                + [
                    cap.reset_credit_expires_at
                    for cap in capacities.values()
                    if cap.reset_credit_expires_at is not None
                ]
            ),
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
                    "weekly_remaining_pct": cap.weekly_remaining_pct,
                    "reset_credit_id": cap.reset_credit_id,
                    "reset_credit_expires_at": cap.reset_credit_expires_at,
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
