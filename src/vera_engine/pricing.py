"""Fetch live model pricing from OpenRouter's public API.

Uses stdlib only (urllib). Falls back to hardcoded defaults on failure.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_FETCH_TIMEOUT_SECONDS = 10
_CACHE_TTL_SECONDS = 24 * 60 * 60

_cache: dict[str, dict[str, float]] | None = None
_cache_expires_at: float = 0.0


def _per_mtok(per_token: str | float | None) -> float | None:
    if per_token is None or per_token == "":
        return None
    try:
        return float(per_token) * 1_000_000
    except (ValueError, TypeError):
        return None


def _parse_models(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Parse OpenRouter /api/v1/models into {model_id: {input, output}} per-MTok."""
    out: dict[str, dict[str, float]] = {}
    for entry in payload.get("data", []) or []:
        model_id = entry.get("id")
        pricing = entry.get("pricing") or {}
        input_cost = _per_mtok(pricing.get("prompt"))
        output_cost = _per_mtok(pricing.get("completion"))
        if model_id and input_cost is not None and output_cost is not None:
            out[model_id] = {"input": input_cost, "output": output_cost}
    return out


def fetch_openrouter_pricing() -> dict[str, dict[str, float]]:
    """Fetch live pricing from OpenRouter. Raises on network error."""
    req = urllib.request.Request(
        OPENROUTER_MODELS_URL,
        headers={"Accept": "application/json", "User-Agent": "vera-engine"},
    )
    with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_SECONDS) as resp:
        payload = json.loads(resp.read())
    return _parse_models(payload)


def get_openrouter_pricing(
    *, force: bool = False
) -> dict[str, dict[str, float]] | None:
    """Return cached pricing, refreshing if stale. None if never fetched."""
    global _cache, _cache_expires_at
    now = time.monotonic()
    if not force and _cache is not None and now < _cache_expires_at:
        return _cache
    try:
        prices = fetch_openrouter_pricing()
    except Exception as exc:
        logger.warning("OpenRouter pricing fetch failed: %s", exc)
        return _cache
    if not prices:
        return _cache
    _cache = prices
    _cache_expires_at = now + _CACHE_TTL_SECONDS
    return _cache


def openrouter_model_costs(openrouter_id: str) -> tuple[float, float] | None:
    """Return (input_cost, output_cost) in $/MTok, or None if unavailable."""
    prices = get_openrouter_pricing()
    if prices is None:
        return None
    record = prices.get(openrouter_id)
    if record is None:
        return None
    return (record["input"], record["output"])


def reset_cache() -> None:
    global _cache, _cache_expires_at
    _cache = None
    _cache_expires_at = 0.0
