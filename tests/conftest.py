"""Shared fixtures."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolated_capacity_cache(tmp_path, monkeypatch):
    """Keep tests from writing the real ~/.cache/vera-engine/capacity.json.

    A test that reaches check_all() would otherwise leave pytest-time probe
    failures in the snapshot the CLI serves for the next 30 minutes.
    """
    monkeypatch.setattr(
        "vera_engine.capacity._CAPACITY_CACHE_PATH", tmp_path / "capacity-cache.json"
    )


@pytest.fixture(autouse=True)
def _offline_catalog_prices():
    """Pin get_catalog() to the hardcoded fallback prices.

    Live OpenRouter prices move daily and need the network. Tests that assert
    on live-price behavior patch the pricing module themselves.
    """
    with patch("vera_engine.pricing.openrouter_model_costs", return_value=None):
        yield


def _no_subscription_probe(provider: str, key: str):
    """Stand-in probe: no subscription login; the API key decides availability."""
    import os

    from vera_engine.capacity import ProviderCapacity

    def probe() -> ProviderCapacity:
        if os.environ.get(key):
            return ProviderCapacity(
                provider, available=True, detail="API key set", auth_method="api-key"
            )
        return ProviderCapacity(provider, available=False, detail="not on subscription")

    return probe


@pytest.fixture(autouse=True)
def _no_live_subscription_probes(monkeypatch):
    """Keep CLI strategy resolution off the network, tmux and the real ~/.codex.

    Without this, a developer's real subscription login would change which
    strategy the CLI tests resolve. Tests that need a live subscription patch
    these names themselves.
    """
    for name, provider, key in (
        ("check_anthropic", "anthropic", "ANTHROPIC_API_KEY"),
        ("check_openai", "openai", "OPENAI_API_KEY"),
        ("check_xai", "xai", "XAI_API_KEY"),
    ):
        monkeypatch.setattr(
            f"vera_engine.cli.{name}", _no_subscription_probe(provider, key)
        )
