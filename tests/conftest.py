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
