"""Shared fixtures."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _offline_catalog_prices():
    """Pin get_catalog() to the hardcoded fallback prices.

    Live OpenRouter prices move daily and need the network. Tests that assert
    on live-price behavior patch the pricing module themselves.
    """
    with patch("vera_engine.pricing.openrouter_model_costs", return_value=None):
        yield
