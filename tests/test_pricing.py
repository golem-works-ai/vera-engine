"""Tests for OpenRouter pricing fetcher."""

from unittest.mock import patch

import pytest

from vera_engine.pricing import (
    _parse_models,
    _per_mtok,
    openrouter_model_costs,
    reset_cache,
)


def test_per_mtok_string():
    assert _per_mtok("0.000003") == pytest.approx(3.0)


def test_per_mtok_float():
    assert _per_mtok(0.000015) == pytest.approx(15.0)


def test_per_mtok_none():
    assert _per_mtok(None) is None


def test_per_mtok_empty():
    assert _per_mtok("") is None


def test_per_mtok_garbage():
    assert _per_mtok("not-a-number") is None


def test_parse_models_extracts_pricing():
    payload = {
        "data": [
            {
                "id": "anthropic/claude-sonnet-5",
                "pricing": {"prompt": "0.000002", "completion": "0.00001"},
            },
            {
                "id": "anthropic/claude-opus-5",
                "pricing": {"prompt": "0.000005", "completion": "0.000025"},
            },
        ]
    }
    result = _parse_models(payload)
    assert "anthropic/claude-sonnet-5" in result
    assert result["anthropic/claude-sonnet-5"]["input"] == pytest.approx(2.0)
    assert result["anthropic/claude-sonnet-5"]["output"] == pytest.approx(10.0)


def test_parse_models_skips_missing_pricing():
    payload = {
        "data": [
            {"id": "some/model", "pricing": {"prompt": "0.001"}},
        ]
    }
    result = _parse_models(payload)
    assert "some/model" not in result


def test_openrouter_model_costs_with_cache():
    reset_cache()
    fake_prices = {
        "anthropic/claude-sonnet-5": {"input": 2.0, "output": 10.0},
    }
    with patch("vera_engine.pricing.get_openrouter_pricing", return_value=fake_prices):
        costs = openrouter_model_costs("anthropic/claude-sonnet-5")
    assert costs == (2.0, 10.0)


def test_openrouter_model_costs_unknown_model():
    reset_cache()
    with patch("vera_engine.pricing.get_openrouter_pricing", return_value={}):
        assert openrouter_model_costs("unknown/model") is None


def test_openrouter_model_costs_no_cache():
    reset_cache()
    with patch("vera_engine.pricing.get_openrouter_pricing", return_value=None):
        assert openrouter_model_costs("anthropic/claude-sonnet-5") is None
