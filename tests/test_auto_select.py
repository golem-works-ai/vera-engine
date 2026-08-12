"""Tests for cost-based auto-selection with capacity awareness."""

from unittest.mock import patch

import pytest

from vera_engine.auto_select import select_model, _usable_providers, _LOW_PCT_THRESHOLD, _LOW_USD_THRESHOLD
from vera_engine.capacity import ProviderCapacity
from vera_engine.config import EngineConfig


def _cap(provider, available=True, pct=None, usd=None, detail="", auth="api-key"):
    return ProviderCapacity(provider, available, pct, usd, detail, auth)


def _all_healthy():
    return {
        "anthropic": _cap("anthropic", pct=80.0, auth="oauth"),
        "openrouter": _cap("openrouter", usd=500.0),
        "openai": _cap("openai"),
    }


def _only_anthropic_oauth():
    return {
        "anthropic": _cap("anthropic", pct=80.0, auth="oauth"),
        "openrouter": _cap("openrouter", available=False),
        "openai": _cap("openai", available=False),
    }


def _only_anthropic_api_key():
    return {
        "anthropic": _cap("anthropic", auth="api-key"),
        "openrouter": _cap("openrouter", available=False),
        "openai": _cap("openai", available=False),
    }


def _only_openrouter():
    return {
        "anthropic": _cap("anthropic", available=False),
        "openrouter": _cap("openrouter", usd=500.0),
        "openai": _cap("openai", available=False),
    }


def _none_available():
    return {
        "anthropic": _cap("anthropic", available=False),
        "openrouter": _cap("openrouter", available=False),
        "openai": _cap("openai", available=False),
    }


def _openrouter_low():
    return {
        "anthropic": _cap("anthropic", pct=80.0, auth="oauth"),
        "openrouter": _cap("openrouter", usd=0.50),
        "openai": _cap("openai"),
    }


def test_selects_cheapest_with_all_healthy():
    config = EngineConfig(provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0})
    with patch("vera_engine.auto_select.check_all", return_value=_all_healthy()):
        sel = select_model(config)
    from vera_engine.models import get_catalog
    cheapest = min(s.blended_cost(config.input_weight) for s in get_catalog())
    assert sel.model.blended_cost(config.input_weight) == pytest.approx(cheapest)


def test_subscription_ratio_changes_winner():
    config = EngineConfig(provider_ratios={"anthropic": 0.01, "openrouter": 1.0, "openai": 1.0})
    with patch("vera_engine.auto_select.check_all", return_value=_all_healthy()):
        sel = select_model(config)
    assert sel.model.provider == "anthropic"


def test_filters_by_engine():
    config = EngineConfig(provider_ratios={})
    with patch("vera_engine.auto_select.check_all", return_value=_all_healthy()):
        sel = select_model(config, engine="opencode")
    assert sel.model.engine == "opencode"


def test_filters_by_provider():
    config = EngineConfig(provider_ratios={})
    with patch("vera_engine.auto_select.check_all", return_value=_all_healthy()):
        sel = select_model(config, provider="openrouter")
    assert sel.model.provider == "openrouter"


def test_raises_when_none_available():
    config = EngineConfig(provider_ratios={})
    with patch("vera_engine.auto_select.check_all", return_value=_none_available()):
        with pytest.raises(ValueError, match="no usable model"):
            select_model(config)


def test_only_anthropic_available():
    config = EngineConfig(provider_ratios={})
    with patch("vera_engine.auto_select.check_all", return_value=_only_anthropic_oauth()):
        sel = select_model(config)
    assert sel.model.provider == "anthropic"


def test_only_openrouter_available():
    config = EngineConfig(provider_ratios={})
    with patch("vera_engine.auto_select.check_all", return_value=_only_openrouter()):
        sel = select_model(config)
    assert sel.model.provider == "openrouter"


def test_skips_low_capacity_provider():
    config = EngineConfig(provider_ratios={"anthropic": 1.0, "openrouter": 0.01, "openai": 1.0})
    with patch("vera_engine.auto_select.check_all", return_value=_openrouter_low()):
        sel = select_model(config)
    assert sel.model.provider != "openrouter"


def test_oauth_provider_gets_none_strategy():
    config = EngineConfig(provider_ratios={})
    with patch("vera_engine.auto_select.check_all", return_value=_only_anthropic_oauth()):
        sel = select_model(config)
    assert sel.strategy == "none"


def test_api_key_provider_gets_env_key_strategy():
    config = EngineConfig(provider_ratios={})
    with patch("vera_engine.auto_select.check_all", return_value=_only_anthropic_api_key()):
        sel = select_model(config)
    assert sel.strategy == "env-key"


def test_openrouter_gets_env_key_strategy():
    config = EngineConfig(provider_ratios={})
    with patch("vera_engine.auto_select.check_all", return_value=_only_openrouter()):
        sel = select_model(config)
    assert sel.strategy == "env-key"


def test_usable_providers_filters_unavailable():
    caps = _none_available()
    assert _usable_providers(caps) == set()


def test_usable_providers_filters_low_pct():
    caps = {
        "anthropic": _cap("anthropic", pct=_LOW_PCT_THRESHOLD - 1),
        "openai": _cap("openai"),
    }
    usable = _usable_providers(caps)
    assert "anthropic" not in usable
    assert "openai" in usable


def test_usable_providers_filters_low_usd():
    caps = {
        "openrouter": _cap("openrouter", usd=_LOW_USD_THRESHOLD - 0.5),
    }
    assert _usable_providers(caps) == set()
