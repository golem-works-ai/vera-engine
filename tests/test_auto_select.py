"""Tests for cost-based auto-selection with capacity awareness."""

from unittest.mock import patch

import pytest

from vera_engine.auto_select import select_model, _usable_providers, _LOW_PCT_THRESHOLD, _LOW_USD_THRESHOLD
from vera_engine.capacity import ProviderCapacity
from vera_engine.config import EngineConfig
from vera_engine.models import Tier, get_catalog


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


def test_tier_stone_excludes_clay():
    config = EngineConfig(provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0})
    with patch("vera_engine.auto_select.check_all", return_value=_all_healthy()):
        sel = select_model(config, tier=Tier.stone)
    assert sel.model.tier >= Tier.stone


def test_tier_bronze_selects_cheapest_at_or_above_bronze():
    config = EngineConfig(provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0})
    with patch("vera_engine.auto_select.check_all", return_value=_all_healthy()):
        sel = select_model(config, tier=Tier.bronze)
    assert sel.model.tier >= Tier.bronze
    # grok-4.6 (added 2026-08-15) is the cheapest bronze-tier model in the
    # catalog; assert cost-minimality instead of a specific model_id so this
    # test doesn't need updating every time a cheaper model is added.
    bronze_or_above = [m for m in get_catalog() if m.tier >= Tier.bronze]
    cheapest = min(bronze_or_above, key=lambda m: m.blended_cost())
    assert sel.model.model_id == cheapest.model_id


def test_tier_iron_selects_fable():
    config = EngineConfig(provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0})
    with patch("vera_engine.auto_select.check_all", return_value=_all_healthy()):
        sel = select_model(config, tier=Tier.iron)
    assert sel.model.tier >= Tier.iron
    assert "fable" in sel.model.model_id


def test_tier_none_includes_all():
    config = EngineConfig(provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0})
    with patch("vera_engine.auto_select.check_all", return_value=_all_healthy()):
        sel = select_model(config, tier=None)
    from vera_engine.models import get_catalog
    cheapest = min(s.blended_cost(config.input_weight) for s in get_catalog())
    assert sel.model.blended_cost(config.input_weight) == pytest.approx(cheapest)


# ── exclude (model blacklist) ───────────────────────────────────────────────


def test_exclude_skips_blacklisted_model():
    config = EngineConfig(provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0})
    with patch("vera_engine.auto_select.check_all", return_value=_all_healthy()):
        first = select_model(config)
        second = select_model(config, exclude={first.model.model_id})
    assert second.model.model_id != first.model.model_id


def test_exclude_all_raises():
    from vera_engine.models import get_catalog
    all_ids = {s.model_id for s in get_catalog()}
    config = EngineConfig(provider_ratios={})
    with patch("vera_engine.auto_select.check_all", return_value=_all_healthy()):
        with pytest.raises(ValueError, match="no usable model"):
            select_model(config, exclude=all_ids)


def test_exclude_with_tier_picks_next_at_tier():
    config = EngineConfig(provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0})
    with patch("vera_engine.auto_select.check_all", return_value=_all_healthy()):
        first = select_model(config, tier=Tier.stone)
        second = select_model(config, tier=Tier.stone, exclude={first.model.model_id})
    assert second.model.tier >= Tier.stone
    assert second.model.model_id != first.model.model_id


def test_exclude_empty_set_is_noop():
    config = EngineConfig(provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0})
    with patch("vera_engine.auto_select.check_all", return_value=_all_healthy()):
        without = select_model(config)
        with_empty = select_model(config, exclude=set())
    assert without.model.model_id == with_empty.model.model_id
