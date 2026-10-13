"""Tests for model catalog."""

from unittest.mock import patch

import pytest

from vera_engine.models import (
    CATALOG,
    PROVIDER_CREDENTIALS,
    TIER_NAMES,
    ModelSpec,
    Tier,
    get_catalog,
)


def test_model_spec_blended_cost():
    spec = ModelSpec("test", "test-engine", "test-provider", 2.0, 10.0, Tier.stone)
    assert spec.blended_cost() == pytest.approx(3.6)
    assert spec.blended_cost(0.5) == pytest.approx(6.0)
    assert spec.blended_cost(1.0) == pytest.approx(2.0)
    assert spec.blended_cost(0.0) == pytest.approx(10.0)


def test_model_spec_effective_cost():
    spec = ModelSpec("test", "test-engine", "test-provider", 2.0, 10.0, Tier.stone)
    assert spec.effective_cost(0.5, 0.8) == pytest.approx(1.8)


def test_tier_ordering():
    assert Tier.clay < Tier.stone < Tier.bronze < Tier.iron


def test_tier_names():
    assert TIER_NAMES == ("clay", "stone", "bronze", "iron")


def test_all_catalog_models_have_valid_tier():
    for spec in CATALOG:
        assert isinstance(spec.tier, Tier), f"{spec.model_id} has invalid tier"


def test_catalog_not_empty():
    assert len(CATALOG) > 0


def test_get_catalog_returns_specs():
    catalog = get_catalog()
    assert len(catalog) > 0
    assert all(isinstance(s, ModelSpec) for s in catalog)


def test_all_catalog_providers_have_credentials():
    providers = {s.provider for s in CATALOG}
    for p in providers:
        assert p in PROVIDER_CREDENTIALS, (
            f"provider {p!r} missing from PROVIDER_CREDENTIALS"
        )


def test_all_catalog_costs_positive():
    for spec in CATALOG:
        assert spec.input_cost > 0, f"{spec.model_id} has non-positive input cost"
        assert spec.output_cost > 0, f"{spec.model_id} has non-positive output cost"


def test_model_ids_unique():
    ids = [s.model_id for s in CATALOG]
    assert len(ids) == len(set(ids)), f"duplicate model IDs: {ids}"


def test_catalog_has_direct_grok_47():
    rows = [s for s in get_catalog() if s.model_id == "grok-4.7"]
    assert len(rows) == 1
    spec = rows[0]
    assert spec.engine == "grok"
    assert spec.provider == "xai"
    assert spec.tier == Tier.bronze
    assert spec.input_cost > 0
    assert spec.output_cost > 0


def test_catalog_has_no_grok_45():
    ids = [s.model_id for s in get_catalog()]
    assert not any("grok-4.5" in model_id for model_id in ids)


def test_xai_provider_credential_is_xai_api_key():
    assert PROVIDER_CREDENTIALS["xai"] == "XAI_API_KEY"


def test_catalog_has_gpt6_codex_and_openrouter_rows():
    by_id = {s.model_id: s for s in get_catalog()}
    assert by_id["gpt-6-sol"].engine == "codex"
    assert by_id["gpt-6-luna"].engine == "codex"
    assert by_id["gpt-6-astra"].tier == Tier.iron
    assert by_id["openrouter/openai/gpt-6.1-sol"].engine == "opencode"
    assert by_id["openrouter/openai/gpt-6-luna"].engine == "opencode"
    assert "gpt-5.6-sol" not in by_id
    assert "gpt-5.6-terra" in by_id


@pytest.mark.parametrize(
    ("model_id", "engine", "provider"),
    [
        ("claude-opus-5-5", "claude-code", "anthropic"),
        ("openrouter/anthropic/claude-opus-5.5", "opencode", "openrouter"),
    ],
)
def test_opus_55_uses_provider_pricing_with_offline_fallback(model_id, engine, provider):
    by_id = {spec.model_id: spec for spec in get_catalog()}
    spec = by_id[model_id]
    assert (spec.engine, spec.provider, spec.tier) == (engine, provider, Tier.bronze)
    assert (spec.input_cost, spec.output_cost) == (4.0, 20.0)

    def live_costs(openrouter_id):
        if openrouter_id == "anthropic/claude-opus-5.5":
            return (3.0, 15.0)
        return None

    with patch("vera_engine.pricing.openrouter_model_costs", side_effect=live_costs):
        by_id = {spec.model_id: spec for spec in get_catalog()}
    spec = by_id[model_id]
    assert (spec.input_cost, spec.output_cost) == (3.0, 15.0)


def test_catalog_retires_legacy_opus_models():
    ids = {spec.model_id for spec in get_catalog()}
    assert not ids.intersection(
        {
            "claude-opus-5",
            "openrouter/anthropic/claude-opus-5",
            "claude-opus-4-6",
            "openrouter/anthropic/claude-opus-4.6",
        }
    )
