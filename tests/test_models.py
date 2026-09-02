"""Tests for model catalog."""

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


def test_catalog_has_direct_grok_46():
    rows = [s for s in get_catalog() if s.model_id == "grok-4.6"]
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
