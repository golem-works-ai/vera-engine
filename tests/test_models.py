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
    assert by_id["gpt-5.6-sol"].tier == Tier.bronze
    assert "gpt-5.5" not in by_id
    assert by_id["openrouter/openai/gpt-5.5"].tier == Tier.stone
    assert "gpt-5.6-terra" in by_id


def test_catalog_tiers_from_2026_10_22_plan_benchmark():
    by_id = {s.model_id: s for s in get_catalog()}
    assert by_id["claude-opus-5-5"].tier == Tier.iron
    assert by_id["openrouter/anthropic/claude-opus-5.5"].tier == Tier.iron
    assert by_id["claude-sonnet-5-5"].tier == Tier.bronze
    assert by_id["claude-opus-4-8"].tier == Tier.bronze
    assert by_id["openrouter/z-ai/glm-5.1"].tier == Tier.clay


def test_clay_catalog_exact_qualified_rows(monkeypatch):
    monkeypatch.setattr("vera_engine.pricing.openrouter_model_costs", lambda _: None)
    rows = {s.model_id: s for s in get_catalog() if s.tier == Tier.clay}
    assert {model: spec.effort for model, spec in rows.items()} == {
        "claude-haiku-5-5": None,
        "openrouter/anthropic/claude-haiku-5.5": None,
        "openrouter/z-ai/glm-5.1": None,
        "gpt-6-luna": None,
        "openrouter/openai/gpt-6-luna": None,
        "openrouter/xiaomi/mimo-v2.6-flash": "low",
        "openrouter/z-ai/glm-5.3-flash": "low",
        "openrouter/qwen/qwen3.8-27b": "low",
    }
    for model, prices in {
        "openrouter/xiaomi/mimo-v2.6-flash": (0.14, 0.28),
        "openrouter/z-ai/glm-5.3-flash": (0.036, 0.90),
        "openrouter/qwen/qwen3.8-27b": (0.04, 2.30),
    }.items():
        spec = rows[model]
        assert (spec.engine, spec.provider) == ("opencode", "openrouter")
        assert (spec.input_cost, spec.output_cost) == prices


@pytest.mark.parametrize("effort", [None, "low", "medium", "high"])
def test_model_spec_effort(effort):
    spec = ModelSpec("test", "codex", "openai", 1.0, 2.0, Tier.clay, effort=effort)
    assert spec.effort == effort


def test_model_def_and_spec_reject_invalid_effort():
    from vera_engine.models import _ModelDef

    with pytest.raises(ValueError, match="effort must be one of"):
        ModelSpec("test", "codex", "openai", 1.0, 2.0, Tier.clay, effort="extreme")
    with pytest.raises(ValueError, match="effort must be one of"):
        _ModelDef("test", "codex", "openai", "", 1.0, 2.0, Tier.clay, effort="extreme")
