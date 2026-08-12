"""Tests for model catalog."""

import pytest

from vera_engine.models import CATALOG, ModelSpec, PROVIDER_CREDENTIALS, get_catalog


def test_model_spec_blended_cost():
    spec = ModelSpec("test", "test-engine", "test-provider", 2.0, 10.0)
    assert spec.blended_cost() == pytest.approx(3.6)
    assert spec.blended_cost(0.5) == pytest.approx(6.0)
    assert spec.blended_cost(1.0) == pytest.approx(2.0)
    assert spec.blended_cost(0.0) == pytest.approx(10.0)


def test_model_spec_effective_cost():
    spec = ModelSpec("test", "test-engine", "test-provider", 2.0, 10.0)
    assert spec.effective_cost(0.5, 0.8) == pytest.approx(1.8)


def test_catalog_not_empty():
    assert len(CATALOG) > 0


def test_get_catalog_returns_specs():
    catalog = get_catalog()
    assert len(catalog) > 0
    assert all(isinstance(s, ModelSpec) for s in catalog)


def test_all_catalog_providers_have_credentials():
    providers = {s.provider for s in CATALOG}
    for p in providers:
        assert p in PROVIDER_CREDENTIALS, f"provider {p!r} missing from PROVIDER_CREDENTIALS"


def test_all_catalog_costs_positive():
    for spec in CATALOG:
        assert spec.input_cost > 0, f"{spec.model_id} has non-positive input cost"
        assert spec.output_cost > 0, f"{spec.model_id} has non-positive output cost"


def test_model_ids_unique():
    ids = [s.model_id for s in CATALOG]
    assert len(ids) == len(set(ids)), f"duplicate model IDs: {ids}"
