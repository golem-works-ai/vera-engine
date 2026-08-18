"""Tests for cost-based auto-selection with capacity awareness."""

from unittest.mock import patch

import pytest

from vera_engine import build_and_run
from vera_engine.auto_select import (
    _LOW_PCT_THRESHOLD,
    _LOW_USD_THRESHOLD,
    _usable_providers,
    qualifying_models,
    resolve_request,
    selection_cost,
    selection_cost_ratio,
    select_cheapest,
    select_model,
)
from vera_engine.capacity import ProviderCapacity
from vera_engine.config import EngineConfig
from vera_engine.invocation import RunResult
from vera_engine.models import ModelSpec, Tier, get_catalog
from vera_engine.request import AgentRunRequest
from vera_engine.selection import list_engines


def _cap(provider, available=True, pct=None, usd=None, detail="", auth="api-key"):
    return ProviderCapacity(provider, available, pct, usd, detail, auth)


def _all_healthy():
    return {
        "anthropic": _cap("anthropic", pct=80.0, auth="oauth"),
        "openrouter": _cap("openrouter", usd=500.0),
        "openai": _cap("openai"),
        "xai": _cap("xai", auth="oauth"),
    }


def _only_anthropic_oauth():
    return {
        "anthropic": _cap("anthropic", pct=80.0, auth="oauth"),
        "openrouter": _cap("openrouter", available=False),
        "openai": _cap("openai", available=False),
        "xai": _cap("xai", available=False),
    }


def _only_anthropic_api_key():
    return {
        "anthropic": _cap("anthropic", auth="api-key"),
        "openrouter": _cap("openrouter", available=False),
        "openai": _cap("openai", available=False),
        "xai": _cap("xai", available=False),
    }


def _only_openrouter():
    return {
        "anthropic": _cap("anthropic", available=False),
        "openrouter": _cap("openrouter", usd=500.0),
        "openai": _cap("openai", available=False),
        "xai": _cap("xai", available=False),
    }


def _only_xai_oauth():
    return {
        "anthropic": _cap("anthropic", available=False),
        "openrouter": _cap("openrouter", available=False),
        "openai": _cap("openai", available=False),
        "xai": _cap("xai", auth="oauth"),
    }


def _only_xai_api_key():
    return {
        "anthropic": _cap("anthropic", available=False),
        "openrouter": _cap("openrouter", available=False),
        "openai": _cap("openai", available=False),
        "xai": _cap("xai", auth="api-key"),
    }


def _none_available():
    return {
        "anthropic": _cap("anthropic", available=False),
        "openrouter": _cap("openrouter", available=False),
        "openai": _cap("openai", available=False),
        "xai": _cap("xai", available=False),
    }


def _openrouter_low():
    return {
        "anthropic": _cap("anthropic", pct=80.0, auth="oauth"),
        "openrouter": _cap("openrouter", usd=0.50),
        "openai": _cap("openai"),
        "xai": _cap("xai", auth="oauth"),
    }


def _is_opencode_anthropic(spec: ModelSpec) -> bool:
    return spec.engine == "opencode" and spec.model_id.startswith("openrouter/anthropic/")


def test_selects_cheapest_with_all_healthy():
    config = EngineConfig(provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0})
    capacities = _all_healthy()
    with patch("vera_engine.auto_select.check_all", return_value=capacities):
        sel = select_model(config)
    cheapest = min(
        selection_cost(spec, config, capacities)
        for spec in qualifying_models(config)
        if spec.provider in _usable_providers(capacities)
    )
    assert sel.effective_cost == pytest.approx(cheapest)


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
    config = EngineConfig(
        provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0, "xai": 1.0}
    )
    caps = _all_healthy()
    with patch("vera_engine.auto_select.check_all", return_value=caps):
        sel = select_model(config, tier=Tier.bronze)
    assert sel.model.tier >= Tier.bronze
    usable = _usable_providers(caps)
    ranked = [
        spec
        for spec in qualifying_models(config, tier=Tier.bronze)
        if spec.provider in usable
    ]
    assert ranked
    assert sel.model.model_id == ranked[0].model_id


def test_tier_bronze_never_selects_opencode_anthropic():
    config = EngineConfig(
        provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0, "xai": 1.0}
    )
    with patch("vera_engine.auto_select.check_all", return_value=_only_openrouter()):
        probed = select_model(config, tier=Tier.bronze)
    cheapest = select_cheapest(config, tier=Tier.bronze)
    assert not _is_opencode_anthropic(probed.model)
    assert not _is_opencode_anthropic(cheapest.model)
    rows = qualifying_models(config, engine="opencode", tier=Tier.bronze)
    assert rows
    assert all(not _is_opencode_anthropic(spec) for spec in rows)


def test_xai_subscription_selects_grok_none():
    config = EngineConfig(
        provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0, "xai": 0.2}
    )
    with patch("vera_engine.auto_select.check_all", return_value=_only_xai_oauth()):
        sel = select_model(config, tier=Tier.bronze)
    assert sel.model.engine == "grok"
    assert sel.model.model_id == "grok-4.6"
    assert sel.model.provider == "xai"
    assert sel.strategy == "none"


def test_xai_api_key_selects_grok_env_key():
    config = EngineConfig(
        provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0, "xai": 1.0}
    )
    with patch("vera_engine.auto_select.check_all", return_value=_only_xai_api_key()):
        sel = select_model(config, tier=Tier.bronze)
    assert sel.model.engine == "grok"
    assert sel.model.model_id == "grok-4.6"
    assert sel.strategy == "env-key"


def test_xai_unavailable_falls_back_to_openrouter_grok():
    config = EngineConfig(
        provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0, "xai": 1.0}
    )
    with patch("vera_engine.auto_select.check_all", return_value=_only_openrouter()):
        sel = select_model(config, tier=Tier.bronze)
    assert sel.model.model_id == "openrouter/x-ai/grok-4.6"
    assert sel.model.engine == "opencode"
    assert sel.strategy == "env-key"


def test_xai_oauth_without_toml_is_cheaper_than_list():
    config = EngineConfig(provider_ratios={})
    with patch("vera_engine.auto_select.check_all", return_value=_only_xai_oauth()):
        sel = select_model(config, tier=Tier.bronze)
    assert sel.model.provider == "xai"
    assert sel.strategy == "none"
    list_cost = sel.model.blended_cost(config.input_weight)
    assert sel.effective_cost == pytest.approx(list_cost * 0.1)


@pytest.mark.parametrize("provider", ["anthropic", "openai", "xai"])
def test_oauth_subscription_uses_default_subscription_ratio(provider):
    spec = ModelSpec("test", "test-engine", provider, 2.0, 6.0, Tier.clay)
    capacities = {provider: _cap(provider, auth="oauth")}
    config = EngineConfig(provider_ratios={})

    assert selection_cost_ratio(spec, config, capacities) == pytest.approx(0.1)
    assert selection_cost(spec, config, capacities) == pytest.approx(0.28)


def test_explicit_provider_ratio_overrides_oauth_subscription_default():
    spec = ModelSpec("test", "test-engine", "openai", 2.0, 6.0, Tier.clay)
    capacities = {"openai": _cap("openai", auth="oauth")}
    config = EngineConfig(provider_ratios={"openai": 0.25})

    assert selection_cost_ratio(spec, config, capacities) == pytest.approx(0.25)


def test_tier_iron_selects_fable():
    config = EngineConfig(provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0})
    with patch("vera_engine.auto_select.check_all", return_value=_all_healthy()):
        sel = select_model(config, tier=Tier.iron)
    assert sel.model.tier >= Tier.iron
    assert "fable" in sel.model.model_id


def test_tier_none_includes_all():
    config = EngineConfig(provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0})
    capacities = _all_healthy()
    with patch("vera_engine.auto_select.check_all", return_value=capacities):
        sel = select_model(config, tier=None)
    cheapest = min(
        selection_cost(spec, config, capacities)
        for spec in qualifying_models(config)
        if spec.provider in _usable_providers(capacities)
    )
    assert sel.effective_cost == pytest.approx(cheapest)


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


# ── no-probe cheapest-at-tier ───────────────────────────────────────────────


def _uniform_config() -> EngineConfig:
    return EngineConfig(
        provider_ratios={"anthropic": 1.0, "openrouter": 1.0, "openai": 1.0, "xai": 1.0}
    )


def _rank_key(config: EngineConfig, spec: ModelSpec) -> tuple[float, str]:
    return (
        spec.effective_cost(config.cost_ratio(spec.provider), config.input_weight),
        spec.model_id,
    )


def test_select_cheapest_picks_lowest_effective_cost():
    config = _uniform_config()
    winner = min(get_catalog(), key=lambda spec: _rank_key(config, spec))
    sel = select_cheapest(config, tier=None)
    assert sel.model.model_id == winner.model_id
    assert sel.strategy == "env-key"
    assert sel.effective_cost == pytest.approx(_rank_key(config, winner)[0])


def test_select_cheapest_tier_clay_includes_higher_tiers():
    config = _uniform_config()
    rows = qualifying_models(config, tier=Tier.clay)
    assert rows
    assert all(spec.tier >= Tier.clay for spec in rows)
    sel = select_cheapest(config, tier=Tier.clay)
    assert sel.model.tier >= Tier.clay


def test_select_cheapest_tier_iron_excludes_below():
    config = _uniform_config()
    rows = qualifying_models(config, tier=Tier.iron)
    assert rows
    assert all(spec.tier >= Tier.iron for spec in rows)
    sel = select_cheapest(config, tier=Tier.iron)
    assert sel.model.tier >= Tier.iron


def test_select_cheapest_engine_filter():
    config = EngineConfig(provider_ratios={})
    rows = qualifying_models(config, engine="codex")
    assert rows
    assert all(spec.engine == "codex" for spec in rows)
    sel = select_cheapest(config, engine="codex")
    assert sel.model.engine == "codex"


def test_select_cheapest_engines_allowlist():
    config = EngineConfig(provider_ratios={})
    rows = qualifying_models(config, engines={"claude-code", "opencode"})
    assert rows
    assert all(spec.engine != "codex" for spec in rows)
    sel = select_cheapest(config, engines={"claude-code", "opencode"})
    assert sel.model.engine != "codex"


def test_select_cheapest_engine_not_in_allowlist_raises():
    config = EngineConfig(provider_ratios={})
    with pytest.raises(ValueError):
        select_cheapest(config, engine="codex", engines={"opencode"})


def test_select_cheapest_exclude_moves_to_next():
    config = _uniform_config()
    first = select_cheapest(config)
    second = select_cheapest(config, exclude={first.model.model_id})
    ranked = qualifying_models(config)
    assert first.model.model_id == ranked[0].model_id
    assert second.model.model_id == ranked[1].model_id


def test_select_cheapest_pins_whitelist():
    config = EngineConfig(provider_ratios={})
    ranked = qualifying_models(config)
    some_id = ranked[-1].model_id
    sel = select_cheapest(config, pins={some_id})
    assert sel.model.model_id == some_id


def test_select_cheapest_empty_pins_raises():
    config = EngineConfig(provider_ratios={})
    with pytest.raises(ValueError, match="no usable model found"):
        select_cheapest(config, pins=frozenset())


def test_select_cheapest_does_not_call_check_all(monkeypatch):
    monkeypatch.setattr(
        "vera_engine.auto_select.check_all",
        lambda: (_ for _ in ()).throw(AssertionError("probed")),
    )
    config = _uniform_config()
    sel = select_cheapest(config)
    assert sel.model.model_id


def test_qualifying_models_uses_supplied_catalog(monkeypatch):
    config = EngineConfig(provider_ratios={})
    snap = ModelSpec(
        model_id="only-snap",
        engine="codex",
        provider="openai",
        input_cost=1.0,
        output_cost=1.0,
        tier=Tier.clay,
    )
    monkeypatch.setattr(
        "vera_engine.auto_select.get_catalog",
        lambda: (_ for _ in ()).throw(AssertionError("live catalog")),
    )
    rows = qualifying_models(config, catalog=(snap,))
    assert rows == [snap]


def test_qualifying_models_default_catalog_is_get_catalog(monkeypatch):
    config = EngineConfig(provider_ratios={})
    snap = ModelSpec(
        model_id="only-live",
        engine="codex",
        provider="openai",
        input_cost=1.0,
        output_cost=1.0,
        tier=Tier.clay,
    )
    monkeypatch.setattr("vera_engine.auto_select.get_catalog", lambda: (snap,))
    rows = qualifying_models(config)
    assert rows == [snap]


# ── resolve_request ──────────────────────────────────────────────────────────


def test_resolve_request_fills_engine_and_model_from_tier(tmp_path):
    req = AgentRunRequest(engine=None, prompt="x", workspace=tmp_path, tier=Tier.clay)
    resolved = resolve_request(req, _uniform_config(), probe=False)
    assert resolved.engine
    assert resolved.model
    matching = [spec for spec in get_catalog() if spec.model_id == resolved.model]
    assert matching
    assert matching[0].tier >= Tier.clay
    assert matching[0].engine == resolved.engine


def test_resolve_request_honors_pinned_engine(tmp_path):
    req = AgentRunRequest(engine="codex", prompt="x", workspace=tmp_path, tier=Tier.clay)
    resolved = resolve_request(req, _uniform_config(), probe=False)
    assert resolved.engine == "codex"
    assert resolved.model


def test_resolve_request_leaves_explicit_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vera_engine.auto_select.select_cheapest",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("select_cheapest")),
    )
    req = AgentRunRequest(engine="codex", prompt="x", workspace=tmp_path, model="gpt-5")
    resolved = resolve_request(req, EngineConfig(provider_ratios={}), probe=False)
    assert resolved.engine == "codex"
    assert resolved.model == "gpt-5"


def test_resolve_request_unknown_model_without_engine_raises(tmp_path):
    req = AgentRunRequest(
        engine=None, prompt="x", workspace=tmp_path, model="not-a-real-id"
    )
    with pytest.raises(ValueError, match="unknown model"):
        resolve_request(req, EngineConfig(provider_ratios={}))


def test_resolve_request_probe_false_does_not_call_check_all(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vera_engine.auto_select.check_all",
        lambda: (_ for _ in ()).throw(AssertionError("probed")),
    )
    req = AgentRunRequest(engine=None, prompt="x", workspace=tmp_path, tier=Tier.clay)
    resolved = resolve_request(req, _uniform_config(), probe=False)
    assert resolved.engine
    assert resolved.model


def test_build_and_run_resolves_missing_engine(tmp_path, monkeypatch):
    captured = {}

    def fake_render_local(invocation, bundle):
        captured["invocation"] = invocation
        return RunResult(
            returncode=0,
            stdout="",
            stderr="",
            timed_out=False,
            engine=invocation.engine,
            argv=invocation.argv,
        )

    monkeypatch.setattr("vera_engine.render_local", fake_render_local)
    result = build_and_run(
        AgentRunRequest(
            engine=None,
            prompt="hi",
            workspace=tmp_path,
            tier=Tier.clay,
            credential_strategy="none",
        )
    )
    assert "invocation" in captured
    assert captured["invocation"].engine
    assert captured["invocation"].engine in list_engines()
    assert result.engine == captured["invocation"].engine
