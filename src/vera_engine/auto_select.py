"""Cost-based model and engine auto-selection with capacity awareness."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from vera_engine.capacity import ProviderCapacity, check_all
from vera_engine.config import EngineConfig
from vera_engine.models import ModelSpec, Tier, get_catalog

logger = logging.getLogger(__name__)

_LOW_PCT_THRESHOLD = 5.0
_LOW_USD_THRESHOLD = 1.0


@dataclass(frozen=True)
class Selection:
    model: ModelSpec
    strategy: str
    effective_cost: float


def qualifying_models(
    config: EngineConfig,
    *,
    engine: str | None = None,
    engines: frozenset[str] | set[str] | None = None,
    provider: str | None = None,
    tier: Tier | None = None,
    exclude: frozenset[str] | set[str] | None = None,
    pins: frozenset[str] | set[str] | None = None,
    catalog: tuple[ModelSpec, ...] | None = None,
) -> list[ModelSpec]:
    """Catalog rows that meet the filters, cheapest first. No capacity probe.

    Pass ``catalog`` as the import-time ``CATALOG`` so two dispatches in one
    process cannot disagree because a price moved. Default ``None`` keeps ``get_catalog()``.
    """
    if engine is not None and engines is not None and engine not in engines:
        raise ValueError(f"engine {engine!r} is not in engines allowlist {sorted(engines)}")

    excluded = exclude or set()
    rows = catalog if catalog is not None else get_catalog()
    candidates: list[tuple[float, str, ModelSpec]] = []

    for spec in rows:
        if spec.model_id in excluded:
            continue
        if pins is not None and spec.model_id not in pins:
            continue
        if engine is not None and spec.engine != engine:
            continue
        if engines is not None and spec.engine not in engines:
            continue
        if provider is not None and spec.provider != provider:
            continue
        if tier is not None and spec.tier < tier:
            continue
        effective = spec.effective_cost(config.cost_ratio(spec.provider), config.input_weight)
        candidates.append((effective, spec.model_id, spec))

    candidates.sort()
    return [spec for _, _, spec in candidates]


def select_cheapest(
    config: EngineConfig,
    *,
    engine: str | None = None,
    engines: frozenset[str] | set[str] | None = None,
    provider: str | None = None,
    tier: Tier | None = None,
    exclude: frozenset[str] | set[str] | None = None,
    pins: frozenset[str] | set[str] | None = None,
    catalog: tuple[ModelSpec, ...] | None = None,
) -> Selection:
    """Cheapest qualifier at or above ``tier``. No capacity probe.

    Async dispatch uses this. ``select_model`` probes live capacity and must stay local-only.
    """
    rows = qualifying_models(
        config,
        engine=engine,
        engines=engines,
        provider=provider,
        tier=tier,
        exclude=exclude,
        pins=pins,
        catalog=catalog,
    )
    if not rows:
        raise ValueError("no usable model found")
    spec = rows[0]
    cost = spec.effective_cost(config.cost_ratio(spec.provider), config.input_weight)
    return Selection(model=spec, strategy="env-key", effective_cost=cost)


def select_model(
    config: EngineConfig,
    *,
    engine: str | None = None,
    provider: str | None = None,
    tier: Tier | None = None,
    exclude: frozenset[str] | set[str] | None = None,
) -> Selection:
    """Pick the cheapest model at or above the requested tier.

    When tier is set, only models with tier >= the requested tier are
    considered. This is the minimum capability floor — ``--tier stone``
    includes stone, bronze, and iron models.

    ``exclude`` is a set of model IDs to skip. Vera uses this for retry
    after failure: blacklist the model that just failed, call again with
    the same tier, and vera-engine picks the next cheapest.

    Returns a Selection with the model, inferred credential strategy,
    and effective cost. Raises ValueError if nothing is usable.
    """
    capacities = check_all()
    usable = _usable_providers(capacities)
    candidates = [
        spec
        for spec in qualifying_models(
            config, engine=engine, provider=provider, tier=tier, exclude=exclude
        )
        if spec.provider in usable
    ]

    if not candidates:
        detail = {p: c.detail for p, c in capacities.items()}
        raise ValueError(f"no usable model found (capacity: {detail})")

    spec = candidates[0]
    cost = spec.effective_cost(config.cost_ratio(spec.provider), config.input_weight)
    cap = capacities.get(spec.provider)
    if cap and cap.remaining_pct is not None and cap.remaining_pct < 20:
        logger.warning(
            "%s capacity low (%s) — consider switching providers",
            spec.provider,
            cap.detail,
        )
    strategy = _infer_strategy(spec, capacities)
    return Selection(model=spec, strategy=strategy, effective_cost=cost)


def _infer_strategy(spec: ModelSpec, capacities: dict[str, ProviderCapacity]) -> str:
    cap = capacities.get(spec.provider)
    if cap and cap.auth_method == "oauth":
        return "none"
    return "env-key"


def _usable_providers(capacities: dict[str, ProviderCapacity]) -> set[str]:
    usable = set()
    for provider, cap in capacities.items():
        if not cap.available:
            continue
        if cap.remaining_pct is not None and cap.remaining_pct < _LOW_PCT_THRESHOLD:
            logger.info("skipping %s: capacity too low (%s)", provider, cap.detail)
            continue
        if cap.remaining_usd is not None and cap.remaining_usd < _LOW_USD_THRESHOLD:
            logger.info("skipping %s: balance too low (%s)", provider, cap.detail)
            continue
        usable.add(provider)
    return usable
