"""Cost-based model and engine auto-selection with capacity awareness."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from vera_engine.capacity import ProviderCapacity, check_all
from vera_engine.config import EngineConfig
from vera_engine.models import ModelSpec, get_catalog

logger = logging.getLogger(__name__)

_LOW_PCT_THRESHOLD = 5.0
_LOW_USD_THRESHOLD = 1.0


@dataclass(frozen=True)
class Selection:
    model: ModelSpec
    strategy: str
    effective_cost: float


def select_model(
    config: EngineConfig,
    *,
    engine: str | None = None,
    provider: str | None = None,
) -> Selection:
    """Pick the cheapest model with available capacity.

    Returns a Selection with the model, inferred credential strategy,
    and effective cost. Raises ValueError if nothing is usable.
    """
    capacities = check_all()
    usable = _usable_providers(capacities)
    candidates: list[tuple[float, str, ModelSpec]] = []

    for spec in get_catalog():
        if engine and spec.engine != engine:
            continue
        if provider and spec.provider != provider:
            continue
        if spec.provider not in usable:
            continue
        effective = spec.effective_cost(config.cost_ratio(spec.provider), config.input_weight)
        candidates.append((effective, spec.model_id, spec))

    if not candidates:
        detail = {p: c.detail for p, c in capacities.items()}
        raise ValueError(f"no usable model found (capacity: {detail})")

    candidates.sort()
    cost, _, spec = candidates[0]
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
