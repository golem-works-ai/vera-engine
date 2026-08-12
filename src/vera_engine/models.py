"""Model catalog with cost data.

All pricing comes from the OpenRouter API (cached 24h, stdlib-only fetch).
Hardcoded fallbacks cover offline/failure scenarios.

Cost blending uses input_weight from config (default 0.8) to reflect that
coding agent runs are ~80% input tokens by volume.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """A model available for coding agent runs."""

    model_id: str
    engine: str
    provider: str
    input_cost: float
    output_cost: float

    def blended_cost(self, input_weight: float = 0.8) -> float:
        return self.input_cost * input_weight + self.output_cost * (1 - input_weight)

    def effective_cost(self, cost_ratio: float, input_weight: float = 0.8) -> float:
        return self.blended_cost(input_weight) * cost_ratio


PROVIDER_CREDENTIALS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
}


@dataclass(frozen=True)
class _ModelDef:
    """Internal definition used to build the catalog."""

    model_id: str
    engine: str
    provider: str
    openrouter_id: str
    fallback_input: float
    fallback_output: float


_MODEL_DEFS: tuple[_ModelDef, ...] = (
    # Anthropic direct via claude-code
    _ModelDef("claude-opus-5", "claude-code", "anthropic", "anthropic/claude-opus-5", 5.0, 25.0),
    _ModelDef("claude-sonnet-5", "claude-code", "anthropic", "anthropic/claude-sonnet-5", 2.0, 10.0),
    _ModelDef("claude-haiku-4-5-20251001", "claude-code", "anthropic", "anthropic/claude-haiku-4.5", 1.0, 5.0),
    # OpenRouter via opencode
    _ModelDef("openrouter/anthropic/claude-opus-5", "opencode", "openrouter", "anthropic/claude-opus-5", 5.0, 25.0),
    _ModelDef("openrouter/anthropic/claude-sonnet-5", "opencode", "openrouter", "anthropic/claude-sonnet-5", 2.0, 10.0),
    _ModelDef("openrouter/anthropic/claude-haiku-4.5", "opencode", "openrouter", "anthropic/claude-haiku-4.5", 1.0, 5.0),
)


def _resolve_costs(d: _ModelDef) -> tuple[float, float]:
    if not d.openrouter_id:
        return (d.fallback_input, d.fallback_output)
    from vera_engine.pricing import openrouter_model_costs
    live = openrouter_model_costs(d.openrouter_id)
    if live is not None:
        return live
    return (d.fallback_input, d.fallback_output)


def get_catalog() -> tuple[ModelSpec, ...]:
    """Build the full catalog with live pricing where available."""
    return tuple(
        ModelSpec(
            model_id=d.model_id,
            engine=d.engine,
            provider=d.provider,
            input_cost=costs[0],
            output_cost=costs[1],
        )
        for d in _MODEL_DEFS
        for costs in [_resolve_costs(d)]
    )


CATALOG = get_catalog()
