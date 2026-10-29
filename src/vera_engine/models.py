"""Model catalog with cost data and capability tiers.

All pricing comes from the OpenRouter API (cached 24h, stdlib-only fetch).
Hardcoded fallbacks cover offline/failure scenarios.

Cost blending uses input_weight from config (default 0.8) to reflect that
coding agent runs are ~80% input tokens by volume.

Tiers represent rough capability bands (ascending):
  clay < stone < bronze < iron
Selecting --tier X includes all models at tier X or above.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


class Tier(IntEnum):
    clay = 0
    stone = 1
    bronze = 2
    iron = 3


TIER_NAMES: tuple[str, ...] = tuple(t.name for t in Tier)


@dataclass(frozen=True)
class ModelSpec:
    """A model available for coding agent runs."""

    model_id: str
    engine: str
    provider: str
    input_cost: float
    output_cost: float
    tier: Tier

    def blended_cost(self, input_weight: float = 0.8) -> float:
        return self.input_cost * input_weight + self.output_cost * (1 - input_weight)

    def effective_cost(self, cost_ratio: float, input_weight: float = 0.8) -> float:
        return self.blended_cost(input_weight) * cost_ratio


PROVIDER_CREDENTIALS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
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
    tier: Tier


_MODEL_DEFS: tuple[_ModelDef, ...] = (
    # Anthropic direct via claude-code
    _ModelDef(
        "claude-fable-5",
        "claude-code",
        "anthropic",
        "anthropic/claude-fable-5",
        10.0,
        25.0,
        Tier.iron,
    ),
    _ModelDef(
        "claude-opus-5",
        "claude-code",
        "anthropic",
        "anthropic/claude-opus-5",
        5.0,
        25.0,
        Tier.bronze,
    ),
    _ModelDef(
        "claude-sonnet-5",
        "claude-code",
        "anthropic",
        "anthropic/claude-sonnet-5",
        2.0,
        10.0,
        Tier.stone,
    ),
    # Tiers below come from the 2026-10-22 plan-review benchmark (vera
    # docs/presentations/onboarding-users, slide 8). Opus 5.5 ties Fable 5.
    _ModelDef(
        "claude-opus-5-5",
        "claude-code",
        "anthropic",
        "anthropic/claude-opus-5.5",
        4.0,
        20.0,
        Tier.iron,
    ),
    _ModelDef(
        "claude-sonnet-5-5",
        "claude-code",
        "anthropic",
        "anthropic/claude-sonnet-5.5",
        2.0,
        10.0,
        Tier.bronze,
    ),
    _ModelDef(
        "claude-opus-4-8",
        "claude-code",
        "anthropic",
        "anthropic/claude-opus-4.8",
        5.0,
        25.0,
        Tier.bronze,
    ),
    # 4.6 models use ~30% fewer tokens per task. Costs are token-adjusted
    # (actual $/MTok * 0.7) to reflect per-task savings. No live pricing
    # override — the adjustment is the point.
    _ModelDef(
        "claude-opus-4-6", "claude-code", "anthropic", "", 3.50, 17.50, Tier.bronze
    ),
    _ModelDef(
        "claude-sonnet-4-6", "claude-code", "anthropic", "", 2.10, 10.50, Tier.stone
    ),
    _ModelDef(
        "claude-haiku-5-5",
        "claude-code",
        "anthropic",
        "anthropic/claude-haiku-5.5",
        0.10,
        0.50,
        Tier.clay,
    ),
    # OpenAI via codex
    # Prices from openrouter.ai/api/v1/models on 2026-10-11. The codex CLI on a
    # ChatGPT account accepts gpt-6-sol/luna/astra and gpt-5.6-sol (probed
    # 2026-10-22), not gpt-6.1-* or gpt-5.5.
    _ModelDef(
        "gpt-6-astra", "codex", "openai", "openai/gpt-6-astra", 10.0, 50.0, Tier.iron
    ),
    _ModelDef(
        "gpt-6-sol", "codex", "openai", "openai/gpt-6-sol", 2.0, 10.0, Tier.bronze
    ),
    _ModelDef(
        "gpt-5.6-sol", "codex", "openai", "openai/gpt-5.6-sol", 4.0, 20.0, Tier.bronze
    ),
    _ModelDef(
        "gpt-5.6-terra", "codex", "openai", "openai/gpt-5.6-terra", 1.0, 6.0, Tier.stone
    ),
    _ModelDef(
        "gpt-6-luna", "codex", "openai", "openai/gpt-6-luna", 0.10, 0.50, Tier.clay
    ),
    # OpenRouter via opencode
    _ModelDef(
        "openrouter/anthropic/claude-fable-5",
        "opencode",
        "openrouter",
        "anthropic/claude-fable-5",
        5.0,
        25.0,
        Tier.iron,
    ),
    _ModelDef(
        "openrouter/anthropic/claude-opus-5",
        "opencode",
        "openrouter",
        "anthropic/claude-opus-5",
        5.0,
        25.0,
        Tier.bronze,
    ),
    _ModelDef(
        "openrouter/anthropic/claude-sonnet-5",
        "opencode",
        "openrouter",
        "anthropic/claude-sonnet-5",
        2.0,
        10.0,
        Tier.stone,
    ),
    _ModelDef(
        "openrouter/anthropic/claude-opus-5.5",
        "opencode",
        "openrouter",
        "anthropic/claude-opus-5.5",
        4.0,
        20.0,
        Tier.iron,
    ),
    _ModelDef(
        "openrouter/anthropic/claude-sonnet-5.5",
        "opencode",
        "openrouter",
        "anthropic/claude-sonnet-5.5",
        2.0,
        10.0,
        Tier.bronze,
    ),
    _ModelDef(
        "openrouter/anthropic/claude-opus-4.8",
        "opencode",
        "openrouter",
        "anthropic/claude-opus-4.8",
        5.0,
        25.0,
        Tier.bronze,
    ),
    _ModelDef(
        "openrouter/anthropic/claude-opus-4.6",
        "opencode",
        "openrouter",
        "",
        3.50,
        17.50,
        Tier.bronze,
    ),
    _ModelDef(
        "openrouter/anthropic/claude-sonnet-4.6",
        "opencode",
        "openrouter",
        "",
        2.10,
        10.50,
        Tier.stone,
    ),
    _ModelDef(
        "openrouter/anthropic/claude-haiku-5.5",
        "opencode",
        "openrouter",
        "anthropic/claude-haiku-5.5",
        0.10,
        0.50,
        Tier.clay,
    ),
    _ModelDef(
        "openrouter/moonshotai/kimi-k3",
        "opencode",
        "openrouter",
        "moonshotai/kimi-k3",
        3.0,
        15.0,
        Tier.bronze,
    ),
    _ModelDef(
        "openrouter/moonshotai/kimi-k2.7-code",
        "opencode",
        "openrouter",
        "moonshotai/kimi-k2.7-code",
        0.67,
        3.40,
        Tier.clay,
    ),
    _ModelDef(
        "openrouter/openai/gpt-6-astra",
        "opencode",
        "openrouter",
        "openai/gpt-6-astra",
        10.0,
        50.0,
        Tier.iron,
    ),
    _ModelDef(
        "openrouter/openai/gpt-6.1-sol",
        "opencode",
        "openrouter",
        "openai/gpt-6.1-sol",
        2.0,
        10.0,
        Tier.bronze,
    ),
    # Codex rejects gpt-5.5 on a ChatGPT account, so it lives here only.
    _ModelDef(
        "openrouter/openai/gpt-5.5",
        "opencode",
        "openrouter",
        "openai/gpt-5.5",
        5.0,
        30.0,
        Tier.stone,
    ),
    _ModelDef(
        "openrouter/openai/gpt-5.6-terra",
        "opencode",
        "openrouter",
        "openai/gpt-5.6-terra",
        1.0,
        6.0,
        Tier.stone,
    ),
    _ModelDef(
        "openrouter/openai/gpt-6-luna",
        "opencode",
        "openrouter",
        "openai/gpt-6-luna",
        0.10,
        0.50,
        Tier.clay,
    ),
    # xAI direct via grok. Same list price as the OpenRouter row.
    # Capacity picks oauth vs env-key. Do not add grok-4.5.
    _ModelDef("grok-4.7", "grok", "xai", "x-ai/grok-4.7", 2.0, 6.0, Tier.bronze),
    # Pricing fetched live from openrouter.ai/api/v1/models on 2026-10-11.
    _ModelDef(
        "openrouter/x-ai/grok-4.7",
        "opencode",
        "openrouter",
        "x-ai/grok-4.7",
        2.0,
        6.0,
        Tier.bronze,
    ),
    _ModelDef(
        "openrouter/moonshotai/kimi-k2.6",
        "opencode",
        "openrouter",
        "moonshotai/kimi-k2.6",
        0.5415,
        2.28,
        Tier.clay,
    ),
    _ModelDef(
        "openrouter/google/gemini-3.7-flash",
        "opencode",
        "openrouter",
        "google/gemini-3.7-flash",
        0.375,
        1.875,
        Tier.clay,
    ),
    # Pricing fetched live from openrouter.ai/api/v1/models on 2026-10-09.
    _ModelDef(
        "openrouter/z-ai/glm-5.3",
        "opencode",
        "openrouter",
        "z-ai/glm-5.3",
        0.04,
        4.8,
        Tier.stone,
    ),
    _ModelDef(
        "openrouter/z-ai/glm-5.1",
        "opencode",
        "openrouter",
        "z-ai/glm-5.1",
        1.4,
        4.4,
        Tier.clay,
    ),
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
            tier=d.tier,
        )
        for d in _MODEL_DEFS
        for costs in [_resolve_costs(d)]
    )


CATALOG = get_catalog()
