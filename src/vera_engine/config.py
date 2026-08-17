"""Configuration loading from TOML files."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_INPUT_WEIGHT = 0.8


@dataclass(frozen=True)
class EngineConfig:
    """Loaded configuration with provider cost ratios and cost weighting."""

    provider_ratios: dict[str, float] = field(default_factory=dict)
    input_weight: float = _DEFAULT_INPUT_WEIGHT

    def cost_ratio(self, provider: str) -> float:
        return self.provider_ratios.get(provider, 1.0)


_DEFAULT_RATIOS: dict[str, float] = {
    "anthropic": 1.0,
    "openrouter": 1.01375,
    "openai": 1.0,
    "xai": 1.0,
}


def _config_paths(workspace: Path | None = None) -> list[Path]:
    paths = []
    if workspace:
        paths.append(workspace / ".vera-engine.toml")
    paths.append(Path.home() / ".vera-engine.toml")
    return paths


def load_config(workspace: Path | None = None) -> EngineConfig:
    """Load config from the first .vera-engine.toml found.

    Search order: workspace dir, then home dir. If neither exists,
    return defaults (input_weight 0.8, openrouter 1.01375, others 1.0).
    """
    for path in _config_paths(workspace):
        if path.is_file():
            return _parse_config(path)
    return EngineConfig(provider_ratios=dict(_DEFAULT_RATIOS))


def _parse_config(path: Path) -> EngineConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    ratios = dict(_DEFAULT_RATIOS)
    providers = raw.get("providers", {})
    for name, section in providers.items():
        if isinstance(section, dict) and "cost_ratio" in section:
            ratio = section["cost_ratio"]
            if not isinstance(ratio, (int, float)) or ratio <= 0:
                raise ValueError(
                    f"providers.{name}.cost_ratio must be a positive number, "
                    f"got {ratio!r} in {path}"
                )
            ratios[name] = float(ratio)

    input_weight = raw.get("input_weight", _DEFAULT_INPUT_WEIGHT)
    if not isinstance(input_weight, (int, float)) or not (0.0 <= input_weight <= 1.0):
        raise ValueError(
            f"input_weight must be a number between 0.0 and 1.0, "
            f"got {input_weight!r} in {path}"
        )

    return EngineConfig(provider_ratios=ratios, input_weight=float(input_weight))
