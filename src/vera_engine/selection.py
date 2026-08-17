"""Engine builder registry and selection."""

from __future__ import annotations

from vera_engine.builders.base import EngineBuilder
from vera_engine.builders.claude_code import ClaudeCodeBuilder
from vera_engine.builders.codex import CodexBuilder
from vera_engine.builders.grok import GrokBuilder
from vera_engine.builders.opencode import OpenCodeBuilder

# Registry: engine name -> builder instance.
_BUILDERS: dict[str, EngineBuilder] = {}


def _register(builder: EngineBuilder) -> None:
    name = builder.engine_name
    if name in _BUILDERS:
        raise RuntimeError(f"duplicate engine registration: {name!r}")
    _BUILDERS[name] = builder


# Register built-in engines.
_register(ClaudeCodeBuilder())
_register(OpenCodeBuilder())
_register(CodexBuilder())
_register(GrokBuilder())


def get_builder(engine: str) -> EngineBuilder:
    """Return the builder for an engine name. Raises on unknown engine."""
    try:
        return _BUILDERS[engine]
    except KeyError:
        raise ValueError(
            f"unknown engine {engine!r} (available: {sorted(_BUILDERS.keys())})"
        ) from None


def list_engines() -> list[str]:
    """Return sorted list of registered engine names."""
    return sorted(_BUILDERS.keys())
