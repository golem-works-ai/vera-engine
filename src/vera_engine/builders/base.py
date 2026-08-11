"""Abstract builder interface for engine plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod

from vera_engine.request import AgentRunRequest
from vera_engine.invocation import EngineInvocation


class EngineBuilder(ABC):
    """Translates an AgentRunRequest into an EngineInvocation.

    Each engine (claude-code, opencode, codex, etc.) implements one builder.
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Canonical engine identifier (e.g. 'claude-code')."""
        ...

    @property
    @abstractmethod
    def supported_strategies(self) -> frozenset[str]:
        """Credential strategies this engine supports."""
        ...

    @abstractmethod
    def build_invocation(
        self, request: AgentRunRequest, strategy: str
    ) -> EngineInvocation:
        """Build a fully resolved invocation from a request."""
        ...

    @abstractmethod
    def default_model(self) -> str | None:
        """Default model for this engine, or None if the engine picks its own."""
        ...

    def validate_strategy(self, strategy: str) -> None:
        """Raise if the strategy is not supported by this engine."""
        if strategy not in self.supported_strategies:
            raise ValueError(
                f"engine {self.engine_name!r} does not support "
                f"credential_strategy={strategy!r} "
                f"(supported: {sorted(self.supported_strategies)})"
            )
