"""AgentRunRequest: declarative description of what to run."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

from vera_engine.credentials import reject_credential_fields

_VALID_EFFORTS = frozenset({"low", "medium", "high"})
_VALID_STRATEGIES = frozenset({"env-key", "proxy", "none"})


@dataclass(frozen=True)
class AgentRunRequest:
    """Declarative request describing WHAT to run.

    Callers build this; the engine layer resolves HOW.
    """

    engine: str
    prompt: str
    workspace: Path
    model: str | None = None
    effort: str = "high"
    timeout_seconds: int = 2400
    extra_env: dict[str, str] = field(default_factory=dict)
    credential_strategy: str = "env-key"

    def __post_init__(self) -> None:
        if not self.engine:
            raise ValueError("engine must not be empty")
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if self.effort not in _VALID_EFFORTS:
            raise ValueError(
                f"effort must be one of {sorted(_VALID_EFFORTS)}, got {self.effort!r}"
            )
        if self.credential_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"credential_strategy must be one of {sorted(_VALID_STRATEGIES)}, "
                f"got {self.credential_strategy!r}"
            )
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        reject_credential_fields(type(self))


# Validate at import time that the dataclass fields are clean.
reject_credential_fields(AgentRunRequest)
