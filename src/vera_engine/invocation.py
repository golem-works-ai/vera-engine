"""EngineInvocation: resolved subprocess specification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vera_engine.credentials import SecretRef, reject_credential_fields

_VALID_HOME_STRATEGIES = frozenset({"hermetic", "shared", "real"})


@dataclass(frozen=True)
class MaterializedFile:
    """A file to write into the workspace before spawning the engine."""

    relative_path: str
    content: str
    mode: int = 0o644
    expand_references: bool = False
    cleanup: bool = True  # Remove after the engine exits.

    def __post_init__(self) -> None:
        if Path(self.relative_path).is_absolute():
            raise ValueError(
                f"relative_path must be workspace-relative, got {self.relative_path!r}"
            )
        if ".." in Path(self.relative_path).parts:
            raise ValueError(
                f"relative_path must not contain '..', got {self.relative_path!r}"
            )


@dataclass(frozen=True)
class EngineInvocation:
    """Fully resolved specification for spawning an engine subprocess.

    Built by an EngineBuilder from an AgentRunRequest.
    """

    engine: str
    argv: tuple[str, ...]
    env: dict[str, str | SecretRef]
    workdir: Path
    home_strategy: str = "hermetic"
    timeout_seconds: int = 2400
    files: tuple[MaterializedFile, ...] = ()
    prompt_path: Path | None = None
    resume: str | None = None

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("argv must not be empty")
        if self.home_strategy not in _VALID_HOME_STRATEGIES:
            raise ValueError(
                f"home_strategy must be one of {sorted(_VALID_HOME_STRATEGIES)}, "
                f"got {self.home_strategy!r}"
            )
        reject_credential_fields(type(self))


@dataclass(frozen=True)
class EngineUsageReport:
    """Parsed cost/usage envelope emitted by an engine CLI in JSON mode.

    Subscription-OAuth runs bypass vera's LLM proxy (the only existing
    cost-tracking pipeline), so vera-engine parses the structured output the
    CLIs already emit (`--output-format json`) and surfaces it here for vera
    to persist downstream. All fields are optional: a given CLI may emit any
    subset, and non-JSON output yields a ``None`` report entirely.
    """

    total_cost_usd: float | None = None
    usage: dict[str, int] | None = None
    model_usage: dict[str, dict[str, object]] | None = None


@dataclass(frozen=True)
class RunResult:
    """Result of a local engine run."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    engine: str
    argv: tuple[str, ...]
    session_id: str | None = None
    usage_report: EngineUsageReport | None = None

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.timed_out


# Validate at import time.
reject_credential_fields(EngineInvocation)
reject_credential_fields(EngineUsageReport)
reject_credential_fields(RunResult)
