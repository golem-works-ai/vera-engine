"""Abstract builder interface for engine plugins."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from vera_engine.invocation import EngineInvocation, EngineUsageReport
from vera_engine.request import AgentRunRequest

logger = logging.getLogger(__name__)


def parse_usage_report_json(
    stdout: str, *, engine: str = "unknown"
) -> EngineUsageReport | None:
    """Parse the shared cost/usage envelope emitted by CLIs in JSON mode.

    The ``claude`` (Claude Code) CLI's ``--output-format json`` envelope is
    confirmed to expose ``total_cost_usd``, ``usage``, and ``modelUsage``.
    The ``grok`` (xAI) CLI is *assumed* to share this schema, but that
    assumption is unverified against real ``grok --output-format json``
    output -- see PR #15 discussion. The parsing lives here once so both
    builders delegate to it and avoid copy drift, but callers should not
    treat a clean parse from grok as schema-confirmed. The session-id key
    differs in casing between the two CLIs (``session_id`` vs
    ``sessionId``) and is read separately by each builder's
    :meth:`parse_session_id`.

    Returns None on non-JSON or non-dict output; any subset of fields may be
    present, with absent fields left as None. ``usage``/``modelUsage`` are
    only stored when they are themselves dicts, so a malformed envelope cannot
    smuggle a non-mapping through. If the envelope parses as a dict but none
    of the tracked fields are present, this is logged as a warning -- for an
    engine whose schema isn't confirmed (i.e. grok), a run of all-None fields
    is indistinguishable from a silent field-name mismatch, so the warning is
    the defensive fallback until real captured output confirms the schema.
    """
    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    raw_usage = data.get("usage")
    raw_model_usage = data.get("modelUsage")
    total_cost_usd = data.get("total_cost_usd")

    if (
        total_cost_usd is None
        and not isinstance(raw_usage, dict)
        and not isinstance(raw_model_usage, dict)
    ):
        logger.warning(
            "%s --output-format json produced no recognized cost/usage fields "
            "(total_cost_usd/usage/modelUsage all absent); the shared schema "
            "assumption may not hold for this engine",
            engine,
        )

    return EngineUsageReport(
        total_cost_usd=total_cost_usd,
        usage=raw_usage if isinstance(raw_usage, dict) else None,
        model_usage=raw_model_usage if isinstance(raw_model_usage, dict) else None,
    )


class EngineBuilder(ABC):
    """Translates an AgentRunRequest into an EngineInvocation.

    Each engine (claude-code, opencode, codex, grok, etc.) implements one builder.
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

    @abstractmethod
    def parse_session_id(self, stdout: str) -> str | None:
        """Extract a session id from engine stdout.

        Returns the session id when the engine emitted structured output
        carrying one, or None when no session id could be found (e.g. the
        run did not request structured output, or the output was not JSON).
        """
        ...

    def parse_usage_report(self, stdout: str) -> EngineUsageReport | None:
        """Extract a cost/usage report from engine stdout.

        Returns the parsed :class:`EngineUsageReport` when the engine emitted
        JSON carrying cost/usage fields, or None when no report could be
        extracted (e.g. the run did not request structured output, the output
        was not JSON, or the engine does not emit cost data). The base default
        returns None so engines like codex/opencode that do not parse cost
        data stay unaffected.
        """
        return None

    def validate_strategy(self, strategy: str) -> None:
        """Raise if the strategy is not supported by this engine."""
        if strategy not in self.supported_strategies:
            raise ValueError(
                f"engine {self.engine_name!r} does not support "
                f"credential_strategy={strategy!r} "
                f"(supported: {sorted(self.supported_strategies)})"
            )
