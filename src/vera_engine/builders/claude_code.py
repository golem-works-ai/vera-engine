"""Claude Code engine builder."""

from __future__ import annotations

import json

from vera_engine.builders.base import EngineBuilder
from vera_engine.credentials import SecretRef
from vera_engine.invocation import EngineInvocation, EngineUsageReport, MaterializedFile
from vera_engine.request import AgentRunRequest

_EFFORT_MAP = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}


DEFAULT_PROMPT_FILENAME = ".vera-engine-claude-prompt.md"


def _parse_claude_usage_report(stdout: str) -> EngineUsageReport | None:
    """Parse the cost/usage envelope emitted by ``claude --output-format json``.

    Field names follow the Claude Code streaming-JSON envelope schema as
    documented by Anthropic's ``claude`` CLI: the top-level object carries
    ``total_cost_usd`` (float), ``usage`` (per-run token counts), and
    ``modelUsage`` (per-model cost/token breakdown). ``session_id`` is read
    separately by :meth:`parse_session_id`.

    Returns None on non-JSON or non-dict output; any subset of fields may be
    present, with absent fields left as None. ``usage``/``modelUsage`` are
    only stored when they are themselves dicts, so a malformed envelope cannot
    smuggle a non-mapping through.
    """
    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    raw_usage = data.get("usage")
    raw_model_usage = data.get("modelUsage")

    return EngineUsageReport(
        total_cost_usd=data.get("total_cost_usd"),
        usage=raw_usage if isinstance(raw_usage, dict) else None,
        model_usage=raw_model_usage if isinstance(raw_model_usage, dict) else None,
    )


class ClaudeCodeBuilder(EngineBuilder):
    @property
    def engine_name(self) -> str:
        return "claude-code"

    @property
    def supported_strategies(self) -> frozenset[str]:
        return frozenset({"env-key", "proxy", "none"})

    def default_model(self) -> str | None:
        return None

    def build_invocation(
        self, request: AgentRunRequest, strategy: str
    ) -> EngineInvocation:
        self.validate_strategy(strategy)

        prompt_file = MaterializedFile(
            relative_path=DEFAULT_PROMPT_FILENAME,
            content=request.prompt,
            cleanup=True,
        )
        prompt_path = request.workspace / DEFAULT_PROMPT_FILENAME

        argv: list[str] = ["claude", "-p", str(prompt_path), "--verbose"]

        if request.structured_output:
            argv.extend(["--output-format", "json"])

        model = request.model or self.default_model()
        if model:
            argv.extend(["--model", model])

        if request.resume:
            argv.extend(["--resume", request.resume])

        env: dict[str, str | SecretRef] = {}

        if strategy == "env-key":
            env["ANTHROPIC_API_KEY"] = SecretRef("ANTHROPIC_API_KEY")
        elif strategy == "proxy":
            env["ANTHROPIC_API_KEY"] = SecretRef("ANTHROPIC_API_KEY")
            env["ANTHROPIC_BASE_URL"] = SecretRef("ANTHROPIC_BASE_URL")
        # "none" strategy: no credentials injected, uses local OAuth session.

        env["CLAUDE_CODE_EFFORT_LEVEL"] = _EFFORT_MAP[request.effort]

        for k, v in request.extra_env.items():
            if k not in env:
                env[k] = v

        # OAuth session lives in ~/.claude, so we need the real HOME.
        home = "hermetic" if strategy in ("env-key", "proxy") else "real"

        return EngineInvocation(
            engine=self.engine_name,
            argv=tuple(argv),
            env=env,
            workdir=request.workspace,
            home_strategy=home,
            timeout_seconds=request.timeout_seconds,
            files=(prompt_file,),
            prompt_path=prompt_path,
            resume=request.resume,
        )

    def parse_session_id(self, stdout: str) -> str | None:
        try:
            data = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        sid = data.get("session_id")
        if isinstance(sid, str) and sid:
            return sid
        return None

    def parse_usage_report(self, stdout: str) -> EngineUsageReport | None:
        return _parse_claude_usage_report(stdout)
