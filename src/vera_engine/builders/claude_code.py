"""Claude Code engine builder."""

from __future__ import annotations

from vera_engine.builders.base import EngineBuilder
from vera_engine.credentials import SecretRef
from vera_engine.invocation import EngineInvocation
from vera_engine.request import AgentRunRequest

_EFFORT_MAP = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}


class ClaudeCodeBuilder(EngineBuilder):
    @property
    def engine_name(self) -> str:
        return "claude-code"

    @property
    def supported_strategies(self) -> frozenset[str]:
        return frozenset({"env-key", "proxy"})

    def default_model(self) -> str | None:
        return None  # Claude Code picks its own default.

    def build_invocation(
        self, request: AgentRunRequest, strategy: str
    ) -> EngineInvocation:
        self.validate_strategy(strategy)

        argv: list[str] = ["claude", "-p", request.prompt, "--verbose"]

        model = request.model or self.default_model()
        if model:
            argv.extend(["--model", model])

        env: dict[str, str | SecretRef] = {}

        if strategy == "env-key":
            env["ANTHROPIC_API_KEY"] = SecretRef("ANTHROPIC_API_KEY")
        elif strategy == "proxy":
            env["ANTHROPIC_API_KEY"] = SecretRef("ANTHROPIC_API_KEY")
            env["ANTHROPIC_BASE_URL"] = SecretRef("ANTHROPIC_BASE_URL")

        env["CLAUDE_CODE_EFFORT_LEVEL"] = _EFFORT_MAP[request.effort]

        # Merge extra env (plain strings only; no SecretRef override).
        for k, v in request.extra_env.items():
            if k not in env:
                env[k] = v

        return EngineInvocation(
            engine=self.engine_name,
            argv=tuple(argv),
            env=env,
            workdir=request.workspace,
            home_strategy="hermetic",
            timeout_seconds=request.timeout_seconds,
        )
