"""Claude Code engine builder."""

from __future__ import annotations

from vera_engine.builders.base import EngineBuilder
from vera_engine.credentials import SecretRef
from vera_engine.invocation import EngineInvocation, MaterializedFile
from vera_engine.request import AgentRunRequest

_EFFORT_MAP = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}


DEFAULT_PROMPT_FILENAME = ".vera-engine-claude-prompt.md"


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

        model = request.model or self.default_model()
        if model:
            argv.extend(["--model", model])

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
        )
