"""Grok CLI engine builder."""

from __future__ import annotations

from vera_engine.builders.base import EngineBuilder
from vera_engine.credentials import SecretRef
from vera_engine.invocation import EngineInvocation, MaterializedFile
from vera_engine.request import AgentRunRequest

DEFAULT_PROMPT_FILENAME = ".vera-engine-grok-prompt.md"


class GrokBuilder(EngineBuilder):
    @property
    def engine_name(self) -> str:
        return "grok"

    @property
    def supported_strategies(self) -> frozenset[str]:
        return frozenset({"env-key", "none"})

    def default_model(self) -> str | None:
        return "grok-4.6"

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

        argv: list[str] = [
            "grok",
            "--prompt-file",
            str(prompt_path),
            "--output-format",
            "plain",
            "--always-approve",
        ]

        model = request.model or self.default_model()
        if model:
            argv.extend(["-m", model])

        argv.extend(["--reasoning-effort", request.effort])

        env: dict[str, str | SecretRef] = {}

        if strategy == "env-key":
            env["XAI_API_KEY"] = SecretRef("XAI_API_KEY")

        for k, v in request.extra_env.items():
            if k not in env:
                env[k] = v

        home = "hermetic" if strategy == "env-key" else "real"

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
