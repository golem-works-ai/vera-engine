"""OpenCode engine builder."""

from __future__ import annotations

import json

from vera_engine.builders.base import EngineBuilder
from vera_engine.credentials import SecretRef
from vera_engine.invocation import EngineInvocation, MaterializedFile
from vera_engine.request import AgentRunRequest


class OpenCodeBuilder(EngineBuilder):
    @property
    def engine_name(self) -> str:
        return "opencode"

    @property
    def supported_strategies(self) -> frozenset[str]:
        return frozenset({"env-key", "proxy"})

    def default_model(self) -> str | None:
        return "openrouter/anthropic/claude-sonnet-4"

    def build_invocation(
        self, request: AgentRunRequest, strategy: str
    ) -> EngineInvocation:
        self.validate_strategy(strategy)

        model = request.model or self.default_model()

        # OpenCode reads its prompt from a file via -f.
        prompt_file = MaterializedFile(
            relative_path=".vera-engine-prompt.md",
            content=request.prompt,
            cleanup=True,
        )

        argv: list[str] = ["opencode"]
        if model:
            argv.extend(["-m", model])
        argv.extend(["-f", ".vera-engine-prompt.md"])

        env: dict[str, str | SecretRef] = {}

        if strategy == "env-key":
            env["OPENROUTER_API_KEY"] = SecretRef("OPENROUTER_API_KEY")
        elif strategy == "proxy":
            env["OPENROUTER_API_KEY"] = SecretRef("OPENROUTER_API_KEY")

        # Materialize opencode.json config if using a proxy.
        files: list[MaterializedFile] = [prompt_file]
        if strategy == "proxy":
            config = {"provider": {"base_url": "$ANTHROPIC_BASE_URL"}}
            files.append(
                MaterializedFile(
                    relative_path="opencode.json",
                    content=json.dumps(config, indent=2),
                    cleanup=True,
                )
            )
            env["ANTHROPIC_BASE_URL"] = SecretRef("ANTHROPIC_BASE_URL")

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
            files=tuple(files),
        )
