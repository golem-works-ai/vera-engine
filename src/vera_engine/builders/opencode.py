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

        argv: list[str] = ["opencode"]
        if model:
            argv.extend(["-m", model])
        argv.append("run")
        if request.resume:
            argv.extend(["--session", request.resume])
        if request.structured_output:
            argv.extend(["--format", "json"])
        argv.append(request.prompt)

        env: dict[str, str | SecretRef] = {}

        if strategy == "env-key" or strategy == "proxy":
            env["OPENROUTER_API_KEY"] = SecretRef("OPENROUTER_API_KEY")

        files: list[MaterializedFile] = []
        if strategy == "proxy":
            config = {"provider": {"base_url": "$ANTHROPIC_BASE_URL"}}
            # Stored body holds $ANTHROPIC_BASE_URL; default is False so literal $ in prompts stay intact.
            files.append(
                MaterializedFile(
                    relative_path="opencode.json",
                    content=json.dumps(config, indent=2),
                    expand_references=True,
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
            resume=request.resume,
        )

    def parse_session_id(self, stdout: str) -> str | None:
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            sid = data.get("sessionID")
            if isinstance(sid, str) and sid:
                return sid
        return None
