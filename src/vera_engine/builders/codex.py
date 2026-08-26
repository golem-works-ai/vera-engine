"""Codex CLI engine builder."""

from __future__ import annotations

import json

from vera_engine.builders.base import EngineBuilder
from vera_engine.credentials import SecretRef
from vera_engine.invocation import EngineInvocation
from vera_engine.request import AgentRunRequest


class CodexBuilder(EngineBuilder):
    @property
    def engine_name(self) -> str:
        return "codex"

    @property
    def supported_strategies(self) -> frozenset[str]:
        return frozenset({"env-key", "none"})

    def default_model(self) -> str | None:
        return None

    def build_invocation(
        self, request: AgentRunRequest, strategy: str
    ) -> EngineInvocation:
        self.validate_strategy(strategy)

        model = request.model or self.default_model()

        argv: list[str] = ["codex", "exec"]
        if request.resume:
            argv.extend(["resume", request.resume])
        if request.structured_output:
            argv.append("--json")
        if model:
            argv.extend(["--model", model])
        # codex exec reads effort from -c model_reasoning_effort, not from a --effort flag.
        argv.extend(["-c", f"model_reasoning_effort={request.effort}"])
        argv.append(request.prompt)

        env: dict[str, str | SecretRef] = {}

        if strategy == "env-key":
            env["OPENAI_API_KEY"] = SecretRef("OPENAI_API_KEY")

        for k, v in request.extra_env.items():
            if k not in env:
                env[k] = v

        return EngineInvocation(
            engine=self.engine_name,
            argv=tuple(argv),
            env=env,
            workdir=request.workspace,
            home_strategy="real",
            timeout_seconds=request.timeout_seconds,
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
            if data.get("type") != "thread.started":
                continue
            tid = data.get("thread_id")
            if isinstance(tid, str) and tid:
                return tid
        return None
