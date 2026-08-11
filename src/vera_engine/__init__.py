"""vera-engine: Engine abstraction layer for AI coding agent CLIs."""

from vera_engine.request import AgentRunRequest
from vera_engine.credentials import SecretRef, CredentialBundle
from vera_engine.invocation import EngineInvocation, MaterializedFile, RunResult
from vera_engine.selection import get_builder, list_engines
from vera_engine.render.local import render_local

__all__ = [
    "AgentRunRequest",
    "SecretRef",
    "CredentialBundle",
    "EngineInvocation",
    "MaterializedFile",
    "RunResult",
    "get_builder",
    "list_engines",
    "render_local",
    "build_and_run",
]


def build_and_run(
    request: AgentRunRequest,
    credentials: dict[str, str] | None = None,
) -> RunResult:
    """Build an invocation from a request and run it locally.

    This is the main convenience entry point. For more control,
    use get_builder() and render_local() separately.
    """
    builder = get_builder(request.engine)
    invocation = builder.build_invocation(request, request.credential_strategy)
    bundle = CredentialBundle(values=credentials or {})
    return render_local(invocation, bundle)
