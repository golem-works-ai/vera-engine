"""vera-engine: Engine abstraction layer for AI coding agent CLIs."""

from vera_engine.auto_select import Selection, select_model
from vera_engine.config import EngineConfig, load_config
from vera_engine.request import AgentRunRequest
from vera_engine.credentials import SecretRef, CredentialBundle
from vera_engine.invocation import EngineInvocation, MaterializedFile, RunResult
from vera_engine.models import CATALOG, ModelSpec, PROVIDER_CREDENTIALS, Tier, TIER_NAMES, get_catalog
from vera_engine.selection import get_builder, list_engines
from vera_engine.render.local import render_local

__all__ = [
    "AgentRunRequest",
    "CATALOG",
    "CredentialBundle",
    "EngineConfig",
    "EngineInvocation",
    "MaterializedFile",
    "ModelSpec",
    "PROVIDER_CREDENTIALS",
    "RunResult",
    "SecretRef",
    "Tier",
    "TIER_NAMES",
    "build_and_run",
    "get_builder",
    "list_engines",
    "load_config",
    "render_local",
    "select_model",
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
