"""vera-engine: Engine abstraction layer for AI coding agent CLIs."""

from vera_engine.auto_select import (
    Selection,
    qualifying_models,
    resolve_request,
    select_cheapest,
    select_model,
)
from vera_engine.config import EngineConfig, load_config
from vera_engine.credentials import (
    FORBIDDEN_FIELD_PATTERN,
    CredentialBundle,
    CredentialGuardError,
    SecretRef,
    assert_no_credential_shaped_fields,
)
from vera_engine.invocation import EngineInvocation, MaterializedFile, RunResult
from vera_engine.models import (
    CATALOG,
    PROVIDER_CREDENTIALS,
    TIER_NAMES,
    ModelSpec,
    Tier,
    get_catalog,
)
from vera_engine.render.local import render_local
from vera_engine.request import AgentRunRequest
from vera_engine.selection import get_builder, list_engines

__all__ = [
    "CATALOG",
    "FORBIDDEN_FIELD_PATTERN",
    "PROVIDER_CREDENTIALS",
    "TIER_NAMES",
    "AgentRunRequest",
    "CredentialBundle",
    "CredentialGuardError",
    "EngineConfig",
    "EngineInvocation",
    "MaterializedFile",
    "ModelSpec",
    "RunResult",
    "SecretRef",
    "Tier",
    "assert_no_credential_shaped_fields",
    "build_and_run",
    "get_builder",
    "list_engines",
    "load_config",
    "qualifying_models",
    "render_local",
    "resolve_request",
    "select_cheapest",
    "select_model",
]


def build_and_run(
    request: AgentRunRequest,
    credentials: dict[str, str] | None = None,
    *,
    config: EngineConfig | None = None,
    probe: bool = False,
    engines: frozenset[str] | set[str] | None = None,
) -> RunResult:
    """Build an invocation from a request and run it locally.

    This is the main convenience entry point. For more control,
    use get_builder() and render_local() separately.
    """
    cfg = config if config is not None else load_config(request.workspace)
    resolved = resolve_request(request, cfg, probe=probe, engines=engines)
    builder = get_builder(resolved.engine)
    invocation = builder.build_invocation(resolved, resolved.credential_strategy)
    bundle = CredentialBundle(values=credentials or {})
    return render_local(invocation, bundle, builder=builder)
