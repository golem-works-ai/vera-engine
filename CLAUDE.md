# vera-engine

Engine abstraction layer for AI coding agent CLIs (claude-code, opencode, codex).

## Core Rules

- **Zero dependencies.** stdlib only. No third-party imports.
- **Fail fast.** Missing credentials, unknown engines, invalid combinations all raise immediately. No silent fallbacks.
- **Credentials by reference only.** `SecretRef` carries a name, never a value. `CredentialBundle` exists only at spawn time.
- **Engine-agnostic core.** `request.py`, `invocation.py`, `credentials.py` must never import from `builders/`.

## Architecture

- `request.py` — `AgentRunRequest`: declarative "what to run"
- `credentials.py` — `SecretRef`, `CredentialBundle`, structural guards
- `invocation.py` — `EngineInvocation`: resolved "how to run it"
- `selection.py` — builder registry, engine+strategy validation
- `builders/` — per-engine argv/env construction (one file per engine)
- `render/local.py` — subprocess spawner with hermetic env
- `cli.py` — `python -m vera_engine run` entry point

## Conventions

- Task runner: `python -m pytest tests/` for tests
- Package layout: `src/vera_engine/` (src-layout)
- Python 3.11+, type hints on all public functions
