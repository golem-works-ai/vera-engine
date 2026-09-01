# vera-engine

Engine abstraction layer for AI coding agent CLIs (claude-code, opencode, codex, grok).

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

## Selection

Catalog rows live in `_MODEL_DEFS` in `models.py`. Do not paste the catalog here.

Callers ask for a tier. The engine picks the cheapest qualifier at or above that tier.

`--tier X` is a floor, not a pin.

Every provider with both subscription and API probes subscription first. This holds even when an API key is set. The key is the fallback.

Defaults live in `config.py`:

- `input_weight` is 0.8
- OpenRouter `cost_ratio` is 1.01375

Do not reset these to 0.5 or 1.0.

Do not add a `tenant.default_runtime` concept here. This package does not let a tenant pick the runtime.

### Grok

Do not add a grok-4.5 catalog row.

Native Grok is:

- model `grok-4.6`
- engine `grok`
- provider `xai`

OpenCode `openrouter/x-ai/grok-4.6` is a valid Grok fallback.

Usage probe is tmux plus `/usage`. The weekly bar is used percent. Invert it for remaining. `grok -p /usage` starts an agent. Details live in `capacity.py`. Do not copy the probe algorithm here.

### Claude

`--tier` auto-select must not pick OpenCode Claude (`openrouter/anthropic/…`). `qualifying_models` already drops those rows.

Exception: explicit `--engine opencode --model openrouter/anthropic/…` may still work.

Probe Anthropic remaining via a 1-token Haiku `POST /v1/messages` even when `ANTHROPIC_API_KEY` is set.

Use `CLAUDE_CODE_OAUTH_TOKEN` or `~/.claude/.credentials.json`. Do not send the API key.

Hit `https://api.anthropic.com` directly. Do not follow `ANTHROPIC_BASE_URL`.

Parse `anthropic-ratelimit-unified-{5h,7d}-*` headers. Pressure is remaining / (elapsed * 0.9 + 0.05). Report the window with the larger pressure.
