# vera-engine

Engine abstraction layer for spawning AI coding agent CLIs as subprocesses.

Supports claude-code, opencode, codex, grok, and any CLI agent that accepts a prompt and runs in a workspace directory.

## Design Principles

- **Declarative request, engine-specific subprocess.** Callers describe WHAT to run; the layer resolves HOW.
- **Credential isolation is first-class.** Secrets travel as named references (`SecretRef`), never as values on request objects. A hermetic HOME prevents engines from reading developer credentials.
- **Zero external dependencies.** stdlib only. Copy into any repo.
- **Engine-agnostic core.** Request, invocation, and credential types know nothing about specific engines. Each engine gets a builder plugin.
- **Fail fast, no silent fallbacks.** Missing credentials, unknown engines, and invalid combinations all raise immediately.

## Usage

```bash
# Run via CLI
python -m vera_engine run \
  --engine claude-code \
  --model claude-sonnet-4-5-20250929 \
  --prompt "fix the bug" \
  --workspace .

# Or as a library
from vera_engine import AgentRunRequest, build_and_run
from pathlib import Path

request = AgentRunRequest(
    engine="claude-code",
    prompt="fix the bug",
    workspace=Path("."),
)
result = build_and_run(request, credentials={"ANTHROPIC_API_KEY": "..."})
```

## Adding an Engine

Create a new file in `src/vera_engine/builders/` implementing `EngineBuilder`. Register it in `src/vera_engine/selection.py`. That's it.

## Running Precommit

```bash
python -m pytest tests/
```
