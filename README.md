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

### Session resume

Pass `--resume <id>` (or `AgentRunRequest(resume=...)`) to continue an existing
conversation context without re-sending the full prompt. Pair it with
`--structured-output` (`structured_output=True`) so the engine emits JSON the
layer can parse a session id out of; that id is returned on `RunResult.session_id`
for the next resume call. All four engines support resume.

```bash
# First run captures a session id
python -m vera_engine run --engine claude-code --structured-output --prompt "start" --workspace .
# Follow-up resumes it
python -m vera_engine run --engine claude-code --resume <session_id> --prompt "next" --workspace .
```

A resume that exits non-zero fails fast (`ResumeFailedError`) rather than
returning empty output — the session id is likely invalid or stale.

### Model costs

`python -m vera_engine models` shows the same effective costs used by
capacity-aware model selection. It checks local Claude Code, Codex, and Grok
OAuth sessions. A detected subscription uses a `0.1` cost ratio unless
`.vera-engine.toml` sets a provider-specific ratio.

`python -m vera_engine capacity` caches its snapshot locally for 30 minutes.
It reports remaining quota, elapsed-window use, and remaining/time pressure.
Anthropic remaining comes from unified 5h and 7d rate-limit headers on a
1-token Haiku probe. Run `python -m vera_engine capacity --refresh` to query
the providers immediately.

## Adding an Engine

Create a new file in `src/vera_engine/builders/` implementing `EngineBuilder`. Register it in `src/vera_engine/selection.py`. That's it.

## Running Precommit

```bash
python -m pytest tests/
```
