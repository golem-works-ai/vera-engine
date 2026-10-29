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

### Prompt delivery

Linux rejects any single argv element of 131072 bytes or more (`E2BIG`), so large
prompts never travel as an argv element. Each engine delivers the prompt through
the largest-prompt channel its CLI supports:

| Engine | Channel |
|---|---|
| codex | stdin (`codex exec ... -` reads instructions from stdin) |
| claude-code | stdin (`claude -p` with no positional prompt reads stdin) |
| opencode | argv positional — no stdin or prompt-file channel exists, so prompts of 131072+ UTF-8 bytes fail fast at build time with a clear `ValueError` |
| grok | per-run materialized `--prompt-file` |

### Write access

Headless runs start read-only. Pass `--write` (or `AgentRunRequest(write_access=True)`) when the agent must edit files or commit.

| Engine | Flag added |
|---|---|
| codex | `--sandbox danger-full-access` |
| claude-code | `--permission-mode bypassPermissions` |
| opencode | `--auto` |
| grok | none, since it always passes `--always-approve` |

Codex gets full access, not `workspace-write`. The narrower mode blocks network and blocks commits in a linked git worktree, whose gitdir lies outside the workspace.

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

### Cost & usage reporting

Pair `--structured-output` with `claude-code` or `grok` and the layer also
parses the cost/usage JSON the CLI emits, returning it on
`RunResult.usage_report` as an `EngineUsageReport` (`total_cost_usd`,
`usage`, `model_usage` — any may be `None` when the CLI omitted it). This
matters for subscription-OAuth runs, which bypass the LLM proxy and would
otherwise be invisible to cost dashboards. Non-structured runs, and the
`codex`/`opencode` engines, leave `usage_report` as `None`.

### Model costs

`python -m vera_engine models` shows the same effective costs used by
capacity-aware model selection. It checks local Claude Code, Codex, and Grok
OAuth sessions. A detected subscription uses a `0.1` cost ratio unless
`.vera-engine.toml` sets a provider-specific ratio.

`python -m vera_engine capacity` caches its snapshot locally for 30 minutes.
It reports remaining quota, elapsed-window use, and used/time pressure.
Anthropic remaining comes from unified 5h and 7d rate-limit headers on a
1-token Haiku probe. Run `python -m vera_engine capacity --refresh` to query
the providers immediately.

OpenAI remaining quota comes from the ChatGPT login.
`OPENAI_API_KEY` is the fallback when that login is missing or below 5%.
An available full reset expiring in less than 48 hours adds 100 percentage
points to reported Codex remaining quota (one reset maximum). Displayed token
usage is `max(0, 100 - remaining)`; pressure is zero at 100% or more remaining.
Before a local subscription Codex run starts,
live weekly quota below 5% triggers redemption of that expiring reset and
deletes the capacity cache. Capacity inspection itself never redeems resets.

OpenRouter remaining credits come from `GET /credits` under
`OPENROUTER_BASE_URL`, default `https://openrouter.ai/api/v1`. Set that
variable when OpenRouter traffic goes through a gateway, so the probe sends
the gateway key to the gateway and never to openrouter.ai. A gateway that
does not serve `/credits` still counts the key as usable, with the balance
unknown. A 401 or 403 marks OpenRouter unavailable.

## Adding an Engine

Create a new file in `src/vera_engine/builders/` implementing `EngineBuilder`. Register it in `src/vera_engine/selection.py`. That's it.

## Running Precommit

```bash
python -m pytest tests/
```
