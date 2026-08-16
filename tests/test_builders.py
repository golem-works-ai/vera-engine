"""Tests for per-engine EngineBuilder implementations."""

import json

import pytest

from vera_engine.builders.claude_code import ClaudeCodeBuilder, DEFAULT_PROMPT_FILENAME
from vera_engine.builders.codex import CodexBuilder
from vera_engine.builders.opencode import OpenCodeBuilder
from vera_engine.credentials import SecretRef
from vera_engine.request import AgentRunRequest


# --- ClaudeCodeBuilder ---


def test_claude_code_supported_strategies():
    builder = ClaudeCodeBuilder()
    assert builder.supported_strategies == frozenset({"env-key", "proxy", "none"})


def test_claude_code_default_model_is_none():
    assert ClaudeCodeBuilder().default_model() is None


def test_claude_code_env_key_strategy(tmp_path):
    req = AgentRunRequest(engine="claude-code", prompt="do the thing", workspace=tmp_path)
    inv = ClaudeCodeBuilder().build_invocation(req, "env-key")

    expected_prompt_path = tmp_path / DEFAULT_PROMPT_FILENAME
    assert inv.argv == ("claude", "-p", str(expected_prompt_path), "--verbose")
    assert inv.env["ANTHROPIC_API_KEY"] == SecretRef("ANTHROPIC_API_KEY")
    assert "ANTHROPIC_BASE_URL" not in inv.env
    assert inv.env["CLAUDE_CODE_EFFORT_LEVEL"] == "high"
    assert inv.workdir == tmp_path
    assert inv.home_strategy == "hermetic"
    assert inv.prompt_path == expected_prompt_path
    assert len(inv.files) == 1
    assert inv.files[0].relative_path == DEFAULT_PROMPT_FILENAME
    assert inv.files[0].content == "do the thing"


def test_claude_code_proxy_strategy(tmp_path):
    req = AgentRunRequest(
        engine="claude-code", prompt="x", workspace=tmp_path, credential_strategy="proxy"
    )
    inv = ClaudeCodeBuilder().build_invocation(req, "proxy")
    assert inv.env["ANTHROPIC_API_KEY"] == SecretRef("ANTHROPIC_API_KEY")
    assert inv.env["ANTHROPIC_BASE_URL"] == SecretRef("ANTHROPIC_BASE_URL")


def test_claude_code_none_strategy_uses_real_home(tmp_path):
    req = AgentRunRequest(engine="claude-code", prompt="x", workspace=tmp_path, credential_strategy="none")
    inv = ClaudeCodeBuilder().build_invocation(req, "none")
    assert inv.home_strategy == "real"
    secret_refs = [v for v in inv.env.values() if isinstance(v, SecretRef)]
    assert len(secret_refs) == 0


def test_claude_code_model_override(tmp_path):
    req = AgentRunRequest(
        engine="claude-code", prompt="x", workspace=tmp_path, model="claude-opus"
    )
    inv = ClaudeCodeBuilder().build_invocation(req, "env-key")
    assert "--model" in inv.argv
    assert inv.argv[inv.argv.index("--model") + 1] == "claude-opus"


def test_claude_code_extra_env_merged_without_override(tmp_path):
    req = AgentRunRequest(
        engine="claude-code",
        prompt="x",
        workspace=tmp_path,
        extra_env={"FOO": "bar", "CLAUDE_CODE_EFFORT_LEVEL": "should-not-win"},
    )
    inv = ClaudeCodeBuilder().build_invocation(req, "env-key")
    assert inv.env["FOO"] == "bar"
    # Builder-set values take priority; extra_env only fills gaps.
    assert inv.env["CLAUDE_CODE_EFFORT_LEVEL"] == "high"


# --- OpenCodeBuilder ---


def test_opencode_supported_strategies():
    builder = OpenCodeBuilder()
    assert builder.supported_strategies == frozenset({"env-key", "proxy"})


def test_opencode_default_model():
    assert OpenCodeBuilder().default_model() == "openrouter/anthropic/claude-sonnet-4"


def test_opencode_env_key_strategy(tmp_path):
    req = AgentRunRequest(engine="opencode", prompt="do stuff", workspace=tmp_path)
    inv = OpenCodeBuilder().build_invocation(req, "env-key")

    assert inv.argv == (
        "opencode",
        "-m",
        "openrouter/anthropic/claude-sonnet-4",
        "run",
        "do stuff",
    )
    assert inv.env["OPENROUTER_API_KEY"] == SecretRef("OPENROUTER_API_KEY")
    assert "ANTHROPIC_BASE_URL" not in inv.env
    assert len(inv.files) == 0


def test_opencode_proxy_strategy_adds_config_file(tmp_path):
    req = AgentRunRequest(
        engine="opencode", prompt="x", workspace=tmp_path, credential_strategy="proxy"
    )
    inv = OpenCodeBuilder().build_invocation(req, "proxy")

    assert inv.env["OPENROUTER_API_KEY"] == SecretRef("OPENROUTER_API_KEY")
    assert inv.env["ANTHROPIC_BASE_URL"] == SecretRef("ANTHROPIC_BASE_URL")
    assert len(inv.files) == 1
    config_file = inv.files[0]
    config = json.loads(config_file.content)
    assert config == {"provider": {"base_url": "$ANTHROPIC_BASE_URL"}}
    assert config_file.expand_references is True


def test_opencode_unsupported_strategy_raises(tmp_path):
    req = AgentRunRequest(engine="opencode", prompt="x", workspace=tmp_path)
    with pytest.raises(ValueError, match="does not support credential_strategy='none'"):
        OpenCodeBuilder().build_invocation(req, "none")


def test_opencode_model_override(tmp_path):
    req = AgentRunRequest(engine="opencode", prompt="x", workspace=tmp_path, model="custom-model")
    inv = OpenCodeBuilder().build_invocation(req, "env-key")
    assert inv.argv[2] == "custom-model"


def test_opencode_extra_env_merged(tmp_path):
    req = AgentRunRequest(
        engine="opencode", prompt="x", workspace=tmp_path, extra_env={"FOO": "bar"}
    )
    inv = OpenCodeBuilder().build_invocation(req, "env-key")
    assert inv.env["FOO"] == "bar"


# --- CodexBuilder ---


def test_codex_supported_strategies():
    builder = CodexBuilder()
    assert builder.supported_strategies == frozenset({"env-key", "none"})


def test_codex_default_model():
    assert CodexBuilder().default_model() is None


def test_codex_env_key_strategy(tmp_path):
    req = AgentRunRequest(engine="codex", prompt="fix the bug", workspace=tmp_path)
    inv = CodexBuilder().build_invocation(req, "env-key")

    assert inv.argv == ("codex", "exec", "-c", "model_reasoning_effort=high", "fix the bug")
    assert inv.env["OPENAI_API_KEY"] == SecretRef("OPENAI_API_KEY")
    assert inv.files == ()
    assert inv.prompt_path is None


def test_codex_none_strategy_has_no_credentials(tmp_path):
    req = AgentRunRequest(
        engine="codex", prompt="x", workspace=tmp_path, credential_strategy="none"
    )
    inv = CodexBuilder().build_invocation(req, "none")
    assert "OPENAI_API_KEY" not in inv.env
    assert inv.env == {}


def test_codex_unsupported_strategy_raises(tmp_path):
    req = AgentRunRequest(engine="codex", prompt="x", workspace=tmp_path)
    with pytest.raises(ValueError, match="does not support credential_strategy='proxy'"):
        CodexBuilder().build_invocation(req, "proxy")


def test_codex_model_override(tmp_path):
    req = AgentRunRequest(engine="codex", prompt="x", workspace=tmp_path, model="gpt-5")
    inv = CodexBuilder().build_invocation(req, "env-key")
    assert "--model" in inv.argv
    assert inv.argv[inv.argv.index("--model") + 1] == "gpt-5"


def test_codex_prompt_is_last_argv_element(tmp_path):
    req = AgentRunRequest(engine="codex", prompt="the actual prompt", workspace=tmp_path)
    inv = CodexBuilder().build_invocation(req, "env-key")
    assert inv.argv[-1] == "the actual prompt"


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_codex_emits_effort_flag(tmp_path, effort):
    req = AgentRunRequest(
        engine="codex", prompt="x", workspace=tmp_path, effort=effort
    )
    inv = CodexBuilder().build_invocation(req, "env-key")
    assert "-c" in inv.argv
    assert inv.argv[inv.argv.index("-c") + 1] == f"model_reasoning_effort={effort}"


def test_codex_extra_env_merged_without_override(tmp_path):
    req = AgentRunRequest(
        engine="codex",
        prompt="x",
        workspace=tmp_path,
        extra_env={"OPENAI_API_KEY": "should-not-win", "OTHER": "val"},
    )
    inv = CodexBuilder().build_invocation(req, "env-key")
    assert inv.env["OPENAI_API_KEY"] == SecretRef("OPENAI_API_KEY")
    assert inv.env["OTHER"] == "val"
