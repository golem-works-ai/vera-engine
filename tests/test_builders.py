"""Tests for per-engine EngineBuilder implementations."""

import json

import pytest

from vera_engine.builders.base import EngineBuilder
from vera_engine.builders.claude_code import DEFAULT_PROMPT_FILENAME, ClaudeCodeBuilder
from vera_engine.builders.codex import CodexBuilder
from vera_engine.builders.grok import DEFAULT_PROMPT_FILENAME as GROK_PROMPT_FILENAME
from vera_engine.builders.grok import GrokBuilder
from vera_engine.builders.opencode import OpenCodeBuilder
from vera_engine.credentials import SecretRef
from vera_engine.invocation import EngineUsageReport
from vera_engine.request import AgentRunRequest


def test_parse_session_id_is_abstract_on_base():
    # A builder that does not implement parse_session_id cannot be instantiated.
    class _Stub(EngineBuilder):
        engine_name = "stub"
        supported_strategies = frozenset({"none"})

        def default_model(self):
            return None

        def build_invocation(self, request, strategy):
            raise NotImplementedError

    with pytest.raises(TypeError):
        _Stub()  # parse_session_id not implemented


@pytest.mark.parametrize(
    "builder",
    [ClaudeCodeBuilder(), OpenCodeBuilder(), CodexBuilder(), GrokBuilder()],
)
def test_all_builders_implement_parse_session_id(builder):
    # Empty / unstructured stdout never yields a session id.
    assert builder.parse_session_id("") is None


def test_parse_usage_report_default_none_on_base():
    # A builder that implements only the abstract methods inherits the
    # parse_usage_report default (returns None), so codex/opencode stay
    # unaffected by the new hook.
    class _Stub(EngineBuilder):
        engine_name = "stub"
        supported_strategies = frozenset({"none"})

        def default_model(self):
            return None

        def build_invocation(self, request, strategy):
            raise NotImplementedError

        def parse_session_id(self, stdout):
            return None

    stub = _Stub()
    assert stub.parse_usage_report('{"total_cost_usd": 1.0}') is None


@pytest.mark.parametrize(
    "builder",
    [OpenCodeBuilder(), CodexBuilder()],
)
def test_parse_usage_report_default_none_on_unsupported_builders(builder):
    # Builders that do not override parse_usage_report must return None
    # even on rich JSON output, so they stay unaffected.
    stdout = '{"total_cost_usd": 0.1, "usage": {"input_tokens": 1}, "modelUsage": {}}'
    assert builder.parse_usage_report(stdout) is None
    assert builder.parse_usage_report("") is None


# --- ClaudeCodeBuilder ---


def test_claude_code_supported_strategies():
    builder = ClaudeCodeBuilder()
    assert builder.supported_strategies == frozenset({"env-key", "proxy", "none"})


def test_claude_code_default_model_is_none():
    assert ClaudeCodeBuilder().default_model() is None


def test_claude_code_env_key_strategy(tmp_path):
    req = AgentRunRequest(
        engine="claude-code", prompt="do the thing", workspace=tmp_path
    )
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
        engine="claude-code",
        prompt="x",
        workspace=tmp_path,
        credential_strategy="proxy",
    )
    inv = ClaudeCodeBuilder().build_invocation(req, "proxy")
    assert inv.env["ANTHROPIC_API_KEY"] == SecretRef("ANTHROPIC_API_KEY")
    assert inv.env["ANTHROPIC_BASE_URL"] == SecretRef("ANTHROPIC_BASE_URL")


def test_claude_code_none_strategy_uses_real_home(tmp_path):
    req = AgentRunRequest(
        engine="claude-code", prompt="x", workspace=tmp_path, credential_strategy="none"
    )
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


def test_claude_code_structured_output_adds_json_flag(tmp_path):
    req = AgentRunRequest(
        engine="claude-code",
        prompt="x",
        workspace=tmp_path,
        structured_output=True,
    )
    inv = ClaudeCodeBuilder().build_invocation(req, "env-key")
    assert "--output-format" in inv.argv
    assert inv.argv[inv.argv.index("--output-format") + 1] == "json"


def test_claude_code_resume_adds_resume_flag(tmp_path):
    req = AgentRunRequest(
        engine="claude-code", prompt="x", workspace=tmp_path, resume="sess-abc"
    )
    inv = ClaudeCodeBuilder().build_invocation(req, "env-key")
    assert "--resume" in inv.argv
    assert inv.argv[inv.argv.index("--resume") + 1] == "sess-abc"
    assert inv.resume == "sess-abc"


def test_claude_code_parse_session_id_top_level_json():
    builder = ClaudeCodeBuilder()
    stdout = '{"session_id": "abc-123", "result": "ok"}'
    assert builder.parse_session_id(stdout) == "abc-123"


def test_claude_code_parse_session_id_none_on_plain_text():
    assert ClaudeCodeBuilder().parse_session_id("just plain text") is None


def test_claude_code_parse_session_id_none_on_missing_field():
    assert ClaudeCodeBuilder().parse_session_id('{"result": "ok"}') is None


# --- ClaudeCodeBuilder.parse_usage_report ---


def _claude_usage_stdout() -> str:
    return json.dumps(
        {
            "session_id": "claude-sess-1",
            "total_cost_usd": 0.0789,
            "usage": {
                "input_tokens": 200,
                "output_tokens": 80,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 15,
                "reasoning_tokens": 40,
            },
            "modelUsage": {
                "claude-sonnet-4": {
                    "cost": 0.0789,
                    "inputTokens": 200,
                    "outputTokens": 80,
                    "provider": "anthropic",
                }
            },
        }
    )


def test_claude_code_parse_usage_report_extracts_fields():
    report = ClaudeCodeBuilder().parse_usage_report(_claude_usage_stdout())
    assert isinstance(report, EngineUsageReport)
    assert report.total_cost_usd == 0.0789
    assert report.usage == {
        "input_tokens": 200,
        "output_tokens": 80,
        "cache_read_input_tokens": 30,
        "cache_creation_input_tokens": 15,
        "reasoning_tokens": 40,
    }
    assert report.model_usage == {
        "claude-sonnet-4": {
            "cost": 0.0789,
            "inputTokens": 200,
            "outputTokens": 80,
            "provider": "anthropic",
        }
    }


def test_claude_code_parse_usage_report_none_on_plain_text():
    assert ClaudeCodeBuilder().parse_usage_report("plain output") is None


def test_claude_code_parse_usage_report_none_on_non_dict_json():
    assert ClaudeCodeBuilder().parse_usage_report('[1, 2, 3]') is None


def test_claude_code_parse_usage_report_none_on_empty():
    assert ClaudeCodeBuilder().parse_usage_report("") is None


def test_claude_code_parse_usage_report_partial_fields():
    report = ClaudeCodeBuilder().parse_usage_report('{"total_cost_usd": 0.42}')
    assert report is not None
    assert report.total_cost_usd == 0.42
    assert report.usage is None
    assert report.model_usage is None


def test_claude_code_parse_usage_report_absent_cost_yields_none():
    report = ClaudeCodeBuilder().parse_usage_report('{"usage": {"input_tokens": 1}}')
    assert report is not None
    assert report.total_cost_usd is None
    assert report.usage == {"input_tokens": 1}


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
    req = AgentRunRequest(
        engine="opencode", prompt="x", workspace=tmp_path, model="custom-model"
    )
    inv = OpenCodeBuilder().build_invocation(req, "env-key")
    assert inv.argv[2] == "custom-model"


def test_opencode_extra_env_merged(tmp_path):
    req = AgentRunRequest(
        engine="opencode", prompt="x", workspace=tmp_path, extra_env={"FOO": "bar"}
    )
    inv = OpenCodeBuilder().build_invocation(req, "env-key")
    assert inv.env["FOO"] == "bar"


def test_opencode_structured_output_adds_format_json(tmp_path):
    req = AgentRunRequest(
        engine="opencode",
        prompt="x",
        workspace=tmp_path,
        structured_output=True,
    )
    inv = OpenCodeBuilder().build_invocation(req, "env-key")
    assert "--format" in inv.argv
    assert inv.argv[inv.argv.index("--format") + 1] == "json"
    # prompt stays last
    assert inv.argv[-1] == "x"


def test_opencode_resume_adds_session_flag(tmp_path):
    req = AgentRunRequest(
        engine="opencode", prompt="x", workspace=tmp_path, resume="sess-op-1"
    )
    inv = OpenCodeBuilder().build_invocation(req, "env-key")
    assert "--session" in inv.argv
    assert inv.argv[inv.argv.index("--session") + 1] == "sess-op-1"
    assert inv.resume == "sess-op-1"


def test_opencode_parse_session_id_ndjson():
    builder = OpenCodeBuilder()
    stdout = '{"event":"start"}\n{"sessionID":"op-sess-9"}\n{"event":"end"}'
    assert builder.parse_session_id(stdout) == "op-sess-9"


def test_opencode_parse_session_id_none_on_plain():
    assert OpenCodeBuilder().parse_session_id("no json here") is None


def test_opencode_parse_session_id_none_when_absent():
    stdout = '{"event":"start"}\n{"event":"end"}'
    assert OpenCodeBuilder().parse_session_id(stdout) is None


# --- CodexBuilder ---


def test_codex_supported_strategies():
    builder = CodexBuilder()
    assert builder.supported_strategies == frozenset({"env-key", "none"})


def test_codex_default_model():
    assert CodexBuilder().default_model() is None


def test_codex_env_key_strategy(tmp_path):
    req = AgentRunRequest(engine="codex", prompt="fix the bug", workspace=tmp_path)
    inv = CodexBuilder().build_invocation(req, "env-key")

    assert inv.argv == (
        "codex",
        "exec",
        "-c",
        "model_reasoning_effort=high",
        "fix the bug",
    )
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
    with pytest.raises(
        ValueError, match="does not support credential_strategy='proxy'"
    ):
        CodexBuilder().build_invocation(req, "proxy")


def test_codex_model_override(tmp_path):
    req = AgentRunRequest(engine="codex", prompt="x", workspace=tmp_path, model="gpt-5")
    inv = CodexBuilder().build_invocation(req, "env-key")
    assert "--model" in inv.argv
    assert inv.argv[inv.argv.index("--model") + 1] == "gpt-5"


def test_codex_prompt_is_last_argv_element(tmp_path):
    req = AgentRunRequest(
        engine="codex", prompt="the actual prompt", workspace=tmp_path
    )
    inv = CodexBuilder().build_invocation(req, "env-key")
    assert inv.argv[-1] == "the actual prompt"


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_codex_emits_effort_flag(tmp_path, effort):
    req = AgentRunRequest(engine="codex", prompt="x", workspace=tmp_path, effort=effort)
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


def test_codex_structured_output_adds_json_flag(tmp_path):
    req = AgentRunRequest(
        engine="codex", prompt="x", workspace=tmp_path, structured_output=True
    )
    inv = CodexBuilder().build_invocation(req, "env-key")
    assert "--json" in inv.argv
    # prompt stays last
    assert inv.argv[-1] == "x"


def test_codex_resume_restructures_argv(tmp_path):
    req = AgentRunRequest(
        engine="codex", prompt="the prompt", workspace=tmp_path, resume="thr-123"
    )
    inv = CodexBuilder().build_invocation(req, "env-key")
    # exec resume <id> ... <prompt>
    assert inv.argv[0:4] == ("codex", "exec", "resume", "thr-123")
    assert inv.argv[-1] == "the prompt"
    assert inv.resume == "thr-123"


def test_codex_resume_with_structured_output(tmp_path):
    req = AgentRunRequest(
        engine="codex",
        prompt="p",
        workspace=tmp_path,
        resume="thr-1",
        structured_output=True,
    )
    inv = CodexBuilder().build_invocation(req, "env-key")
    assert "resume" in inv.argv
    assert inv.argv[inv.argv.index("resume") + 1] == "thr-1"
    assert "--json" in inv.argv


def test_codex_parse_session_id_thread_started():
    builder = CodexBuilder()
    stdout = (
        '{"type":"thread.started","thread_id":"thr-xyz"}\n'
        '{"type":"message","text":"hi"}'
    )
    assert builder.parse_session_id(stdout) == "thr-xyz"


def test_codex_parse_session_id_none_when_no_thread_started():
    stdout = '{"type":"message","text":"hi"}'
    assert CodexBuilder().parse_session_id(stdout) is None


def test_codex_parse_session_id_none_on_garbage():
    assert CodexBuilder().parse_session_id("not json at all") is None


# --- GrokBuilder ---


def test_grok_supported_strategies():
    builder = GrokBuilder()
    assert builder.supported_strategies == frozenset({"env-key", "none"})


def test_grok_default_model():
    assert GrokBuilder().default_model() == "grok-4.6"


def test_grok_env_key_strategy(tmp_path):
    req = AgentRunRequest(engine="grok", prompt="do the thing", workspace=tmp_path)
    inv = GrokBuilder().build_invocation(req, "env-key")

    expected_prompt_path = tmp_path / GROK_PROMPT_FILENAME
    assert inv.argv[0] == "grok"
    assert "--prompt-file" in inv.argv
    assert inv.argv[inv.argv.index("--prompt-file") + 1] == str(expected_prompt_path)
    assert "--output-format" in inv.argv
    assert inv.argv[inv.argv.index("--output-format") + 1] == "plain"
    assert "--always-approve" in inv.argv
    assert "-m" in inv.argv
    assert inv.argv[inv.argv.index("-m") + 1] == "grok-4.6"
    assert inv.env["XAI_API_KEY"] == SecretRef("XAI_API_KEY")
    assert inv.home_strategy == "hermetic"
    assert inv.prompt_path == expected_prompt_path
    assert len(inv.files) == 1
    assert inv.files[0].content == "do the thing"


def test_grok_none_strategy_uses_real_home(tmp_path):
    req = AgentRunRequest(
        engine="grok", prompt="x", workspace=tmp_path, credential_strategy="none"
    )
    inv = GrokBuilder().build_invocation(req, "none")
    assert inv.home_strategy == "real"
    secret_refs = [v for v in inv.env.values() if isinstance(v, SecretRef)]
    assert len(secret_refs) == 0
    assert "XAI_API_KEY" not in inv.env


def test_grok_model_override(tmp_path):
    req = AgentRunRequest(
        engine="grok", prompt="x", workspace=tmp_path, model="grok-4.6"
    )
    inv = GrokBuilder().build_invocation(req, "env-key")
    assert "-m" in inv.argv
    assert inv.argv[inv.argv.index("-m") + 1] == "grok-4.6"


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_grok_emits_reasoning_effort(tmp_path, effort):
    req = AgentRunRequest(engine="grok", prompt="x", workspace=tmp_path, effort=effort)
    inv = GrokBuilder().build_invocation(req, "env-key")
    assert "--reasoning-effort" in inv.argv
    assert inv.argv[inv.argv.index("--reasoning-effort") + 1] == effort


def test_grok_extra_env_merged_without_override(tmp_path):
    req = AgentRunRequest(
        engine="grok",
        prompt="x",
        workspace=tmp_path,
        extra_env={"XAI_API_KEY": "should-not-win", "OTHER": "val"},
    )
    inv = GrokBuilder().build_invocation(req, "env-key")
    assert inv.env["XAI_API_KEY"] == SecretRef("XAI_API_KEY")
    assert inv.env["OTHER"] == "val"


def test_grok_structured_output_uses_json_format(tmp_path):
    req = AgentRunRequest(
        engine="grok", prompt="x", workspace=tmp_path, structured_output=True
    )
    inv = GrokBuilder().build_invocation(req, "env-key")
    assert "--output-format" in inv.argv
    assert inv.argv[inv.argv.index("--output-format") + 1] == "json"


def test_grok_structured_output_false_keeps_plain(tmp_path):
    req = AgentRunRequest(
        engine="grok", prompt="x", workspace=tmp_path, structured_output=False
    )
    inv = GrokBuilder().build_invocation(req, "env-key")
    assert inv.argv[inv.argv.index("--output-format") + 1] == "plain"


def test_grok_resume_adds_resume_flag(tmp_path):
    req = AgentRunRequest(
        engine="grok", prompt="x", workspace=tmp_path, resume="g-sess-1"
    )
    inv = GrokBuilder().build_invocation(req, "env-key")
    assert "--resume" in inv.argv
    assert inv.argv[inv.argv.index("--resume") + 1] == "g-sess-1"
    assert inv.resume == "g-sess-1"


def test_grok_parse_session_id_top_level_json():
    builder = GrokBuilder()
    stdout = '{"sessionId": "g-sess-42", "text": "ok"}'
    assert builder.parse_session_id(stdout) == "g-sess-42"


def test_grok_parse_session_id_none_on_plain():
    assert GrokBuilder().parse_session_id("plain output") is None


def test_grok_parse_session_id_none_when_absent():
    assert GrokBuilder().parse_session_id('{"text": "ok"}') is None


# --- GrokBuilder.parse_usage_report ---


def _grok_usage_stdout() -> str:
    return json.dumps(
        {
            "sessionId": "g-sess-1",
            "total_cost_usd": 0.0123,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 10,
                "cache_creation_input_tokens": 5,
                "reasoning_tokens": 20,
            },
            "modelUsage": {
                "grok-4.6": {
                    "cost": 0.0123,
                    "inputTokens": 100,
                    "outputTokens": 50,
                    "provider": "xai",
                }
            },
        }
    )


def test_grok_parse_usage_report_extracts_fields():
    report = GrokBuilder().parse_usage_report(_grok_usage_stdout())
    assert isinstance(report, EngineUsageReport)
    assert report.total_cost_usd == 0.0123
    assert report.usage == {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 5,
        "reasoning_tokens": 20,
    }
    assert report.model_usage == {
        "grok-4.6": {
            "cost": 0.0123,
            "inputTokens": 100,
            "outputTokens": 50,
            "provider": "xai",
        }
    }


def test_grok_parse_usage_report_none_on_plain_text():
    assert GrokBuilder().parse_usage_report("plain output") is None


def test_grok_parse_usage_report_none_on_non_dict_json():
    assert GrokBuilder().parse_usage_report('[1, 2, 3]') is None


def test_grok_parse_usage_report_none_on_empty():
    assert GrokBuilder().parse_usage_report("") is None


def test_grok_parse_usage_report_partial_fields():
    # Only total_cost_usd present: other fields stay None.
    report = GrokBuilder().parse_usage_report('{"total_cost_usd": 0.42}')
    assert report is not None
    assert report.total_cost_usd == 0.42
    assert report.usage is None
    assert report.model_usage is None


def test_grok_parse_usage_report_absent_cost_yields_none():
    report = GrokBuilder().parse_usage_report('{"usage": {"input_tokens": 1}}')
    assert report is not None
    assert report.total_cost_usd is None
    assert report.usage == {"input_tokens": 1}


def test_grok_parse_usage_report_warns_on_all_fields_absent(caplog):
    # A JSON dict with none of total_cost_usd/usage/modelUsage is exactly what
    # a grok --output-format json schema mismatch would look like: the report
    # still returns (all-None), but a warning surfaces the mismatch instead of
    # failing silently.
    with caplog.at_level("WARNING", logger="vera_engine.builders.base"):
        report = GrokBuilder().parse_usage_report('{"unrelated": true}')
    assert report is not None
    assert report.total_cost_usd is None
    assert report.usage is None
    assert report.model_usage is None
    assert any("grok" in record.getMessage() for record in caplog.records)


def test_claude_code_parse_usage_report_no_warning_when_fields_present(caplog):
    with caplog.at_level("WARNING", logger="vera_engine.builders.base"):
        ClaudeCodeBuilder().parse_usage_report(_claude_usage_stdout())
    assert caplog.records == []
