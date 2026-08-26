"""Tests for EngineInvocation, MaterializedFile, and RunResult."""

from pathlib import Path

import pytest

from vera_engine.credentials import SecretRef
from vera_engine.invocation import (
    EngineInvocation,
    EngineUsageReport,
    MaterializedFile,
    RunResult,
)


def test_materialized_file_defaults():
    mf = MaterializedFile(relative_path="prompt.md", content="hi")
    assert mf.relative_path == "prompt.md"
    assert mf.content == "hi"
    assert mf.mode == 0o644
    assert mf.expand_references is False
    assert mf.cleanup is True


def test_materialized_file_rejects_absolute_path():
    with pytest.raises(ValueError):
        MaterializedFile(relative_path="/tmp/x", content="hi")


def test_materialized_file_rejects_parent_segment():
    with pytest.raises(ValueError):
        MaterializedFile(relative_path="../secret", content="hi")


def test_materialized_file_cleanup_false():
    mf = MaterializedFile(relative_path="cfg.json", content="{}", cleanup=False)
    assert mf.cleanup is False


def test_engine_invocation_valid_construction(tmp_path):
    inv = EngineInvocation(
        engine="claude-code",
        argv=("claude", "-p", "hi"),
        env={"KEY": "value", "SECRET": SecretRef("SECRET_NAME")},
        workdir=tmp_path,
    )
    assert inv.engine == "claude-code"
    assert inv.argv == ("claude", "-p", "hi")
    assert inv.env["KEY"] == "value"
    assert isinstance(inv.env["SECRET"], SecretRef)
    assert inv.workdir == tmp_path
    assert inv.home_strategy == "hermetic"
    assert inv.timeout_seconds == 2400
    assert inv.files == ()
    assert inv.prompt_path is None


@pytest.mark.parametrize("strategy", ["hermetic", "shared", "real"])
def test_engine_invocation_valid_home_strategies(tmp_path, strategy):
    inv = EngineInvocation(
        engine="codex", argv=("codex",), env={}, workdir=tmp_path, home_strategy=strategy
    )
    assert inv.home_strategy == strategy


def test_engine_invocation_invalid_home_strategy_raises(tmp_path):
    with pytest.raises(ValueError, match="home_strategy must be one of"):
        EngineInvocation(
            engine="codex", argv=("codex",), env={}, workdir=tmp_path, home_strategy="magic"
        )


def test_engine_invocation_empty_argv_raises(tmp_path):
    with pytest.raises(ValueError, match="argv must not be empty"):
        EngineInvocation(engine="codex", argv=(), env={}, workdir=tmp_path)


def test_engine_invocation_with_files_and_prompt_path(tmp_path):
    mf = MaterializedFile(relative_path="p.md", content="hello")
    prompt_path = tmp_path / "p.md"
    inv = EngineInvocation(
        engine="claude-code",
        argv=("claude",),
        env={},
        workdir=tmp_path,
        files=(mf,),
        prompt_path=prompt_path,
    )
    assert inv.files == (mf,)
    assert inv.prompt_path == prompt_path


def test_engine_invocation_is_frozen(tmp_path):
    inv = EngineInvocation(engine="codex", argv=("codex",), env={}, workdir=tmp_path)
    with pytest.raises(Exception):
        inv.engine = "other"


def test_run_result_succeeded_true_on_zero_returncode():
    result = RunResult(
        returncode=0, stdout="ok", stderr="", timed_out=False, engine="codex", argv=("codex",)
    )
    assert result.succeeded is True


def test_run_result_succeeded_false_on_nonzero_returncode():
    result = RunResult(
        returncode=1, stdout="", stderr="err", timed_out=False, engine="codex", argv=("codex",)
    )
    assert result.succeeded is False


def test_run_result_succeeded_false_on_timeout_even_with_zero_returncode():
    result = RunResult(
        returncode=0, stdout="", stderr="", timed_out=True, engine="codex", argv=("codex",)
    )
    assert result.succeeded is False


def test_run_result_is_frozen():
    result = RunResult(
        returncode=0, stdout="", stderr="", timed_out=False, engine="codex", argv=("codex",)
    )
    with pytest.raises(Exception):
        result.returncode = 1


def test_run_result_session_id_defaults_to_none():
    result = RunResult(
        returncode=0, stdout="", stderr="", timed_out=False, engine="codex", argv=("codex",)
    )
    assert result.session_id is None


def test_run_result_session_id_stored():
    result = RunResult(
        returncode=0,
        stdout="",
        stderr="",
        timed_out=False,
        engine="codex",
        argv=("codex",),
        session_id="sess-xyz",
    )
    assert result.session_id == "sess-xyz"


def test_engine_invocation_resume_defaults_to_none(tmp_path):
    inv = EngineInvocation(
        engine="codex", argv=("codex",), env={}, workdir=tmp_path
    )
    assert inv.resume is None


def test_engine_invocation_resume_stored(tmp_path):
    inv = EngineInvocation(
        engine="codex",
        argv=("codex",),
        env={},
        workdir=tmp_path,
        resume="abc-123",
    )
    assert inv.resume == "abc-123"


# --- EngineUsageReport ---


def test_engine_usage_report_defaults_to_none():
    report = EngineUsageReport()
    assert report.total_cost_usd is None
    assert report.usage is None
    assert report.model_usage is None


def test_engine_usage_report_stores_fields():
    report = EngineUsageReport(
        total_cost_usd=0.0123,
        usage={"input_tokens": 10, "output_tokens": 5},
        model_usage={"grok-4.6": {"cost": 0.0123, "tokens": 15}},
    )
    assert report.total_cost_usd == 0.0123
    assert report.usage == {"input_tokens": 10, "output_tokens": 5}
    assert report.model_usage == {"grok-4.6": {"cost": 0.0123, "tokens": 15}}


def test_engine_usage_report_is_frozen():
    report = EngineUsageReport(total_cost_usd=1.0)
    with pytest.raises(Exception):
        report.total_cost_usd = 2.0


def test_engine_usage_report_passes_credential_guard():
    # Construction at import time would have raised via reject_credential_fields;
    # building an instance confirms the guard accepted the dataclass shape.
    assert EngineUsageReport() is not None


# --- RunResult.usage_report ---


def test_run_result_usage_report_defaults_to_none():
    result = RunResult(
        returncode=0, stdout="", stderr="", timed_out=False, engine="codex", argv=("codex",)
    )
    assert result.usage_report is None


def test_run_result_usage_report_stored():
    report = EngineUsageReport(total_cost_usd=0.5)
    result = RunResult(
        returncode=0,
        stdout="",
        stderr="",
        timed_out=False,
        engine="codex",
        argv=("codex",),
        usage_report=report,
    )
    assert result.usage_report is report
    assert result.usage_report.total_cost_usd == 0.5
