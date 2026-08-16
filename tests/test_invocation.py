"""Tests for EngineInvocation, MaterializedFile, and RunResult."""

from pathlib import Path

import pytest

from vera_engine.credentials import SecretRef
from vera_engine.invocation import EngineInvocation, MaterializedFile, RunResult


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
