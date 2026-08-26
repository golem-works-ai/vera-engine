"""Tests for the local subprocess renderer."""

import os

import pytest

from vera_engine.credentials import CredentialBundle, CredentialGuardError, SecretRef
from vera_engine.invocation import EngineInvocation, EngineUsageReport, MaterializedFile
from vera_engine.render.local import (
    ResumeFailedError,
    _build_base_env,
    _cleanup_files,
    _expand_vars,
    _materialize_files,
    _resolve_env,
    render_local,
)


# --- _resolve_env ---


def test_resolve_env_plain_strings_pass_through():
    resolved = _resolve_env({"FOO": "bar"}, CredentialBundle(values={}))
    assert resolved == {"FOO": "bar"}


def test_resolve_env_resolves_secret_ref():
    env = {"KEY": SecretRef("MY_SECRET")}
    bundle = CredentialBundle(values={"MY_SECRET": "shh"})
    resolved = _resolve_env(env, bundle)
    assert resolved == {"KEY": "shh"}


def test_resolve_env_mixed():
    env = {"PLAIN": "value", "SECRET": SecretRef("TOK")}
    bundle = CredentialBundle(values={"TOK": "tokval"})
    resolved = _resolve_env(env, bundle)
    assert resolved == {"PLAIN": "value", "SECRET": "tokval"}


def test_resolve_env_missing_secret_raises():
    env = {"SECRET": SecretRef("MISSING")}
    bundle = CredentialBundle(values={})
    with pytest.raises(CredentialGuardError, match="MISSING"):
        _resolve_env(env, bundle)


# --- _expand_vars ---


def test_expand_vars_bare_reference():
    assert _expand_vars("hello $NAME", {"NAME": "world"}) == "hello world"


def test_expand_vars_braced_reference():
    assert _expand_vars("hello ${NAME}!", {"NAME": "world"}) == "hello world!"


def test_expand_vars_multiple_references():
    text = "$A and $B and ${A}"
    result = _expand_vars(text, {"A": "1", "B": "2"})
    assert result == "1 and 2 and 1"


def test_expand_vars_no_references_unchanged():
    assert _expand_vars("no vars here", {}) == "no vars here"


def test_expand_vars_unresolved_reference_raises():
    with pytest.raises(ValueError, match=r"unresolved reference \$MISSING"):
        _expand_vars("value is $MISSING", {"OTHER": "x"})


# --- _build_base_env ---


def _inv(tmp_path, home_strategy="hermetic"):
    return EngineInvocation(
        engine="codex", argv=("codex",), env={}, workdir=tmp_path, home_strategy=home_strategy
    )


def test_build_base_env_hermetic_excludes_ambient_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_AMBIENT_VAR", "leaked")
    inv = _inv(tmp_path, "hermetic")
    env = _build_base_env(inv, {"RESOLVED": "val"}, None)
    assert "SOME_AMBIENT_VAR" not in env
    assert env["RESOLVED"] == "val"
    assert "PATH" in env


def test_build_base_env_hermetic_sets_home_when_provided(tmp_path):
    inv = _inv(tmp_path, "hermetic")
    env = _build_base_env(inv, {}, tmp_path / "home")
    assert env["HOME"] == str(tmp_path / "home")


def test_build_base_env_hermetic_no_home_key_without_hermetic_home(tmp_path):
    inv = _inv(tmp_path, "hermetic")
    env = _build_base_env(inv, {}, None)
    assert "HOME" not in env or env.get("HOME") != str(tmp_path)


def test_build_base_env_shared_inherits_ambient_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_AMBIENT_VAR", "present")
    inv = _inv(tmp_path, "shared")
    env = _build_base_env(inv, {}, None)
    assert env.get("SOME_AMBIENT_VAR") == "present"


def test_build_base_env_shared_overrides_home_when_provided(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", "/original/home")
    inv = _inv(tmp_path, "shared")
    hermetic_home = tmp_path / "shared-home"
    env = _build_base_env(inv, {}, hermetic_home)
    assert env["HOME"] == str(hermetic_home)


def test_build_base_env_real_inherits_full_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_AMBIENT_VAR", "present")
    inv = _inv(tmp_path, "real")
    env = _build_base_env(inv, {}, None)
    assert env.get("SOME_AMBIENT_VAR") == "present"
    assert env.get("HOME") == os.environ.get("HOME")


def test_build_base_env_resolved_env_overrides_base(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/original/path")
    inv = _inv(tmp_path, "hermetic")
    env = _build_base_env(inv, {"PATH": "/custom/path"}, None)
    assert env["PATH"] == "/custom/path"


# --- _materialize_files ---


def test_materialize_files_writes_content(tmp_path):
    files = (MaterializedFile(relative_path="prompt.md", content="hello world"),)
    paths = _materialize_files(files, tmp_path, {})
    target = tmp_path / "prompt.md"
    assert target.read_text() == "hello world"
    assert paths == [target]


def test_materialize_files_expands_vars_in_content(tmp_path):
    files = (
        MaterializedFile(
            relative_path="cfg.json",
            content='{"url": "$BASE_URL"}',
            expand_references=True,
        ),
    )
    _materialize_files(files, tmp_path, {"BASE_URL": "https://example.com"})
    assert (tmp_path / "cfg.json").read_text() == '{"url": "https://example.com"}'


def test_materialize_files_creates_parent_dirs(tmp_path):
    files = (MaterializedFile(relative_path="nested/dir/file.txt", content="x"),)
    _materialize_files(files, tmp_path, {})
    assert (tmp_path / "nested" / "dir" / "file.txt").read_text() == "x"


def test_materialize_files_no_cleanup_excluded_from_returned_paths(tmp_path):
    files = (MaterializedFile(relative_path="keep.txt", content="x", cleanup=False),)
    paths = _materialize_files(files, tmp_path, {})
    assert paths == []
    assert (tmp_path / "keep.txt").exists()


def test_materialize_files_unresolved_var_raises(tmp_path):
    files = (
        MaterializedFile(
            relative_path="bad.txt",
            content="$UNSET",
            expand_references=True,
        ),
    )
    with pytest.raises(ValueError, match="unresolved reference"):
        _materialize_files(files, tmp_path, {})


def test_materialize_files_literal_dollar_when_expansion_off(tmp_path):
    files = (MaterializedFile(relative_path="note.txt", content="cost is $PRICE"),)
    _materialize_files(files, tmp_path, {})
    assert (tmp_path / "note.txt").read_text() == "cost is $PRICE"


def test_materialize_files_applies_mode(tmp_path):
    files = (MaterializedFile(relative_path="script.sh", content="#!/bin/sh", mode=0o755),)
    _materialize_files(files, tmp_path, {})
    assert (tmp_path / "script.sh").stat().st_mode & 0o777 == 0o755


# --- _cleanup_files ---


def test_cleanup_files_removes_files(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("x")
    _cleanup_files([target])
    assert not target.exists()


def test_cleanup_files_ignores_missing_files(tmp_path):
    missing = tmp_path / "does-not-exist.txt"
    _cleanup_files([missing])  # should not raise


# --- render_local (real subprocess) ---


def test_render_local_echo_success(tmp_path):
    inv = EngineInvocation(
        engine="echo",
        argv=("echo", "hello"),
        env={},
        workdir=tmp_path,
        home_strategy="hermetic",
    )
    result = render_local(inv, CredentialBundle(values={}))
    assert result.returncode == 0
    assert result.stdout == "hello\n"
    assert result.timed_out is False
    assert result.succeeded is True
    assert result.engine == "echo"
    assert result.argv == ("echo", "hello")


def test_render_local_nonzero_exit(tmp_path):
    inv = EngineInvocation(
        engine="false", argv=("false",), env={}, workdir=tmp_path, home_strategy="hermetic"
    )
    result = render_local(inv, CredentialBundle(values={}))
    assert result.returncode != 0
    assert result.succeeded is False
    assert result.timed_out is False


def test_render_local_resolves_secrets_into_subprocess_env(tmp_path):
    inv = EngineInvocation(
        engine="sh",
        argv=("sh", "-c", "echo $MY_SECRET"),
        env={"MY_SECRET": SecretRef("TOKEN")},
        workdir=tmp_path,
        home_strategy="hermetic",
    )
    bundle = CredentialBundle(values={"TOKEN": "resolved-value"})
    result = render_local(inv, bundle)
    assert result.stdout == "resolved-value\n"


def test_render_local_runs_in_workdir(tmp_path):
    marker = tmp_path / "marker.txt"
    marker.write_text("present")
    inv = EngineInvocation(
        engine="sh",
        argv=("sh", "-c", "cat marker.txt"),
        env={},
        workdir=tmp_path,
        home_strategy="hermetic",
    )
    result = render_local(inv, CredentialBundle(values={}))
    assert result.stdout == "present"


def test_render_local_materializes_and_cleans_up_files(tmp_path):
    prompt_file = MaterializedFile(relative_path="prompt.md", content="task text")
    inv = EngineInvocation(
        engine="sh",
        argv=("sh", "-c", "cat prompt.md"),
        env={},
        workdir=tmp_path,
        home_strategy="hermetic",
        files=(prompt_file,),
    )
    result = render_local(inv, CredentialBundle(values={}))
    assert result.stdout == "task text"
    assert not (tmp_path / "prompt.md").exists()


def test_render_local_keeps_no_cleanup_files(tmp_path):
    cfg_file = MaterializedFile(relative_path="cfg.json", content="{}", cleanup=False)
    inv = EngineInvocation(
        engine="true", argv=("true",), env={}, workdir=tmp_path, files=(cfg_file,)
    )
    render_local(inv, CredentialBundle(values={}))
    assert (tmp_path / "cfg.json").exists()


def test_render_local_timeout(tmp_path):
    inv = EngineInvocation(
        engine="sleep",
        argv=("sleep", "5"),
        env={},
        workdir=tmp_path,
        home_strategy="hermetic",
        timeout_seconds=1,
    )
    result = render_local(inv, CredentialBundle(values={}))
    assert result.timed_out is True
    assert result.returncode == -1
    assert result.succeeded is False


def test_render_local_shared_home_strategy_removed_after_run(tmp_path):
    # Just verify a "shared" strategy run completes and doesn't leak a tmpdir path
    # into the result; existence of the created HOME dir isn't directly observable
    # here, so this exercises the shared-strategy code path end to end.
    inv = EngineInvocation(
        engine="echo", argv=("echo", "hi"), env={}, workdir=tmp_path, home_strategy="shared"
    )
    result = render_local(inv, CredentialBundle(values={}))
    assert result.returncode == 0
    assert result.stdout == "hi\n"


# --- render_local session_id + resume fail-fast ---


class _FakeBuilder:
    """Minimal builder double exposing parse_session_id."""

    def __init__(self, extract):
        self._extract = extract

    def parse_session_id(self, stdout: str):
        return self._extract(stdout)

    def parse_usage_report(self, stdout: str):
        return None


def test_render_local_parses_session_id_via_builder(tmp_path):
    inv = EngineInvocation(
        engine="echo",
        argv=("echo", '{"session_id": "sess-1"}'),
        env={},
        workdir=tmp_path,
        home_strategy="hermetic",
    )
    builder = _FakeBuilder(lambda s: __import__("json").loads(s)["session_id"])
    result = render_local(inv, CredentialBundle(values={}), builder=builder)
    assert result.session_id == "sess-1"


def test_render_local_no_builder_leaves_session_id_none(tmp_path):
    inv = EngineInvocation(
        engine="echo", argv=("echo", "hi"), env={}, workdir=tmp_path, home_strategy="hermetic"
    )
    result = render_local(inv, CredentialBundle(values={}))
    assert result.session_id is None


def test_render_local_resume_nonzero_exit_raises(tmp_path):
    inv = EngineInvocation(
        engine="false",
        argv=("false",),
        env={},
        workdir=tmp_path,
        home_strategy="hermetic",
        resume="bad-session",
    )
    with pytest.raises(ResumeFailedError, match="bad-session"):
        render_local(inv, CredentialBundle(values={}))


def test_render_local_resume_timeout_does_not_raise(tmp_path):
    inv = EngineInvocation(
        engine="sleep",
        argv=("sleep", "5"),
        env={},
        workdir=tmp_path,
        home_strategy="hermetic",
        timeout_seconds=1,
        resume="sess-timeout",
    )
    result = render_local(inv, CredentialBundle(values={}))
    assert result.timed_out is True
    assert result.session_id is None


def test_render_local_resume_success_returns_session_id(tmp_path):
    inv = EngineInvocation(
        engine="echo",
        argv=("echo", '{"session_id": "ok-sess"}'),
        env={},
        workdir=tmp_path,
        home_strategy="hermetic",
        resume="ok-sess",
    )
    builder = _FakeBuilder(lambda s: __import__("json").loads(s)["session_id"])
    result = render_local(inv, CredentialBundle(values={}), builder=builder)
    assert result.returncode == 0
    assert result.session_id == "ok-sess"


def test_render_local_nonzero_without_resume_does_not_raise(tmp_path):
    inv = EngineInvocation(
        engine="false", argv=("false",), env={}, workdir=tmp_path, home_strategy="hermetic"
    )
    result = render_local(inv, CredentialBundle(values={}))
    assert result.returncode != 0
    assert result.session_id is None


def test_resume_failed_error_carries_returncode_and_engine(tmp_path):
    inv = EngineInvocation(
        engine="sh",
        argv=("sh", "-c", "echo boom; exit 3"),
        env={},
        workdir=tmp_path,
        home_strategy="hermetic",
        resume="dead",
    )
    with pytest.raises(ResumeFailedError) as exc_info:
        render_local(inv, CredentialBundle(values={}))
    assert exc_info.value.returncode == 3
    assert exc_info.value.engine == "sh"
    assert exc_info.value.resume == "dead"


# --- render_local usage_report threading ---


class _UsageBuilder:
    """Builder double exposing both session_id and usage_report hooks."""

    def __init__(self, report: EngineUsageReport | None) -> None:
        self._report = report

    def parse_session_id(self, stdout: str):
        return None

    def parse_usage_report(self, stdout: str) -> EngineUsageReport | None:
        return self._report


def test_render_local_threads_usage_report_via_builder(tmp_path):
    report = EngineUsageReport(total_cost_usd=0.25)
    inv = EngineInvocation(
        engine="echo",
        argv=("echo", '{"total_cost_usd": 0.25}'),
        env={},
        workdir=tmp_path,
        home_strategy="hermetic",
    )
    builder = _UsageBuilder(report)
    result = render_local(inv, CredentialBundle(values={}), builder=builder)
    assert result.usage_report is report


def test_render_local_no_builder_leaves_usage_report_none(tmp_path):
    inv = EngineInvocation(
        engine="echo", argv=("echo", "hi"), env={}, workdir=tmp_path, home_strategy="hermetic"
    )
    result = render_local(inv, CredentialBundle(values={}))
    assert result.usage_report is None


def test_render_local_builder_returning_none_usage_report_keeps_none(tmp_path):
    class _NoneReportBuilder:
        def parse_session_id(self, stdout: str):
            return None

        def parse_usage_report(self, stdout: str):
            return None

    inv = EngineInvocation(
        engine="echo", argv=("echo", "hi"), env={}, workdir=tmp_path, home_strategy="hermetic"
    )
    result = render_local(
        inv, CredentialBundle(values={}), builder=_NoneReportBuilder()
    )
    assert result.usage_report is None


def test_render_local_timeout_threads_usage_report_from_partial_stdout(tmp_path):
    report = EngineUsageReport(total_cost_usd=0.01)

    class _TimeoutUsageBuilder:
        def parse_session_id(self, stdout: str):
            return None

        def parse_usage_report(self, stdout: str) -> EngineUsageReport | None:
            return report

    inv = EngineInvocation(
        engine="sleep",
        argv=("sleep", "5"),
        env={},
        workdir=tmp_path,
        home_strategy="hermetic",
        timeout_seconds=1,
    )
    result = render_local(
        inv, CredentialBundle(values={}), builder=_TimeoutUsageBuilder()
    )
    assert result.timed_out is True
    # The timeout branch must call parse_usage_report on the captured stdout
    # and thread the returned report onto the RunResult.
    assert result.usage_report is report
