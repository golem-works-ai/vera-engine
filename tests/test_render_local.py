"""Tests for the local subprocess renderer."""

import os

import pytest

from vera_engine.credentials import CredentialBundle, CredentialGuardError, SecretRef
from vera_engine.invocation import EngineInvocation, MaterializedFile
from vera_engine.render.local import (
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
    files = (MaterializedFile(relative_path="cfg.json", content='{"url": "$BASE_URL"}'),)
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
    files = (MaterializedFile(relative_path="bad.txt", content="$UNSET"),)
    with pytest.raises(ValueError, match="unresolved reference"):
        _materialize_files(files, tmp_path, {})


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
