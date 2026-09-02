"""Local subprocess renderer.

Resolves SecretRefs, builds a hermetic environment, materializes config
files, spawns the engine, captures output, and cleans up.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from vera_engine.credentials import CredentialBundle, SecretRef
from vera_engine.invocation import EngineInvocation, MaterializedFile, RunResult

if TYPE_CHECKING:
    from vera_engine.builders.base import EngineBuilder


class ResumeFailedError(RuntimeError):
    """A resume invocation exited non-zero.

    The session id being resumed is likely invalid or stale. Failing fast
    here surfaces the problem instead of returning empty output silently.
    Timeouts are not considered a resume failure — the subprocess did not
    exit, it was killed.
    """

    def __init__(self, resume: str, returncode: int, engine: str, stderr: str) -> None:
        self.resume = resume
        self.returncode = returncode
        self.engine = engine
        self.stderr = stderr
        super().__init__(
            f"resume of session {resume!r} on engine {engine!r} "
            f"exited non-zero (code {returncode})"
        )


def _resolve_env(
    env: dict[str, str | SecretRef],
    bundle: CredentialBundle,
) -> dict[str, str]:
    """Replace SecretRefs with actual values from the bundle."""
    resolved: dict[str, str] = {}
    for key, value in env.items():
        if isinstance(value, SecretRef):
            resolved[key] = bundle.resolve(value)
        else:
            resolved[key] = value
    return resolved


_REFERENCE_RE = re.compile(r"\$(?:\{(?P<braced>\w+)\}|(?P<bare>\w+))")


def _expand_vars(text: str, env: dict[str, str]) -> str:
    """Expand $NAME and ${NAME} references. Fails fast on unresolved ones."""

    def _replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("bare")
        try:
            return env[name]
        except KeyError:
            raise ValueError(
                f"unresolved reference ${name} in materialized file "
                f"(available: {sorted(env)})"
            ) from None

    return _REFERENCE_RE.sub(_replace, text)


def _build_base_env(
    invocation: EngineInvocation,
    resolved_env: dict[str, str],
    hermetic_home: Path | None,
) -> dict[str, str]:
    """Build the subprocess environment.

    Hermetic mode: empty base + PATH + resolved env.
    Shared mode: inherit current env + resolved env.
    Real mode: full current env + resolved env.
    """
    if invocation.home_strategy == "hermetic":
        base: dict[str, str] = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }
        if hermetic_home:
            base["HOME"] = str(hermetic_home)
    elif invocation.home_strategy == "shared":
        base = dict(os.environ)
        if hermetic_home:
            base["HOME"] = str(hermetic_home)
    else:
        base = dict(os.environ)

    base.update(resolved_env)
    return base


def _materialize_files(
    files: tuple[MaterializedFile, ...],
    workdir: Path,
    env: dict[str, str],
) -> list[Path]:
    """Write materialized files to the workspace. Returns paths for cleanup."""
    paths: list[Path] = []
    for mf in files:
        target = workdir / mf.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        content = _expand_vars(mf.content, env) if mf.expand_references else mf.content
        target.write_text(content, encoding="utf-8")
        target.chmod(mf.mode)
        if mf.cleanup:
            paths.append(target)
    return paths


def _cleanup_files(paths: list[Path]) -> None:
    """Remove materialized files, ignoring errors."""
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def render_local(
    invocation: EngineInvocation,
    bundle: CredentialBundle,
    builder: EngineBuilder | None = None,
) -> RunResult:
    """Spawn the engine subprocess locally and return the result.

    When ``builder`` is provided, its ``parse_session_id`` is invoked on the
    captured stdout to populate ``RunResult.session_id``. A resume invocation
    (``invocation.resume`` set) that exits non-zero raises
    ``ResumeFailedError``; a timeout is not treated as a resume failure.
    """
    resolved_env = _resolve_env(invocation.env, bundle)

    # Create hermetic HOME if needed.
    hermetic_home: Path | None = None
    hermetic_tmpdir = None
    if invocation.home_strategy in ("hermetic", "shared"):
        hermetic_tmpdir = tempfile.mkdtemp(prefix="vera-engine-home-")
        hermetic_home = Path(hermetic_tmpdir)

    try:
        full_env = _build_base_env(invocation, resolved_env, hermetic_home)

        # Materialize config files.
        cleanup_paths = _materialize_files(
            invocation.files, invocation.workdir, full_env
        )

        try:
            result = subprocess.run(
                list(invocation.argv),
                cwd=str(invocation.workdir),
                env=full_env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=invocation.timeout_seconds,
            )
            # Fail fast: a resume that exits non-zero almost certainly means
            # the session id is invalid or stale. Timeouts fall through to the
            # except branch below and are reported as a timeout, not a resume
            # failure.
            if invocation.resume is not None and result.returncode != 0:
                raise ResumeFailedError(
                    invocation.resume,
                    result.returncode,
                    invocation.engine,
                    result.stderr,
                )
            session_id = (
                builder.parse_session_id(result.stdout) if builder is not None else None
            )
            usage_report = (
                builder.parse_usage_report(result.stdout)
                if builder is not None
                else None
            )
            return RunResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False,
                engine=invocation.engine,
                argv=invocation.argv,
                session_id=session_id,
                usage_report=usage_report,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_stdout = (
                exc.stdout or ""
                if isinstance(exc.stdout, str)
                else (exc.stdout or b"").decode("utf-8", errors="replace")
            )
            timeout_stderr = (
                exc.stderr or ""
                if isinstance(exc.stderr, str)
                else (exc.stderr or b"").decode("utf-8", errors="replace")
            )
            usage_report = (
                builder.parse_usage_report(timeout_stdout)
                if builder is not None
                else None
            )
            return RunResult(
                returncode=-1,
                stdout=timeout_stdout,
                stderr=timeout_stderr,
                timed_out=True,
                engine=invocation.engine,
                argv=invocation.argv,
                session_id=None,
                usage_report=usage_report,
            )
        finally:
            _cleanup_files(cleanup_paths)
    finally:
        # Clean up hermetic home directory.
        if hermetic_tmpdir:
            shutil.rmtree(hermetic_tmpdir, ignore_errors=True)
