"""Local subprocess renderer.

Resolves SecretRefs, builds a hermetic environment, materializes config
files, spawns the engine, captures output, and cleans up.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from vera_engine.credentials import CredentialBundle, SecretRef
from vera_engine.invocation import EngineInvocation, MaterializedFile, RunResult


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


def _expand_vars(text: str, env: dict[str, str]) -> str:
    """Expand $NAME references in text using the resolved env."""
    for name, value in env.items():
        text = text.replace(f"${name}", value)
    return text


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
        content = _expand_vars(mf.content, env)
        target.write_text(content, encoding="utf-8")
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
) -> RunResult:
    """Spawn the engine subprocess locally and return the result."""
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
                capture_output=True,
                text=True,
                timeout=invocation.timeout_seconds,
            )
            return RunResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False,
                engine=invocation.engine,
                argv=invocation.argv,
            )
        except subprocess.TimeoutExpired as exc:
            return RunResult(
                returncode=-1,
                stdout=exc.stdout or "" if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace"),
                stderr=exc.stderr or "" if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace"),
                timed_out=True,
                engine=invocation.engine,
                argv=invocation.argv,
            )
        finally:
            _cleanup_files(cleanup_paths)
    finally:
        # Clean up hermetic home directory.
        if hermetic_tmpdir:
            try:
                import shutil
                shutil.rmtree(hermetic_tmpdir, ignore_errors=True)
            except OSError:
                pass
