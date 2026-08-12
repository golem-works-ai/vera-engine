"""Acceptance tests that run vera-engine against real engines."""

import subprocess
import sys

import pytest

PROMPT = "Respond with exactly the word 'hello' and nothing else."
TIMEOUT = 120


def _run(*extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "vera_engine", "run", "--prompt", PROMPT, *extra_args],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


@pytest.mark.acceptance
def test_auto_select():
    result = _run()
    assert result.returncode == 0, f"exit {result.returncode}\nstderr: {result.stderr}"
    assert result.stdout.strip(), f"empty stdout\nstderr: {result.stderr}"


@pytest.mark.acceptance
def test_claude_code():
    result = _run("--engine", "claude-code", "--strategy", "none")
    assert result.returncode == 0, f"exit {result.returncode}\nstderr: {result.stderr}"
    assert result.stdout.strip(), f"empty stdout\nstderr: {result.stderr}"


@pytest.mark.acceptance
def test_opencode():
    result = _run("--engine", "opencode")
    assert result.returncode == 0, f"exit {result.returncode}\nstderr: {result.stderr}"
    assert result.stdout.strip(), f"empty stdout\nstderr: {result.stderr}"


@pytest.mark.acceptance
def test_codex():
    result = _run("--engine", "codex")
    if "ChatGPT account" in result.stderr:
        pytest.skip("codex needs an OpenAI API key, not ChatGPT auth")
    assert result.returncode == 0, f"exit {result.returncode}\nstderr: {result.stderr}"
    assert result.stdout.strip(), f"empty stdout\nstderr: {result.stderr}"
