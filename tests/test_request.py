"""Tests for AgentRunRequest validation."""

from pathlib import Path

import pytest

from vera_engine.request import AgentRunRequest


def test_valid_construction_defaults(tmp_path):
    req = AgentRunRequest(engine="claude-code", prompt="do stuff", workspace=tmp_path)
    assert req.engine == "claude-code"
    assert req.prompt == "do stuff"
    assert req.workspace == tmp_path
    assert req.model is None
    assert req.effort == "high"
    assert req.timeout_seconds == 2400
    assert req.extra_env == {}
    assert req.credential_strategy == "env-key"


def test_valid_construction_all_fields(tmp_path):
    req = AgentRunRequest(
        engine="opencode",
        prompt="hello",
        workspace=tmp_path,
        model="some-model",
        effort="low",
        timeout_seconds=60,
        extra_env={"FOO": "bar"},
        credential_strategy="proxy",
    )
    assert req.model == "some-model"
    assert req.effort == "low"
    assert req.timeout_seconds == 60
    assert req.extra_env == {"FOO": "bar"}
    assert req.credential_strategy == "proxy"


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_valid_efforts(tmp_path, effort):
    req = AgentRunRequest(engine="codex", prompt="x", workspace=tmp_path, effort=effort)
    assert req.effort == effort


def test_invalid_effort_raises(tmp_path):
    with pytest.raises(ValueError, match="effort must be one of"):
        AgentRunRequest(engine="codex", prompt="x", workspace=tmp_path, effort="extreme")


@pytest.mark.parametrize("strategy", ["env-key", "proxy", "none"])
def test_valid_strategies(tmp_path, strategy):
    req = AgentRunRequest(
        engine="codex", prompt="x", workspace=tmp_path, credential_strategy=strategy
    )
    assert req.credential_strategy == strategy


def test_invalid_strategy_raises(tmp_path):
    with pytest.raises(ValueError, match="credential_strategy must be one of"):
        AgentRunRequest(
            engine="codex", prompt="x", workspace=tmp_path, credential_strategy="magic"
        )


def test_empty_engine_raises(tmp_path):
    with pytest.raises(ValueError, match="engine must not be empty"):
        AgentRunRequest(engine="", prompt="x", workspace=tmp_path)


def test_empty_prompt_raises(tmp_path):
    with pytest.raises(ValueError, match="prompt must not be empty"):
        AgentRunRequest(engine="codex", prompt="", workspace=tmp_path)


def test_negative_timeout_raises(tmp_path):
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        AgentRunRequest(engine="codex", prompt="x", workspace=tmp_path, timeout_seconds=-1)


def test_zero_timeout_raises(tmp_path):
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        AgentRunRequest(engine="codex", prompt="x", workspace=tmp_path, timeout_seconds=0)


def test_frozen_dataclass_is_immutable(tmp_path):
    req = AgentRunRequest(engine="codex", prompt="x", workspace=tmp_path)
    with pytest.raises(Exception):
        req.engine = "other"


def test_workspace_accepts_path(tmp_path):
    req = AgentRunRequest(engine="codex", prompt="x", workspace=tmp_path)
    assert isinstance(req.workspace, Path)
