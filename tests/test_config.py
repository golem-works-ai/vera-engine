"""Tests for configuration loading."""

import textwrap
from pathlib import Path

import pytest

from vera_engine.config import EngineConfig, load_config, _parse_config


def test_default_config_without_toml(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = load_config(workspace=Path("/nonexistent"))
    assert config.cost_ratio("anthropic") == 1.0
    assert config.cost_ratio("openrouter") == pytest.approx(1.01375)
    assert config.cost_ratio("openai") == 1.0
    assert config.cost_ratio("xai") == 1.0
    assert config.input_weight == 0.8


def test_unknown_provider_returns_one():
    config = EngineConfig(provider_ratios={})
    assert config.cost_ratio("unknown") == 1.0


def test_parse_config(tmp_path):
    cfg = tmp_path / ".vera-engine.toml"
    cfg.write_text(textwrap.dedent("""\
        input_weight = 0.95

        [providers.anthropic]
        cost_ratio = 0.2

        [providers.openrouter]
        cost_ratio = 1.01375
    """))
    config = _parse_config(cfg)
    assert config.cost_ratio("anthropic") == pytest.approx(0.2)
    assert config.cost_ratio("openrouter") == pytest.approx(1.01375)
    assert config.cost_ratio("openai") == 1.0
    assert config.input_weight == pytest.approx(0.95)


def test_load_config_workspace_file(tmp_path):
    cfg = tmp_path / ".vera-engine.toml"
    cfg.write_text(textwrap.dedent("""\
        [providers.anthropic]
        cost_ratio = 0.5
    """))
    config = load_config(workspace=tmp_path)
    assert config.cost_ratio("anthropic") == pytest.approx(0.5)
    assert config.cost_ratio("openrouter") == pytest.approx(1.01375)
    assert config.input_weight == 0.8


def test_parse_config_rejects_negative_ratio(tmp_path):
    cfg = tmp_path / ".vera-engine.toml"
    cfg.write_text(textwrap.dedent("""\
        [providers.anthropic]
        cost_ratio = -1.0
    """))
    with pytest.raises(ValueError, match="positive number"):
        _parse_config(cfg)


def test_parse_config_rejects_zero_ratio(tmp_path):
    cfg = tmp_path / ".vera-engine.toml"
    cfg.write_text(textwrap.dedent("""\
        [providers.openai]
        cost_ratio = 0
    """))
    with pytest.raises(ValueError, match="positive number"):
        _parse_config(cfg)


def test_parse_config_rejects_string_ratio(tmp_path):
    cfg = tmp_path / ".vera-engine.toml"
    cfg.write_text(textwrap.dedent("""\
        [providers.openai]
        cost_ratio = "cheap"
    """))
    with pytest.raises(ValueError, match="positive number"):
        _parse_config(cfg)


def test_parse_config_rejects_input_weight_out_of_range(tmp_path):
    cfg = tmp_path / ".vera-engine.toml"
    cfg.write_text("input_weight = 1.5\n")
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        _parse_config(cfg)


def test_parse_config_rejects_negative_input_weight(tmp_path):
    cfg = tmp_path / ".vera-engine.toml"
    cfg.write_text("input_weight = -0.1\n")
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        _parse_config(cfg)
