"""Tests for the CLI: argument parsing and main()."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vera_engine.auto_select import Selection
from vera_engine.cli import _build_parser, _collect_credentials, main
from vera_engine.credentials import SecretRef
from vera_engine.invocation import EngineInvocation, RunResult
from vera_engine.models import ModelSpec, Tier


# --- argument parsing ---


def test_parser_list_command():
    args = _build_parser().parse_args(["list"])
    assert args.command == "list"


def test_parser_run_engine_is_optional():
    args = _build_parser().parse_args(["run", "--prompt", "hi"])
    assert args.engine is None


def test_parser_run_rejects_unknown_engine():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["run", "--engine", "not-real", "--prompt", "hi"])


def test_parser_run_defaults():
    args = _build_parser().parse_args(["run", "--engine", "codex", "--prompt", "hi"])
    assert args.engine == "codex"
    assert args.prompt == "hi"
    assert args.model is None
    assert args.tier is None
    assert args.effort == "high"
    assert args.workspace == Path(".")
    assert args.timeout == 2400
    assert args.strategy == "env-key"


def test_parser_run_tier():
    args = _build_parser().parse_args(["run", "--prompt", "hi", "--tier", "stone"])
    assert args.tier == "stone"


def test_parser_run_rejects_bad_tier():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["run", "--prompt", "hi", "--tier", "diamond"])


def test_parser_run_all_options():
    args = _build_parser().parse_args(
        [
            "run",
            "--engine",
            "opencode",
            "--prompt",
            "hi",
            "--model",
            "m1",
            "--effort",
            "low",
            "--workspace",
            "/tmp/ws",
            "--timeout",
            "10",
            "--strategy",
            "proxy",
        ]
    )
    assert args.model == "m1"
    assert args.effort == "low"
    assert args.workspace == Path("/tmp/ws")
    assert args.timeout == 10
    assert args.strategy == "proxy"


def test_parser_run_rejects_bad_effort():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["run", "--engine", "codex", "--prompt", "hi", "--effort", "extreme"]
        )


def test_parser_run_prompt_file_option():
    args = _build_parser().parse_args(
        ["run", "--engine", "codex", "-f", "prompt.txt"]
    )
    assert args.prompt_file == Path("prompt.txt")


# --- _collect_credentials ---


def test_collect_credentials_reads_env_vars(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "value123")
    inv = EngineInvocation(
        engine="codex", argv=("codex",), env={"KEY": SecretRef("MY_TOKEN")}, workdir=Path(".")
    )
    bundle = _collect_credentials(inv)
    assert bundle.values == {"MY_TOKEN": "value123"}


def test_collect_credentials_missing_exits(monkeypatch, capsys):
    monkeypatch.delenv("MISSING_TOKEN", raising=False)
    inv = EngineInvocation(
        engine="codex",
        argv=("codex",),
        env={"KEY": SecretRef("MISSING_TOKEN")},
        workdir=Path("."),
    )
    with pytest.raises(SystemExit) as exc_info:
        _collect_credentials(inv)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "MISSING_TOKEN" in captured.err


def test_collect_credentials_ignores_plain_string_env(monkeypatch):
    inv = EngineInvocation(
        engine="codex", argv=("codex",), env={"PLAIN": "literal-value"}, workdir=Path(".")
    )
    bundle = _collect_credentials(inv)
    assert bundle.values == {}


# --- main() ---


def test_main_list_command(capsys):
    rc = main(["list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "claude-code" in out
    assert "codex" in out
    assert "opencode" in out


def test_main_no_command_prints_help():
    rc = main([])
    assert rc == 1


def test_main_run_requires_prompt_or_prompt_file(tmp_path, capsys):
    rc = main(["run", "--engine", "codex", "--workspace", str(tmp_path)])
    assert rc == 1
    assert "one of --prompt or --prompt-file" in capsys.readouterr().err


def test_main_run_rejects_both_prompt_and_prompt_file(tmp_path, capsys):
    prompt_file = tmp_path / "p.txt"
    prompt_file.write_text("from file")
    rc = main(
        [
            "run",
            "--engine",
            "codex",
            "--workspace",
            str(tmp_path),
            "--prompt",
            "inline",
            "--prompt-file",
            str(prompt_file),
        ]
    )
    assert rc == 1
    assert "mutually exclusive" in capsys.readouterr().err


def test_main_run_reads_prompt_from_file(tmp_path, monkeypatch):
    prompt_file = tmp_path / "p.txt"
    prompt_file.write_text("prompt from file")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    captured = {}

    def fake_render_local(invocation, bundle):
        captured["invocation"] = invocation
        captured["bundle"] = bundle
        return RunResult(
            returncode=0, stdout="ok", stderr="", timed_out=False,
            engine=invocation.engine, argv=invocation.argv,
        )

    with patch("vera_engine.cli.render_local", fake_render_local):
        rc = main(
            ["run", "--engine", "codex", "--workspace", str(tmp_path), "-f", str(prompt_file)]
        )

    assert rc == 0
    assert captured["invocation"].argv[-1] == "prompt from file"


def test_main_run_success_prints_stdout_returns_returncode(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    def fake_render_local(invocation, bundle):
        return RunResult(
            returncode=0, stdout="engine output\n", stderr="", timed_out=False,
            engine=invocation.engine, argv=invocation.argv,
        )

    with patch("vera_engine.cli.render_local", fake_render_local):
        rc = main(
            ["run", "--engine", "codex", "--workspace", str(tmp_path), "--prompt", "hi"]
        )

    assert rc == 0
    assert capsys.readouterr().out == "engine output\n"


def test_main_run_nonzero_returncode_propagates(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    def fake_render_local(invocation, bundle):
        return RunResult(
            returncode=7, stdout="", stderr="boom", timed_out=False,
            engine=invocation.engine, argv=invocation.argv,
        )

    with patch("vera_engine.cli.render_local", fake_render_local):
        rc = main(
            ["run", "--engine", "codex", "--workspace", str(tmp_path), "--prompt", "hi"]
        )

    assert rc == 7


def test_main_run_timeout_returns_124(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    def fake_render_local(invocation, bundle):
        return RunResult(
            returncode=-1, stdout="", stderr="", timed_out=True,
            engine=invocation.engine, argv=invocation.argv,
        )

    with patch("vera_engine.cli.render_local", fake_render_local):
        rc = main(
            [
                "run", "--engine", "codex", "--workspace", str(tmp_path),
                "--prompt", "hi", "--timeout", "5",
            ]
        )

    assert rc == 124
    assert "timed out after 5s" in capsys.readouterr().err


def test_main_run_missing_credentials_exits_before_render(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("vera_engine.cli.render_local") as fake_render_local:
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--engine", "codex", "--workspace", str(tmp_path), "--prompt", "hi"])

    assert exc_info.value.code == 1
    fake_render_local.assert_not_called()


def test_main_run_none_strategy_needs_no_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fake_render_local(invocation, bundle):
        return RunResult(
            returncode=0, stdout="", stderr="", timed_out=False,
            engine=invocation.engine, argv=invocation.argv,
        )

    with patch("vera_engine.cli.render_local", fake_render_local):
        rc = main(
            [
                "run", "--engine", "codex", "--workspace", str(tmp_path),
                "--prompt", "hi", "--strategy", "none",
            ]
        )

    assert rc == 0


def _clay_codex_selection() -> Selection:
    spec = ModelSpec(
        model_id="gpt-5.6-luna",
        engine="codex",
        provider="openai",
        input_cost=1.0,
        output_cost=1.0,
        tier=Tier.clay,
    )
    return Selection(model=spec, strategy="env-key", effective_cost=1.0)


def test_main_engine_plus_tier_without_model_selects(tmp_path, monkeypatch):
    sel = _clay_codex_selection()
    fake_select = MagicMock(return_value=sel)
    captured = {}

    def fake_render_local(invocation, bundle):
        captured["invocation"] = invocation
        return RunResult(
            returncode=0, stdout="ok", stderr="", timed_out=False,
            engine=invocation.engine, argv=invocation.argv,
        )

    with (
        patch("vera_engine.cli.select_cheapest", fake_select),
        patch("vera_engine.cli.render_local", fake_render_local),
    ):
        rc = main(
            [
                "run",
                "--engine",
                "codex",
                "--tier",
                "clay",
                "--prompt",
                "hi",
                "--workspace",
                str(tmp_path),
                "--strategy",
                "none",
            ]
        )

    assert rc == 0
    fake_select.assert_called_once()
    kwargs = fake_select.call_args.kwargs
    assert kwargs["engine"] == "codex"
    assert kwargs["tier"] == Tier.clay
    argv = captured["invocation"].argv
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "gpt-5.6-luna"
    assert "OPENAI_API_KEY" not in captured["invocation"].env


def test_main_engine_without_model_or_tier_skips_select(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    fake_cheapest = MagicMock()
    fake_model = MagicMock()

    def fake_render_local(invocation, bundle):
        return RunResult(
            returncode=0, stdout="", stderr="", timed_out=False,
            engine=invocation.engine, argv=invocation.argv,
        )

    with (
        patch("vera_engine.cli.select_cheapest", fake_cheapest),
        patch("vera_engine.cli.select_model", fake_model),
        patch("vera_engine.cli.render_local", fake_render_local),
    ):
        rc = main(
            ["run", "--engine", "codex", "--prompt", "hi", "--workspace", str(tmp_path)]
        )

    assert rc == 0
    fake_cheapest.assert_not_called()
    fake_model.assert_not_called()


def test_main_omitted_engine_and_model_still_probes(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    fake_select = MagicMock(return_value=_clay_codex_selection())

    def fake_render_local(invocation, bundle):
        return RunResult(
            returncode=0, stdout="", stderr="", timed_out=False,
            engine=invocation.engine, argv=invocation.argv,
        )

    with (
        patch("vera_engine.cli.select_model", fake_select),
        patch("vera_engine.cli.render_local", fake_render_local),
    ):
        rc = main(["run", "--prompt", "hi", "--workspace", str(tmp_path)])

    assert rc == 0
    fake_select.assert_called_once()
