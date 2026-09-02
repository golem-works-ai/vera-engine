"""CLI entry point: python -m vera_engine run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from vera_engine.auto_select import (
    select_cheapest,
    select_model,
    selection_cost,
    selection_cost_ratio,
)
from vera_engine.capacity import check_all
from vera_engine.config import load_config
from vera_engine.credentials import CredentialBundle, SecretRef
from vera_engine.models import TIER_NAMES, Tier, get_catalog
from vera_engine.render.local import ResumeFailedError, render_local
from vera_engine.request import AgentRunRequest
from vera_engine.selection import get_builder, list_engines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vera-engine",
        description="Engine abstraction layer for AI coding agent CLIs.",
    )
    sub = parser.add_subparsers(dest="command")

    # "run" subcommand
    run_parser = sub.add_parser("run", help="Run an engine")
    run_parser.add_argument(
        "--engine",
        choices=list_engines(),
        help="Engine to run (auto-selected if omitted)",
    )
    run_parser.add_argument(
        "--prompt",
        help="Prompt text (mutually exclusive with --prompt-file)",
    )
    run_parser.add_argument(
        "-f",
        "--prompt-file",
        type=Path,
        help="Read prompt from file",
    )
    run_parser.add_argument("--model", help="Model override")
    run_parser.add_argument(
        "--tier",
        choices=TIER_NAMES,
        help="Minimum capability tier (clay < stone < bronze < iron)",
    )
    run_parser.add_argument(
        "--effort",
        default="high",
        choices=["low", "medium", "high"],
    )
    run_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
        help="Workspace directory (default: cwd)",
    )
    run_parser.add_argument(
        "--timeout",
        type=int,
        default=2400,
        help="Timeout in seconds (default: 2400)",
    )
    run_parser.add_argument(
        "--strategy",
        default="env-key",
        choices=["env-key", "proxy", "none"],
        help="Credential strategy (default: env-key)",
    )
    run_parser.add_argument(
        "--resume",
        help="Resume an existing session by its session id",
    )
    run_parser.add_argument(
        "--structured-output",
        action="store_true",
        help="Request structured JSON output so a session id can be captured",
    )

    # "list" subcommand
    sub.add_parser("list", help="List available engines")

    # "models" subcommand
    sub.add_parser("models", help="List models with effective costs")

    # "capacity" subcommand
    capacity_parser = sub.add_parser(
        "capacity", help="Check remaining capacity per provider"
    )
    capacity_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass the 30-minute local capacity cache",
    )

    return parser


def _collect_credentials(invocation: EngineInvocation) -> CredentialBundle:
    """Collect credentials the invocation actually needs from the environment.

    Reads exactly the env vars named by the invocation's SecretRefs.
    Fails fast on any missing one.
    """

    required = [v for v in invocation.env.values() if isinstance(v, SecretRef)]
    values = {}
    missing = []
    for ref in required:
        val = os.environ.get(ref.name)
        if val:
            values[ref.name] = val
        else:
            missing.append(ref.name)
    if missing:
        print(
            f"error: missing required credentials: {missing}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return CredentialBundle(values=values)


def _cmd_models(workspace: Path) -> int:
    config = load_config(workspace)
    capacities = check_all()
    w = config.input_weight
    rows: list[tuple[float, str, str, str, str, float, float, float, float]] = []
    for spec in get_catalog():
        ratio = selection_cost_ratio(spec, config, capacities)
        effective = selection_cost(spec, config, capacities)
        rows.append(
            (
                effective,
                spec.model_id,
                spec.engine,
                spec.provider,
                spec.tier.name,
                spec.input_cost,
                spec.output_cost,
                ratio,
                effective,
            )
        )
    rows.sort()

    print(f"input_weight: {w}")
    print(
        f"{'model':<50} {'engine':<14} {'provider':<12} {'tier':<8} {'input':>7} {'output':>8} {'ratio':>6} {'effective':>10}"
    )
    print("-" * 119)
    for _, model_id, engine, provider, tier_name, inp, out, ratio, eff in rows:
        print(
            f"{model_id:<50} {engine:<14} {provider:<12} {tier_name:<8} {inp:>6.2f} {out:>7.2f} {ratio:>6.3f} {eff:>9.2f}"
        )
    return 0


def _cmd_capacity(*, refresh: bool = False) -> int:
    capacities = check_all(force=refresh)
    print(
        f"{'provider':<14} {'status':<14} {'remaining':>9} {'tokens':>8} "
        f"{'window':>8} {'pressure':>9}  detail"
    )
    for provider, cap in capacities.items():
        status = "ok" if cap.available else "unavailable"
        remaining = (
            f"{cap.remaining_pct:.0f}%" if cap.remaining_pct is not None else "-"
        )
        tokens = f"{cap.token_used_pct:.0f}%" if cap.token_used_pct is not None else "-"
        window = (
            f"{cap.window_used_pct:.0f}%" if cap.window_used_pct is not None else "-"
        )
        pressure = (
            f"{cap.usage_pressure:.2f}" if cap.usage_pressure is not None else "-"
        )
        label = f" {cap.window_name}" if cap.window_name is not None else ""
        print(
            f"{provider:<14} {status:<14} {remaining:>9} {tokens:>8} "
            f"{window:>8} {pressure:>9}  {cap.detail}{label}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        for name in list_engines():
            print(name)
        return 0

    if args.command == "models":
        workspace = getattr(args, "workspace", None) or Path(".")
        return _cmd_models(workspace)

    if args.command == "capacity":
        return _cmd_capacity(refresh=args.refresh)

    if args.command != "run":
        parser.print_help()
        return 1

    # Resolve prompt.
    if args.prompt_file:
        if args.prompt:
            print(
                "error: --prompt and --prompt-file are mutually exclusive",
                file=sys.stderr,
            )
            return 1
        prompt = args.prompt_file.read_text(encoding="utf-8")
    elif args.prompt:
        prompt = args.prompt
    else:
        print("error: one of --prompt or --prompt-file is required", file=sys.stderr)
        return 1

    workspace = args.workspace.resolve()
    config = load_config(workspace)

    engine = args.engine
    model = args.model
    strategy = args.strategy

    tier = Tier[args.tier] if args.tier else None

    if not model and engine is None:
        sel = select_model(config, engine=engine, tier=tier)
        engine = sel.model.engine
        model = sel.model.model_id
        strategy = sel.strategy
        tier_label = f" tier={sel.model.tier.name}" if tier else ""
        print(
            f"auto-selected: {model} via {engine} (${sel.effective_cost:.2f}/MTok effective, strategy={strategy}{tier_label})",
            file=sys.stderr,
        )
    elif not model and tier is not None:
        # A pinned --engine still participates in cheapest-at-tier when --model is omitted.
        # The old `if not engine and not model` skip spawned Codex default gpt-5.6-terra.
        sel = select_cheapest(config, engine=engine, tier=tier)
        engine = sel.model.engine
        model = sel.model.model_id
        tier_label = f" tier={sel.model.tier.name}"
        print(
            f"auto-selected: {model} via {engine} (${sel.effective_cost:.2f}/MTok effective, strategy={strategy}{tier_label})",
            file=sys.stderr,
        )
    elif not engine:
        matching = [s for s in get_catalog() if s.model_id == model]
        if not matching:
            print(
                f"error: unknown model {model!r} — specify --engine explicitly",
                file=sys.stderr,
            )
            return 1
        engine = matching[0].engine

    request = AgentRunRequest(
        engine=engine,
        prompt=prompt,
        workspace=workspace,
        model=model,
        effort=args.effort,
        timeout_seconds=args.timeout,
        credential_strategy=strategy,
        tier=tier,
        resume=args.resume,
        structured_output=args.structured_output,
    )

    builder = get_builder(request.engine)
    invocation = builder.build_invocation(request, request.credential_strategy)
    bundle = _collect_credentials(invocation)

    try:
        result = render_local(invocation, bundle, builder=builder)
    except ResumeFailedError as exc:
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return exc.returncode if exc.returncode else 1

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.timed_out:
        print(
            f"\nerror: engine timed out after {request.timeout_seconds}s",
            file=sys.stderr,
        )
        return 124

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
