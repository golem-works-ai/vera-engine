"""CLI entry point: python -m vera_engine run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from vera_engine.credentials import CredentialBundle, SecretRef
from vera_engine.request import AgentRunRequest
from vera_engine.selection import get_builder, list_engines
from vera_engine.render.local import render_local


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
        required=True,
        choices=list_engines(),
        help="Engine to run",
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

    # "list" subcommand
    sub.add_parser("list", help="List available engines")

    return parser


def _collect_credentials(invocation: "EngineInvocation") -> CredentialBundle:
    """Collect credentials the invocation actually needs from the environment.

    Reads exactly the env vars named by the invocation's SecretRefs.
    Fails fast on any missing one.
    """
    from vera_engine.invocation import EngineInvocation  # noqa: F811

    required = [
        v for v in invocation.env.values()
        if isinstance(v, SecretRef)
    ]
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        for name in list_engines():
            print(name)
        return 0

    if args.command != "run":
        parser.print_help()
        return 1

    # Resolve prompt.
    if args.prompt_file:
        if args.prompt:
            print("error: --prompt and --prompt-file are mutually exclusive", file=sys.stderr)
            return 1
        prompt = args.prompt_file.read_text(encoding="utf-8")
    elif args.prompt:
        prompt = args.prompt
    else:
        print("error: one of --prompt or --prompt-file is required", file=sys.stderr)
        return 1

    request = AgentRunRequest(
        engine=args.engine,
        prompt=prompt,
        workspace=args.workspace.resolve(),
        model=args.model,
        effort=args.effort,
        timeout_seconds=args.timeout,
        credential_strategy=args.strategy,
    )

    builder = get_builder(request.engine)
    invocation = builder.build_invocation(request, request.credential_strategy)
    bundle = _collect_credentials(invocation)

    result = render_local(invocation, bundle)

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.timed_out:
        print(f"\nerror: engine timed out after {request.timeout_seconds}s", file=sys.stderr)
        return 124

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
