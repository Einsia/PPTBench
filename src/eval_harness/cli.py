"""Command-line entry point for the PPTBench reconstruction harness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .clip_score import DEFAULT_CLIP_MODEL
from .config import load_config
from .harness import AgentSpec, run_harness


DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"


def build_parser() -> argparse.ArgumentParser:
    """Build the PPT-only benchmark command parser."""

    parser = argparse.ArgumentParser(prog="pptbench-eval")
    parser.add_argument("--config", default="project.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    harness_parser = commands.add_parser(
        "harness-run",
        help="run a direct PowerPoint reconstruction harness",
    )
    harness_parser.add_argument(
        "--harness",
        required=True,
        choices=["single-run"],
    )
    agent_choices = ["codex", "claude-code", "opencode"]
    harness_parser.add_argument("--agent", choices=agent_choices, default="codex")
    harness_parser.add_argument("--model", default=DEFAULT_MODEL)
    harness_parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
    )
    harness_parser.add_argument("--codex-command", default="codex")
    harness_parser.add_argument("--claude-command", default="claude")
    harness_parser.add_argument("--opencode-command", default="opencode")
    harness_parser.add_argument("--max-turns", type=int, default=40)
    harness_parser.add_argument("--sample", action="append", dest="samples")
    harness_parser.add_argument("--limit", type=int)
    harness_parser.add_argument("--run-id")
    harness_parser.add_argument(
        "--timeout", type=int, default=0,
        help="agent time limit in seconds; 0 disables it",
    )
    harness_parser.add_argument(
        "--soffice-command",
        default=os.environ.get("FRONTEND_BENCH_SOFFICE", "soffice"),
    )
    harness_parser.add_argument("--with-clip", action="store_true")
    harness_parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    harness_parser.add_argument("--device", default="auto")
    harness_parser.add_argument("--clip-batch-size", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one PPT-only harness command and return its process status."""

    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    config.benchmark.output_dir.mkdir(parents=True, exist_ok=True)
    executables = {
        "codex": args.codex_command,
        "claude-code": args.claude_command,
        "opencode": args.opencode_command,
    }
    generator = AgentSpec(
        kind=args.agent,
        executable=executables[args.agent],
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_turns=args.max_turns,
    )
    summary = run_harness(
        config,
        harness=args.harness,
        agent=generator,
        selectors=args.samples,
        limit=args.limit,
        run_id=args.run_id,
        timeout_seconds=args.timeout,
        include_caption=False,
        with_clip=args.with_clip,
        clip_model=args.clip_model,
        clip_device=args.device,
        clip_batch_size=args.clip_batch_size,
        soffice_command=args.soffice_command,
    )
    print(
        json.dumps(
            {
                "run_root": summary["run_root"],
                "complete_cases": summary["complete_cases"],
                "failed_cases": summary["failed_cases"],
                "mean_final_composite": summary["mean_final_composite"],
                "mean_clip_score": summary["mean_clip_score"],
            },
            indent=2,
        )
    )
    return 0 if summary["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
