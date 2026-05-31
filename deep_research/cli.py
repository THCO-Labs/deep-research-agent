from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deep_research.agent import ResearchRunError, run_research
from deep_research.settings import ConfigError, Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the benchmark-grade deep research agent.")
    parser.add_argument("question", nargs="+", help="Research question to answer.")
    parser.add_argument("--mode", choices=["fast", "balanced", "max_quality"], default="balanced")
    parser.add_argument("--out", default="runs", help="Directory for run artifacts.")
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--scrape-char-limit", type=int, default=None)
    parser.add_argument(
        "--provider",
        choices=["auto", "google", "groq"],
        default=None,
        help="Model provider. auto uses Groq when GROQ_API_KEY is present, otherwise Google.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--fast-model", default=None)
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--researcher-model", default=None)
    parser.add_argument("--analyst-model", default=None)
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--live", action="store_true", help="Mark this run as live in settings/metrics.")
    parser.add_argument(
        "--progress",
        choices=["live", "raw", "quiet"],
        default="live",
        help="Progress display: concise live feed, raw agent stream, or no progress output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip()
    try:
        settings = Settings.from_env(
            project_root=Path.cwd(),
            mode=args.mode,
            out_dir=args.out,
            max_sources=args.max_sources,
            max_rounds=args.max_rounds,
            provider=args.provider,
            model=args.model,
            fast_model=args.fast_model,
            planner_model=args.planner_model,
            researcher_model=args.researcher_model,
            analyst_model=args.analyst_model,
            verifier_model=args.verifier_model,
            judge_model=args.judge_model,
            scrape_char_limit=args.scrape_char_limit,
            live=args.live,
        )
        result = run_research(
            question,
            settings,
            on_update=None if args.progress == "quiet" else print,
            progress_mode=args.progress,
        )
    except ResearchRunError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Run directory: {exc.result.run_dir}", file=sys.stderr)
        print(f"Verification: {exc.result.verification_path}", file=sys.stderr)
        print(f"Metrics: {exc.result.metrics_path}", file=sys.stderr)
        return 1
    except (ConfigError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"\nRun directory: {result.run_dir}")
    print(f"Report: {result.report_path}")
    print(f"Verification: {result.verification_path}")
    print(f"Metrics: {result.metrics_path}")
    return 0


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
