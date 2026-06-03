from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deep_research.agent import ResearchRunError, resume_research, run_research, verify_research_run
from deep_research.settings import ConfigError, Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LangGraph deep research engine.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run local LangGraph research.")
    _add_runtime_flags(run_parser)
    run_parser.add_argument("question", nargs="+", help="Research question to answer.")

    resume_parser = subparsers.add_parser("resume", help="Resume a run from the latest checkpoint.")
    _add_runtime_flags(resume_parser)
    resume_parser.add_argument("run_id", help="Run directory name under --out.")

    verify_parser = subparsers.add_parser("verify", help="Rerun v2 verification for an existing run.")
    verify_parser.add_argument("run_id", help="Run directory name under --out.")
    verify_parser.add_argument("--out", default="runs", help="Directory for run artifacts.")

    managed_parser = subparsers.add_parser("managed", help="Run a managed deep research provider.")
    managed_parser.add_argument("managed_provider", choices=["gemini", "openai"])
    _add_runtime_flags(managed_parser, include_engine=False)
    managed_parser.add_argument("question", nargs="+", help="Research question to answer.")
    return parser


def _add_runtime_flags(parser: argparse.ArgumentParser, *, include_engine: bool = True) -> None:
    parser.add_argument("--mode", choices=["fast", "balanced", "max_quality"], default="max_quality")
    parser.add_argument("--out", default="runs", help="Directory for run artifacts.")
    if include_engine:
        parser.add_argument(
            "--engine",
            choices=["local_langgraph", "gemini_managed", "openai_managed"],
            default=None,
            help="Research engine.",
        )
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--min-usable-sources", type=int, default=None)
    parser.add_argument("--max-search-queries", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--min-source-words", type=int, default=None)
    parser.add_argument("--input", action="append", default=[], help="Local file or directory to ingest.")
    parser.add_argument("--mcp-manifest", default=None, help="JSON manifest for MCP connector source payloads.")
    parser.add_argument(
        "--provider",
        choices=["auto", "google", "groq", "hybrid", "ollama"],
        default=None,
        help="Model provider for local utility/model policy routing.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--fast-model", default=None)
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--researcher-model", default=None)
    parser.add_argument("--analyst-model", default=None)
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--scrape-char-limit", type=int, default=None)
    parser.add_argument("--no-model-fallbacks", action="store_true")
    parser.add_argument("--no-llm-planning", action="store_true")
    parser.add_argument("--no-llm-synthesis", action="store_true")
    parser.add_argument("--no-semantic-verification", action="store_true")
    parser.add_argument("--allow-failed-verification", action="store_true")
    parser.add_argument("--allow-weak-tool-models", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--progress",
        choices=["live", "raw", "quiet"],
        default="live",
        help="Progress display: concise live feed, raw stream, or no progress output.",
    )


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in {"run", "resume", "verify", "managed", "-h", "--help"}:
        argv.insert(0, "run")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "verify":
            settings = Settings(project_root=Path.cwd(), out_dir=_resolve_out(args.out))
            result = verify_research_run(args.run_id, settings)
        elif args.command == "resume":
            settings = _settings_from_args(args)
            result = resume_research(
                args.run_id,
                settings,
                on_update=None if args.progress == "quiet" else print,
                progress_mode=args.progress,
            )
        elif args.command == "managed":
            engine = "gemini_managed" if args.managed_provider == "gemini" else "openai_managed"
            settings = _settings_from_args(args, engine=engine)
            result = run_research(
                " ".join(args.question).strip(),
                settings,
                on_update=None if args.progress == "quiet" else print,
                progress_mode=args.progress,
            )
        else:
            settings = _settings_from_args(args, engine=args.engine or "local_langgraph")
            result = run_research(
                " ".join(args.question).strip(),
                settings,
                on_update=None if args.progress == "quiet" else print,
                progress_mode=args.progress,
            )
    except ResearchRunError as exc:
        _print_run_error(exc)
        return 1
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"\nRun directory: {result.run_dir}")
    print(f"Report: {result.report_path}")
    print(f"Verification: {result.verification_path}")
    print(f"Metrics: {result.metrics_path}")
    return 0


def _settings_from_args(args: argparse.Namespace, *, engine: str | None = None) -> Settings:
    return Settings.from_env(
        project_root=Path.cwd(),
        mode=args.mode,
        out_dir=args.out,
        max_sources=args.max_sources,
        max_rounds=args.max_rounds,
        research_engine=engine or getattr(args, "engine", None),
        min_usable_sources=args.min_usable_sources,
        max_search_queries=args.max_search_queries,
        max_candidates=args.max_candidates,
        min_source_words=args.min_source_words,
        local_input_paths=tuple(args.input or ()),
        mcp_manifest=args.mcp_manifest,
        provider=args.provider,
        model=args.model,
        fast_model=args.fast_model,
        planner_model=args.planner_model,
        researcher_model=args.researcher_model,
        analyst_model=args.analyst_model,
        verifier_model=args.verifier_model,
        judge_model=args.judge_model,
        scrape_char_limit=args.scrape_char_limit,
        semantic_verification=not args.no_semantic_verification,
        llm_planning=not args.no_llm_planning,
        llm_synthesis=not args.no_llm_synthesis,
        allow_failed_verification=args.allow_failed_verification,
        model_fallbacks=not args.no_model_fallbacks,
        strict_tool_models=not args.allow_weak_tool_models,
        live=args.live,
    )


def _resolve_out(out_dir: str) -> Path:
    path = Path(out_dir)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _print_run_error(exc: ResearchRunError) -> None:
    print(f"Error: {exc}", file=sys.stderr)
    failure_path = exc.result.run_dir / "failure.json"
    if failure_path.exists():
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        print(f"Failure category: {failure.get('category')}", file=sys.stderr)
        if failure.get("retry_after_seconds") is not None:
            print(f"Retry after: {failure.get('retry_after_seconds')}s", file=sys.stderr)
        print(f"Suggested action: {failure.get('suggested_action')}", file=sys.stderr)
    print(f"Run directory: {exc.result.run_dir}", file=sys.stderr)
    print(f"Verification: {exc.result.verification_path}", file=sys.stderr)
    print(f"Metrics: {exc.result.metrics_path}", file=sys.stderr)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
