from __future__ import annotations

import argparse
import time
from pathlib import Path

from deep_research.runtime.progress import format_activity_summary, load_activity_events, render_activity_html


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View a deep research run's visible activity log.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Run directory containing activity.jsonl. Omit to use the latest run under --out.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs"),
        help="Runs directory to search when run_dir is omitted or --latest is used.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Watch the newest run under --out instead of passing a run directory.",
    )
    parser.add_argument("--limit", type=int, default=25, help="Number of recent events to print.")
    parser.add_argument("--follow", action="store_true", help="Refresh the terminal view until interrupted.")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval for --follow.")
    parser.add_argument("--html", action="store_true", help="Regenerate activity.html and print its path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = resolve_run_dir(args.run_dir, out_dir=args.out, latest=args.latest)

    if args.html:
        events = load_activity_events(run_dir)
        html_path = run_dir / "activity.html"
        html_path.write_text(render_activity_html(events, run_name=run_dir.name), encoding="utf-8")
        print(f"Activity dashboard: {html_path}")
        return 0

    while True:
        events = load_activity_events(run_dir)
        print(format_activity_summary(events, run_name=run_dir.name, limit=args.limit))
        if not args.follow:
            return 0
        print("\nRefreshing. Press Ctrl+C to stop.\n")
        try:
            time.sleep(max(args.interval, 0.5))
        except KeyboardInterrupt:
            return 0


def resolve_run_dir(
    run_dir: Path | None,
    *,
    out_dir: Path,
    latest: bool = False,
) -> Path:
    if run_dir is not None and latest:
        raise SystemExit("Pass either run_dir or --latest, not both.")
    resolved = find_latest_run(out_dir) if run_dir is None or latest else run_dir.resolve()
    if not resolved.exists():
        raise SystemExit(f"Run directory does not exist: {resolved}")
    if not (resolved / "activity.jsonl").exists():
        raise SystemExit(f"Missing activity.jsonl in: {resolved}")
    return resolved


def find_latest_run(out_dir: Path) -> Path:
    root = out_dir.resolve()
    if not root.exists():
        raise SystemExit(f"Runs directory does not exist: {root}")
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "activity.jsonl").exists()
    ]
    if not candidates:
        raise SystemExit(f"No runs with activity.jsonl found in: {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


if __name__ == "__main__":
    raise SystemExit(main())
