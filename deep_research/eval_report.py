from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from deep_research.eval import summarize_results


def load_results(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize deep research benchmark results.")
    parser.add_argument("results", type=Path)
    args = parser.parse_args(argv)
    summary = summarize_results(load_results(args.results))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
