"""Legacy demo entrypoint.

Prefer:
    python -m deep_research "your research question"
"""

from deep_research.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
