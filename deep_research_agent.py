"""Compatibility wrapper for the canonical deep_research CLI."""

from deep_research.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
