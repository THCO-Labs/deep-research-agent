from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PathSafetyError(ValueError):
    """Raised when a tool tries to access a path outside the run directory."""


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path

    @classmethod
    def create(cls, out_dir: Path, question: str) -> "RunArtifacts":
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        slug = slugify(question)[:64] or "research"
        run_dir = (out_dir / f"{timestamp}-{slug}").resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "findings").mkdir()
        (run_dir / "source_docs").mkdir()
        (run_dir / "sources.jsonl").write_text("", encoding="utf-8")
        (run_dir / "activity.jsonl").write_text("", encoding="utf-8")
        (run_dir / "activity.md").write_text(
            "# Activity Log\n\nVisible research progress. This is not hidden chain-of-thought.\n\n",
            encoding="utf-8",
        )
        return cls(run_dir=run_dir)

    def resolve_path(self, file_path: str | Path) -> Path:
        candidate = _normalize_virtual_path(file_path)
        if candidate.is_absolute():
            raise PathSafetyError("Run artifact paths must be relative.")
        target = (self.run_dir / candidate).resolve()
        try:
            target.relative_to(self.run_dir)
        except ValueError as exc:
            raise PathSafetyError(f"Path escapes run directory: {file_path}") from exc
        return target

    def write_text(self, file_path: str | Path, content: str) -> Path:
        target = self.resolve_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def read_text(self, file_path: str | Path) -> str:
        return self.resolve_path(file_path).read_text(encoding="utf-8")

    def write_json(self, file_path: str | Path, payload: Any) -> Path:
        return self.write_text(file_path, json.dumps(payload, indent=2, sort_keys=True))

    def write_jsonl(self, file_path: str | Path, rows: list[dict[str, Any]]) -> Path:
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        return self.write_text(file_path, text)

    def append_text(self, file_path: str | Path, content: str) -> Path:
        target = self.resolve_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return target

    def append_jsonl(self, file_path: str | Path, row: dict[str, Any]) -> Path:
        return self.append_text(file_path, json.dumps(row, sort_keys=True) + "\n")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def _normalize_virtual_path(file_path: str | Path) -> Path:
    raw = str(file_path)
    candidate = Path(raw)
    if candidate.drive:
        return candidate
    normalized = raw.replace("\\", "/")
    if normalized.startswith("//"):
        return candidate
    if normalized.startswith("/"):
        normalized = normalized.lstrip("/")
    return Path(normalized)
