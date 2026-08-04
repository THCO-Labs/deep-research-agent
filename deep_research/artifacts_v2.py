from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deep_research.artifacts import RunArtifacts


@dataclass(frozen=True)
class ResearchArtifactsV2:
    artifacts: RunArtifacts

    @classmethod
    def create(cls, out_dir: Path, question: str) -> "ResearchArtifactsV2":
        out_dir_path = Path(out_dir)
        # Treat the path as an existing run dir only if it looks like one
        # (has manifest.json or request.json). A plain existing directory like
        # "runs/" is a parent output directory, not a run dir itself.
        _is_run_dir = out_dir_path.name.startswith("job_") or (
            out_dir_path.exists()
            and (
                (out_dir_path / "manifest.json").exists()
                or (out_dir_path / "request.json").exists()
            )
        )
        if _is_run_dir:
            base = RunArtifacts(run_dir=out_dir_path.resolve())
            base.run_dir.mkdir(parents=True, exist_ok=True)
        else:
            base = RunArtifacts.create(out_dir_path, question)
        for directory in ("documents", "checkpoints", "findings", "source_docs"):
            (base.run_dir / directory).mkdir(exist_ok=True)
        for file_name in (
            "sources.jsonl",
            "evidence_cards.jsonl",
            "activity.jsonl",
        ):
            p = base.run_dir / file_name
            if not p.exists():
                p.write_text("", encoding="utf-8")
        return cls(base)

    @classmethod
    def from_existing(cls, run_dir: Path) -> "ResearchArtifactsV2":
        run_dir = run_dir.resolve()
        for directory in ("documents", "checkpoints", "source_docs"):
            (run_dir / directory).mkdir(exist_ok=True)
        for file_name in ("sources.jsonl", "evidence_cards.jsonl", "activity.jsonl", "activity.md"):
            path = run_dir / file_name
            if not path.exists():
                path.write_text("", encoding="utf-8")
        return cls(RunArtifacts(run_dir=run_dir))

    @property
    def run_dir(self) -> Path:
        return self.artifacts.run_dir

    def write_json(self, file_path: str | Path, payload: Any) -> Path:
        return self.artifacts.write_json(file_path, payload)

    def write_jsonl(self, file_path: str | Path, rows: list[dict[str, Any]]) -> Path:
        return self.artifacts.write_jsonl(file_path, rows)

    def append_jsonl(self, file_path: str | Path, row: dict[str, Any]) -> Path:
        return self.artifacts.append_jsonl(file_path, row)

    def append_text(self, file_path: str | Path, content: str) -> Path:
        return self.artifacts.append_text(file_path, content)

    def write_text(self, file_path: str | Path, content: str) -> Path:
        return self.artifacts.write_text(file_path, content)

    def read_text(self, file_path: str | Path) -> str:
        return self.artifacts.read_text(file_path)

    def resolve_path(self, file_path: str | Path) -> Path:
        return self.artifacts.resolve_path(file_path)

    def read_json(self, file_path: str | Path) -> dict[str, Any]:
        path = self.resolve_path(file_path)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
