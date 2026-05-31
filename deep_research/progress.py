from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Literal

ProgressMode = Literal["live", "raw", "quiet"]
ProgressCallback = Callable[[str], None]


@dataclass
class ActivityLog:
    artifacts: Any
    on_update: ProgressCallback | None = None
    progress_mode: ProgressMode = "live"

    def emit(
        self,
        stage: str,
        message: str,
        *,
        kind: str = "status",
        data: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "stage": stage,
            "kind": kind,
            "message": message,
        }
        if data:
            event["data"] = data
        self.artifacts.append_jsonl("activity.jsonl", event)
        self.artifacts.append_text("activity.md", f"- `{event['timestamp']}` **{stage}**: {message}\n")
        if self.on_update and self.progress_mode == "live":
            self.on_update(progress_line(stage, message))


def progress_line(stage: str, message: str) -> str:
    timestamp = datetime.now().strftime("%H:%M:%S")
    return f"[{timestamp}] {stage}: {message}"


def summarize_stream_update(node: str, content: Any) -> str | None:
    event = summarize_stream_event(node, content)
    if event is None:
        return None
    stage, message = event
    return progress_line(stage, message)


def summarize_stream_event(node: str, content: Any) -> tuple[str, str] | None:
    text = _content_to_text(content)
    if not text:
        return None
    text = _collapse(text)

    if node == "tools":
        if _is_verbose_tool_payload(text):
            return None
        if text.startswith("Wrote "):
            return None
        if text.startswith("ERROR:"):
            return "tool", text[:240]
        return "tool", _shorten(text, 180)

    if node == "model":
        return "agent", _shorten(text, 240)

    return node, _shorten(text, 200)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _collapse(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _is_verbose_tool_payload(text: str) -> bool:
    if text.startswith('{"query":') or text.startswith("{'query':"):
        return True
    if text.startswith('{"source_id":') or text.startswith("{'source_id':"):
        return True
    if '"markdown"' in text or "'markdown'" in text:
        return True
    if '"excerpt"' in text or "'excerpt'" in text:
        return True
    if '"results"' in text and '"canonical_url"' in text:
        return True
    return False
