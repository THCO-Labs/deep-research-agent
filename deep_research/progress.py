from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Callable, Literal

ProgressMode = Literal["live", "raw", "quiet"]
ProgressCallback = Callable[[str], None]


def progress_line(stage: str, message: str) -> str:
    timestamp = datetime.now().strftime("%H:%M:%S")
    return f"[{timestamp}] {stage}: {message}"


def summarize_stream_update(node: str, content: Any) -> str | None:
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
            return progress_line("tool", text[:240])
        return progress_line("tool", _shorten(text, 180))

    if node == "model":
        return progress_line("agent", _shorten(text, 240))

    return progress_line(node, _shorten(text, 200))


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
