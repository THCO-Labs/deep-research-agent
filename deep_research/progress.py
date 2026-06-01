from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any, Callable, Literal

ProgressMode = Literal["live", "raw", "quiet"]
ProgressCallback = Callable[[str], None]


@dataclass
class ActivityLog:
    artifacts: Any
    on_update: ProgressCallback | None = None
    progress_mode: ProgressMode = "live"
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.events:
            self.events.extend(load_activity_events(self.artifacts.run_dir))
        self._write_dashboard()

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
        self.events.append(event)
        self.artifacts.append_jsonl("activity.jsonl", event)
        self.artifacts.append_text("activity.md", f"- `{event['timestamp']}` **{stage}**: {message}\n")
        self._write_dashboard()
        if self.on_update and self.progress_mode == "live":
            self.on_update(progress_line(stage, message))

    def _write_dashboard(self) -> None:
        self.artifacts.write_text(
            "activity.html",
            render_activity_html(self.events, run_name=self.artifacts.run_dir.name),
        )


def progress_line(stage: str, message: str) -> str:
    timestamp = datetime.now().strftime("%H:%M:%S")
    return f"[{timestamp}] {stage}: {message}"


def load_activity_events(run_dir: Path | str) -> list[dict[str, Any]]:
    path = Path(run_dir) / "activity.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def format_activity_summary(
    events: list[dict[str, Any]],
    *,
    run_name: str = "",
    limit: int = 25,
) -> str:
    title = f"Activity: {run_name}" if run_name else "Activity"
    lines = [title, "=" * len(title)]
    if not events:
        lines.append("No activity events yet.")
        return "\n".join(lines)

    counts = Counter(str(event.get("stage", "unknown")) for event in events)
    lines.append(f"Events: {len(events)}")
    lines.append("Stages: " + ", ".join(f"{stage}={count}" for stage, count in sorted(counts.items())))
    lines.append("")
    lines.append("Recent events:")
    for event in events[-limit:]:
        timestamp = str(event.get("timestamp", ""))
        stage = str(event.get("stage", "unknown"))
        message = str(event.get("message", ""))
        lines.append(f"- {timestamp} [{stage}] {message}")
    return "\n".join(lines)


def render_activity_html(events: list[dict[str, Any]], *, run_name: str = "") -> str:
    counts = Counter(str(event.get("stage", "unknown")) for event in events)
    latest = events[-1] if events else None
    latest_text = str(latest.get("message", "Waiting for first event.")) if latest else "Waiting for first event."
    rows = "\n".join(_event_row(event) for event in reversed(events[-250:]))
    stage_chips = "\n".join(
        f'<span class="chip"><span>{escape(stage)}</span><strong>{count}</strong></span>'
        for stage, count in sorted(counts.items())
    )
    if not stage_chips:
        stage_chips = '<span class="chip"><span>waiting</span><strong>0</strong></span>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="5">
  <title>Deep Research Activity</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172018;
      --muted: #5d6a60;
      --line: #d9ded5;
      --surface: #f7f8f4;
      --panel: #ffffff;
      --accent: #0f6f5c;
      --warn: #9a5b00;
      --error: #a6362f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Aptos", "Segoe UI", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f7f8f4 0%, #eef3ea 100%);
    }}
    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }}
    header {{
      display: grid;
      gap: 8px;
      padding-bottom: 22px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(26px, 4vw, 44px);
      line-height: 1.05;
      letter-spacing: 0;
    }}
    .subtle {{ color: var(--muted); margin: 0; max-width: 780px; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin: 22px 0;
    }}
    .chip {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .latest {{
      margin: 0 0 20px;
      padding: 14px 16px;
      border-left: 4px solid var(--accent);
      background: var(--panel);
      box-shadow: 0 1px 0 rgba(0,0,0,.04);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
    }}
    th, td {{
      text-align: left;
      vertical-align: top;
      padding: 11px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      line-height: 1.4;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      background: #f2f5ef;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .stage {{
      display: inline-block;
      min-width: 92px;
      color: var(--accent);
      font-weight: 700;
    }}
    .stage-error {{ color: var(--error); }}
    .stage-retry, .stage-model_fallback {{ color: var(--warn); }}
    code {{
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 12px;
      color: var(--muted);
    }}
    @media (max-width: 720px) {{
      main {{ width: min(100% - 20px, 1120px); padding-top: 20px; }}
      th:nth-child(1), td:nth-child(1) {{ display: none; }}
      .stage {{ min-width: 72px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <p class="subtle">Auto-refreshes every 5 seconds</p>
      <h1>Deep Research Activity</h1>
      <p class="subtle">{escape(run_name or "Current run")}</p>
      <p class="subtle">This dashboard shows observable research actions, tool progress, verification status, and model fallback events. It does not expose hidden chain-of-thought.</p>
    </header>
    <section class="stats" aria-label="Activity counts">
      <span class="chip"><span>Total events</span><strong>{len(events)}</strong></span>
      {stage_chips}
    </section>
    <p class="latest"><strong>Latest:</strong> {escape(latest_text)}</p>
    <table>
      <thead><tr><th>Time</th><th>Stage</th><th>Message</th></tr></thead>
      <tbody>
        {rows or '<tr><td colspan="3">No activity events yet.</td></tr>'}
      </tbody>
    </table>
  </main>
</body>
</html>
"""


def _event_row(event: dict[str, Any]) -> str:
    timestamp = escape(str(event.get("timestamp", "")))
    stage = str(event.get("stage", "unknown"))
    message = escape(str(event.get("message", "")))
    stage_class = "stage stage-" + re.sub(r"[^a-z0-9_-]+", "-", stage.lower())
    return (
        "<tr>"
        f"<td><code>{timestamp}</code></td>"
        f'<td><span class="{stage_class}">{escape(stage)}</span></td>'
        f"<td>{message}</td>"
        "</tr>"
    )


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
