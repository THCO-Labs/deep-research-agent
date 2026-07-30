from __future__ import annotations

from typing import Any


def _format_proxy_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Count: {summary.get('count', 0)}",
        f"Successful: {summary.get('successful_count', 0)}",
        f"Errors: {summary.get('error_count', 0)}",
        f"Overall Score: {float(summary.get('overall_score', 0.0)):.4f}",
        f"RACE Relative Score: {float(summary.get('race_relative_score', summary.get('overall_score', 0.0))):.4f}",
        f"Candidate Absolute Score: {float(summary.get('candidate_absolute_score', 0.0)):.4f}",
        f"Reference Absolute Score: {float(summary.get('reference_absolute_score', 0.0)):.4f}",
        f"Topic Focus: {float(summary.get('topic_focus_score', 0.0)):.4f}",
        f"Comprehensiveness: {float(summary.get('comprehensiveness', 0.0)):.4f}",
        f"Insight: {float(summary.get('insight', 0.0)):.4f}",
        f"Instruction Following: {float(summary.get('instruction_following', 0.0)):.4f}",
        f"Readability: {float(summary.get('readability', 0.0)):.4f}",
        f"Method Counts: {summary.get('method_counts', {})}",
        f"Low Score IDs: {summary.get('low_score_ids', [])}",
    ]
    return "\n".join(lines) + "\n"


def _format_fact_proxy_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Count: {summary.get('count', 0)}",
        f"Successful: {summary.get('successful_count', 0)}",
        f"Errors: {summary.get('error_count', 0)}",
        f"Overall Score: {float(summary.get('overall_score', 0.0)):.4f}",
        f"Valid Rate: {float(summary.get('valid_rate', 0.0)):.4f}",
        f"Source List Consistency: {float(summary.get('source_list_consistency_score', 0.0)):.4f}",
        f"Prompt Overlap: {float(summary.get('prompt_overlap_score', 0.0)):.4f}",
        f"Topic Focus: {float(summary.get('topic_focus_score', 0.0)):.4f}",
        f"Topic Consistency: {float(summary.get('topic_consistency_score', 0.0)):.4f}",
        f"Source Breadth: {float(summary.get('source_breadth_score', 0.0)):.4f}",
        f"Supported Citations: {summary.get('supported_citation_count', 0)}",
        f"Unsupported Citations: {summary.get('unsupported_citation_count', 0)}",
        f"Unknown Citations: {summary.get('unknown_citation_count', 0)}",
        f"Invalid Citations: {summary.get('invalid_citation_count', 0)}",
        f"Low Score IDs: {summary.get('low_score_ids', [])}",
    ]
    return "\n".join(lines) + "\n"




