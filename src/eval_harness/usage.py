"""Build auditable token ledgers from agent invocation records.

Codex reports cumulative usage for a resumed thread.  This module preserves
that reported value while deriving the incremental usage attributable to each
physical invocation.  Claude Code reports per-query incremental usage instead;
the invocation contract records that distinction explicitly.  All aggregation
is based on incremental counters so resumed turns are not charged twice.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any


COUNTER_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)
DERIVED_FIELDS = ("uncached_input_tokens", "total_tokens")
USAGE_FIELDS = (*COUNTER_FIELDS, *DERIVED_FIELDS)
FULL_PIPELINE_STAGES = frozenset({"G0"})
_TURN_DIRECTORY = re.compile(r"^generator-(\d+)(?:[-_].*)?$")


def _integer_counter(value: Any, *, field: str) -> int:
    """Normalize one non-negative token counter without accepting booleans."""

    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer token counter")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and value.is_integer():
        result = int(value)
    else:
        raise ValueError(f"{field} must be an integer token counter")
    if result < 0:
        raise ValueError(f"{field} cannot be negative")
    return result


def normalize_usage(value: Mapping[str, Any] | None) -> dict[str, int]:
    """Normalize reported counters and add non-overlapping derived counters.

    ``cached_input_tokens`` is a subset of ``input_tokens`` and
    ``reasoning_output_tokens`` is a subset of ``output_tokens``.  Therefore
    ``total_tokens`` is input plus output only; reasoning tokens are retained as
    an informational counter and are never added a second time.
    """

    raw = value or {}
    counters = {field: _integer_counter(raw.get(field, 0), field=field) for field in COUNTER_FIELDS}
    if counters["cached_input_tokens"] > counters["input_tokens"]:
        raise ValueError("cached_input_tokens cannot exceed input_tokens")
    counters["uncached_input_tokens"] = counters["input_tokens"] - counters["cached_input_tokens"]
    counters["total_tokens"] = counters["input_tokens"] + counters["output_tokens"]
    return counters


def _has_reported_usage(invocation: Mapping[str, Any]) -> bool:
    usage = invocation.get("usage")
    return isinstance(usage, Mapping) and any(field in usage for field in COUNTER_FIELDS)


def _subtract_cumulative_usage(
    current: Mapping[str, int],
    previous: Mapping[str, int],
) -> dict[str, int]:
    """Return the per-call delta between two cumulative thread reports."""

    delta: dict[str, int] = {}
    for field in COUNTER_FIELDS:
        value = int(current[field]) - int(previous[field])
        if value < 0:
            raise ValueError(f"cumulative {field} decreased in a resumed thread")
        delta[field] = value
    return normalize_usage(delta)


def _path_value(invocation: Mapping[str, Any]) -> str:
    for key in ("invocation_path", "path", "_path"):
        value = invocation.get(key)
        if value:
            return str(value)
    return ""


def _infer_stage_and_role(invocation: Mapping[str, Any]) -> tuple[str, str]:
    explicit_stage = str(invocation.get("stage", "")).strip().upper()
    explicit_role = str(invocation.get("role", "")).strip().lower()
    if explicit_stage:
        if explicit_role:
            return explicit_stage, explicit_role
        return explicit_stage, "generator"

    path_value = _path_value(invocation)
    directory = Path(path_value).parent.name if path_value else ""
    match = _TURN_DIRECTORY.fullmatch(directory)
    if not match:
        raise ValueError("each invocation requires stage/role metadata or a turn-directory path")
    raw_index = match.group(1)
    return f"G{int(raw_index)}", "generator"


def _invocation_succeeded(invocation: Mapping[str, Any]) -> bool:
    return (
        invocation.get("returncode") == 0
        and not bool(invocation.get("timed_out", False))
        and not bool(invocation.get("error_events", []))
        and not bool(invocation.get("json_parse_errors", []))
    )


def build_usage_ledger(
    invocations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build one auditable row per physical invocation.

    The input sequence must be in invocation order.  A resumed invocation with
    ``usage_semantics=cumulative`` is differenced against the most recent report
    for that thread; ``incremental`` reports are used directly.  Failed calls
    and retries remain separate rows.  Calls without a usage report are retained
    with ``usage_missing=true`` rather than disappearing from the cost audit.
    """

    previous_by_thread: dict[str, dict[str, int]] = {}
    attempts_by_stage: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for sequence, invocation in enumerate(invocations, start=1):
        stage, role = _infer_stage_and_role(invocation)
        attempts_by_stage[stage] = attempts_by_stage.get(stage, 0) + 1
        attempt_value = invocation.get("attempt", attempts_by_stage[stage])
        attempt = _integer_counter(attempt_value, field="attempt")
        if attempt < 1:
            raise ValueError("attempt must be positive")

        thread_id = str(invocation.get("thread_id", "")).strip()
        session_mode = str(invocation.get("session_mode", "start")).strip().lower()
        if session_mode not in {"start", "resume"}:
            raise ValueError(f"unsupported session_mode: {session_mode}")
        usage_semantics = str(invocation.get("usage_semantics", "cumulative")).strip().lower()
        if usage_semantics not in {"cumulative", "incremental"}:
            raise ValueError(f"unsupported usage_semantics: {usage_semantics}")
        usage_missing = not _has_reported_usage(invocation)
        reported_usage: dict[str, int] | None = None
        incremental_usage: dict[str, int] | None = None
        delta_missing = False
        if not usage_missing:
            usage = invocation.get("usage")
            if not isinstance(usage, Mapping):
                raise ValueError("reported usage must be a mapping")
            reported_usage = normalize_usage(usage)
            if session_mode == "resume" and usage_semantics == "cumulative":
                previous = previous_by_thread.get(thread_id)
                if not thread_id or previous is None:
                    delta_missing = True
                else:
                    incremental_usage = _subtract_cumulative_usage(
                        reported_usage,
                        previous,
                    )
            else:
                incremental_usage = dict(reported_usage)
            if thread_id and usage_semantics == "cumulative":
                previous_by_thread[thread_id] = dict(reported_usage)

        row = {
            "sequence": sequence,
            "stage": stage,
            "role": role,
            "attempt": attempt,
            "status": "succeeded" if _invocation_succeeded(invocation) else "failed",
            "returncode": invocation.get("returncode"),
            "timed_out": bool(invocation.get("timed_out", False)),
            "thread_id": thread_id,
            "session_mode": session_mode,
            "resume_session_id": invocation.get("resume_session_id"),
            "usage_semantics": usage_semantics,
            "model": str(invocation.get("model", "")),
            "reasoning_effort": str(invocation.get("reasoning_effort", "")),
            "invocation_path": _path_value(invocation),
            "invocation_record_state": str(invocation.get("invocation_record_state", "present")),
            "recovered_from_artifacts": bool(invocation.get("recovered_from_artifacts", False)),
            "recovery_sources": list(invocation.get("recovery_sources") or []),
            "physical_call_evidence": list(invocation.get("physical_call_evidence") or []),
            "launch_state": str(invocation.get("launch_state", "confirmed")),
            "physical_call_confirmed": bool(invocation.get("physical_call_confirmed", True)),
            "usage_missing": usage_missing,
            "incremental_usage_missing": delta_missing,
            "reported_usage": reported_usage,
            "incremental_usage": incremental_usage,
        }
        rows.append(row)
    return rows


def ledger_from_invocation_paths(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load an ordered list of ``invocation.json`` files and build ledger rows."""

    invocations: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value).resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object: {path}")
        invocations.append({**value, "invocation_path": str(path)})
    return build_usage_ledger(invocations)


def _zero_usage() -> dict[str, int]:
    return {field: 0 for field in USAGE_FIELDS}


def _sum_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    total = _zero_usage()
    for row in rows:
        usage = row.get("incremental_usage")
        if not isinstance(usage, Mapping):
            continue
        for field in USAGE_FIELDS:
            total[field] += int(usage.get(field, 0))
    return total


def aggregate_usage_ledger(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate incremental usage by stage, role, and benchmark cost view."""

    stages = sorted({str(row.get("stage", "")) for row in rows if row.get("stage")})
    roles = sorted({str(row.get("role", "")) for row in rows if row.get("role")})
    missing = [
        row
        for row in rows
        if bool(row.get("usage_missing")) or bool(row.get("incremental_usage_missing"))
    ]
    return {
        "schema": "pptbench-token-usage-v1",
        "call_count": len(rows),
        "physical_call_confirmed_count": sum(
            bool(row.get("physical_call_confirmed", True)) for row in rows
        ),
        "launch_unconfirmed_count": sum(
            not bool(row.get("physical_call_confirmed", True)) for row in rows
        ),
        "succeeded_call_count": sum(row.get("status") == "succeeded" for row in rows),
        "failed_call_count": sum(row.get("status") != "succeeded" for row in rows),
        "usage_missing_call_count": len(missing),
        "usage_complete": not missing,
        "by_stage": {
            stage: _sum_rows(row for row in rows if row.get("stage") == stage) for stage in stages
        },
        "by_role": {
            role: _sum_rows(row for row in rows if row.get("role") == role) for role in roles
        },
        "overall": _sum_rows(rows),
        "single_run": _sum_rows(row for row in rows if row.get("stage") == "G0"),
        "full": _sum_rows(row for row in rows if row.get("stage") in FULL_PIPELINE_STAGES),
    }
