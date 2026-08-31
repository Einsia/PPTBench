#!/usr/bin/env python3
"""Verify the frozen PPTBench task and result release.

The verifier deliberately uses explicit checks instead of ``assert``.  Release
validation must behave identically when Python is invoked with ``-O`` (which
removes assertions).  The number of published configurations is read from the
leaderboard and is used to derive the expected number of model--task rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TASKSET = ROOT / "configs" / "taskset.txt"
TASKS_ROOT = ROOT / "benchmark" / "tasks"
MATERIALIZATION_MANIFEST = ROOT / "benchmark" / "materialization_manifest.jsonl"
LICENSE_AUDIT = ROOT / "benchmark" / "source_license_audit.jsonl"
RESULTS_ROOT = ROOT / "results"
FROZEN_TASK_COUNT = 500
EXPECTED_TASK_ID_SHA256 = "06f679576d1a5fa6ccfa4f2dd25e0f4a06d0bce83e294bb7772bc134b8e67a4f"
PRIVATE_FIELD_MARKERS = (
    "duration",
    "elapsed",
    "wall_clock",
    "wallclock",
    "endpoint",
    "provider",
    "protocol",
    "trace",
    "run_id",
)


class ReleaseValidationError(ValueError):
    """Raised when a release invariant is not satisfied."""

def _require(condition: bool, message: str) -> None:
    """Raise a stable, user-facing error when a release check fails."""

    if not condition:
        raise ReleaseValidationError(message)


def _task_ids() -> list[str]:
    _require(TASKSET.is_file(), f"missing frozen task list: {TASKSET}")
    task_ids = [line.strip() for line in TASKSET.read_text(encoding="utf-8").splitlines()]
    task_ids = [task_id for task_id in task_ids if task_id]
    _require(task_ids, "frozen task list is empty")
    _require(len(task_ids) == len(set(task_ids)), "frozen task list contains duplicate IDs")
    return task_ids


def _rows_by_key(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row_index, row in enumerate(rows):
        value = row.get(key)
        _require(value is not None and str(value) != "", f"row {row_index} is missing {key}")
        text = str(value)
        counts[text] = counts.get(text, 0) + 1
    return counts


def _load_csv(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"missing release file: {path}")
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReleaseValidationError(f"cannot read {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"missing release file: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReleaseValidationError(
                        f"{path}:{line_number} is not valid JSON: {exc.msg}"
                    ) from exc
                _require(
                    isinstance(value, dict),
                    f"{path}:{line_number} must contain a JSON object",
                )
                rows.append(value)
    except (OSError, UnicodeError) as exc:
        raise ReleaseValidationError(f"cannot read {path}: {exc}") from exc
    return rows


def _index_by_task(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        task_id = row.get("task_id")
        _require(
            task_id is not None and str(task_id) != "",
            f"{label} row {row_index} is missing task_id",
        )
        task_id = str(task_id)
        _require(task_id not in indexed, f"duplicate task_id in {label}: {task_id}")
        indexed[task_id] = row
    return indexed


def _check_task_partition(
    counts: dict[str, int],
    task_ids: set[str],
    configuration_count: int,
    label: str,
) -> None:
    _require(set(counts) == task_ids, f"{label} task IDs differ from frozen set")
    wrong = {task_id: count for task_id, count in counts.items() if count != configuration_count}
    _require(
        not wrong,
        f"{label} must contain one row per configuration for each task; wrong counts: {wrong}",
    )


def _check_no_private_fields(rows: Iterable[dict[str, Any]], label: str) -> None:
    """Reject private execution metadata at every nesting level of release rows."""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                key_folded = key_text.casefold()
                _require(
                    not any(marker in key_folded for marker in PRIVATE_FIELD_MARKERS),
                    f"private run metadata field in {label}: {path}.{key_text}",
                )
                visit(child, f"{path}.{key_text}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    for index, row in enumerate(rows):
        visit(row, f"{label}[{index}]")


def _check_public_candidate_ids(candidate_ids: set[str]) -> None:
    """Ensure identifiers do not carry a third execution-adapter component."""

    _require(bool(candidate_ids), "release contains no candidate configurations")
    _require(
        all(candidate_id.count("__") <= 1 for candidate_id in candidate_ids),
        "candidate IDs must contain model and optional effort only",
    )


def _verify() -> int:
    task_ids = _task_ids()
    task_count = len(task_ids)
    _require(
        task_count == FROZEN_TASK_COUNT,
        f"expected {FROZEN_TASK_COUNT} tasks, found {task_count}",
    )
    digest = hashlib.sha256(("\n".join(task_ids) + "\n").encode("utf-8")).hexdigest()
    _require(digest == EXPECTED_TASK_ID_SHA256, f"frozen task list digest mismatch: {digest}")

    materialization_rows = _index_by_task(
        _load_jsonl(MATERIALIZATION_MANIFEST), "materialization manifest"
    )
    license_rows = _index_by_task(_load_jsonl(LICENSE_AUDIT), "license audit")
    frozen_task_ids = set(task_ids)
    _require(
        set(materialization_rows) == frozen_task_ids,
        "materialization manifest task IDs differ from frozen set",
    )
    _require(set(license_rows) == frozen_task_ids, "license audit task IDs differ from frozen set")

    resource_count = 0
    for task_id in task_ids:
        task_dir = TASKS_ROOT / task_id
        for name in ("metadata.json", "resources/index.json"):
            _require((task_dir / name).is_file(), f"missing {task_id}/{name}")
        _require(not (task_dir / "source.pdf").exists(), f"bundled source asset: {task_id}")
        _require(not (task_dir / "reference.png").exists(), f"bundled reference asset: {task_id}")
        _require(
            not list((task_dir / "resources").glob("resource_*.png")),
            f"bundled resource asset: {task_id}",
        )
        try:
            metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
            resource_index = json.loads(
                (task_dir / "resources" / "index.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReleaseValidationError(f"invalid task metadata for {task_id}: {exc}") from exc
        _require(isinstance(metadata, dict), f"metadata for {task_id} is not an object")
        _require(isinstance(resource_index, dict), f"resource index for {task_id} is not an object")
        _require(metadata.get("task_id") == task_id, f"metadata task_id mismatch: {task_id}")
        specification = materialization_rows[task_id]
        source_spec = specification.get("source")
        reference_spec = specification.get("reference")
        paper_spec = metadata.get("paper")
        _require(isinstance(source_spec, dict), f"source specification is invalid: {task_id}")
        _require(isinstance(reference_spec, dict), f"reference specification is invalid: {task_id}")
        _require(isinstance(paper_spec, dict), f"paper metadata is invalid: {task_id}")
        _require(
            source_spec.get("sha256") == metadata.get("source_sha256"),
            f"source SHA metadata mismatch: {task_id}",
        )
        _require(
            source_spec.get("filename") == metadata.get("source_filename"),
            f"source filename metadata mismatch: {task_id}",
        )
        _require(
            source_spec.get("versioned_id") == paper_spec.get("versioned_id"),
            f"source version metadata mismatch: {task_id}",
        )
        _require(
            reference_spec.get("historic_file_sha256") == metadata.get("reference_sha256"),
            f"reference SHA metadata mismatch: {task_id}",
        )
        _require(
            reference_spec.get("width", 0) > 0 and reference_spec.get("height", 0) > 0,
            f"reference dimensions are invalid: {task_id}",
        )
        resources = specification.get("resources")
        assets = resource_index.get("assets")
        _require(isinstance(resources, list), f"resource specification is invalid: {task_id}")
        _require(isinstance(assets, list), f"resource index assets are invalid: {task_id}")
        _require(
            all(isinstance(item, dict) for item in resources),
            f"resource specification contains a non-object entry: {task_id}",
        )
        _require(
            all(isinstance(item, dict) for item in assets),
            f"resource index contains a non-object entry: {task_id}",
        )
        _require(
            len(resources) == resource_index.get("resource_count"),
            f"resource count mismatch: {task_id}",
        )
        _require(
            all(isinstance(item.get("file"), str) and item.get("file") for item in resources),
            f"resource specification has an invalid file entry: {task_id}",
        )
        _require(
            all(isinstance(item.get("file"), str) and item.get("file") for item in assets),
            f"resource index has an invalid file entry: {task_id}",
        )
        _require(
            [item["file"] for item in resources] == [item["file"] for item in assets],
            f"resource file list mismatch: {task_id}",
        )
        resource_count += len(resources)

    case_rows = _load_jsonl(RESULTS_ROOT / "case-results.jsonl")
    score_rows = _load_csv(RESULTS_ROOT / "scores.csv")
    ranking = _load_csv(RESULTS_ROOT / "leaderboard.csv")
    ranking_ids = [row.get("candidate_id", "") for row in ranking]
    _require(bool(ranking_ids), "leaderboard is empty")
    _require(all(ranking_ids), "leaderboard contains an empty candidate_id")
    leaderboard_ids = set(ranking_ids)
    _require(
        len(leaderboard_ids) == len(ranking_ids),
        "leaderboard contains duplicate candidate IDs",
    )
    _check_public_candidate_ids(leaderboard_ids)
    configuration_count = len(leaderboard_ids)
    expected_case_count = task_count * configuration_count

    _require(
        len(case_rows) == expected_case_count,
        f"case result count mismatch: expected {expected_case_count}, found {len(case_rows)}",
    )
    _require(
        len(score_rows) == expected_case_count,
        f"score result count mismatch: expected {expected_case_count}, found {len(score_rows)}",
    )
    _require(
        all(int(row["task_count"]) == task_count for row in ranking),
        "leaderboard task_count does not match frozen task count",
    )

    case_model_counts = _rows_by_key(case_rows, "candidate_id")
    score_model_counts = _rows_by_key(score_rows, "candidate_id")
    _require(set(case_model_counts) == leaderboard_ids, "case result configurations differ from leaderboard")
    _require(set(score_model_counts) == leaderboard_ids, "score result configurations differ from leaderboard")
    _require(
        all(count == task_count for count in case_model_counts.values()),
        "case results must contain one row per task for each configuration",
    )
    _require(
        all(count == task_count for count in score_model_counts.values()),
        "scores must contain one row per task for each configuration",
    )
    case_pairs = {
        (str(row.get("task_id", "")), str(row.get("candidate_id", ""))) for row in case_rows
    }
    score_pairs = {(row.get("task_id", ""), row.get("candidate_id", "")) for row in score_rows}
    _require(len(case_pairs) == expected_case_count, "duplicate case result pair")
    _require(len(score_pairs) == expected_case_count, "duplicate score pair")
    _require(case_pairs == score_pairs, "case and score pair partitions differ")
    _check_task_partition(
        _rows_by_key(case_rows, "task_id"), frozen_task_ids, configuration_count, "case results"
    )
    _check_task_partition(
        _rows_by_key(score_rows, "task_id"), frozen_task_ids, configuration_count, "scores"
    )

    metrics_rows = _load_csv(RESULTS_ROOT / "generation-metrics.csv")
    _require(
        len(metrics_rows) == configuration_count,
        f"generation metrics count mismatch: expected {configuration_count}, found {len(metrics_rows)}",
    )
    metrics_ids = {row.get("candidate_id", "") for row in metrics_rows}
    _require(metrics_ids == leaderboard_ids, "generation metrics configurations differ from leaderboard")
    _require(
        all(int(row["task_count"]) == task_count for row in metrics_rows),
        "generation metrics task_count does not match frozen task count",
    )
    _check_no_private_fields(
        [*case_rows, *score_rows, *ranking, *metrics_rows], "release result tables"
    )

    bootstrap_path = RESULTS_ROOT / "bootstrap.json"
    _require(bootstrap_path.is_file(), f"missing release file: {bootstrap_path}")
    try:
        bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseValidationError(f"cannot read {bootstrap_path}: {exc}") from exc
    _require(isinstance(bootstrap, dict), "bootstrap.json must contain an object")
    _require(bootstrap.get("task_count") == task_count, "bootstrap task_count mismatch")
    _require(bootstrap.get("model_count") == configuration_count, "bootstrap model_count mismatch")
    full_ranking = bootstrap.get("full_ranking")
    _require(isinstance(full_ranking, list), "bootstrap full_ranking must be a list")
    _require(
        all(isinstance(row, dict) for row in full_ranking),
        "bootstrap full_ranking contains a non-object entry",
    )
    _require(
        len(full_ranking) == configuration_count,
        "bootstrap ranking length differs from published configurations",
    )
    _require(
        all(isinstance(row.get("model"), str) and row.get("model") for row in full_ranking),
        "bootstrap full_ranking contains an invalid model identifier",
    )
    bootstrap_ids = {row["model"] for row in full_ranking}
    _require(bootstrap_ids == leaderboard_ids, "bootstrap configurations differ from leaderboard")
    _check_no_private_fields(full_ranking, "bootstrap ranking")

    print(
        "PPTBench release verified: "
        f"{task_count} downloader-only task specifications, {resource_count} addressed resources, "
        f"{expected_case_count} detailed results, {configuration_count} configurations"
    )
    return 0


def main() -> int:
    try:
        return _verify()
    except (ReleaseValidationError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"PPTBench release verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
