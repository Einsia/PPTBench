"""Audit the frozen task package and single-run reconstruction outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from PIL import Image

from arxiv_flow_dataset.task_materializer import pixel_sha256
from eval_harness.image_similarity_gate import assess_raster_gate
from eval_harness.pptx import PptxError, audit_pptx_resource_usage, extract_pptx_components
from eval_harness.paths import default_project_root
from eval_harness.usage import aggregate_usage_ledger
from .check_pptx_bounds import check_deck
from .collect_rollouts import _load_complete_results


RESOURCE_INDEX_SCHEMAS = {
    "pptbench-task-resources-v1",
    "pptbench-task-resources-v2",
}
MAXIMUM_RESOURCES_PER_TASK = 12
ROLLOUT_PROTOCOL = "image-only-single-run"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _task_root(project_root: Path) -> Path:
    """Return the materialized task root used by this checkout."""

    candidates = (
        project_root / "data" / "pptbench-tasks",
        project_root / "tasks",
        project_root / "benchmark" / "tasks",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _task_ids(project_root: Path, limit: int | None = None) -> list[str]:
    """Load frozen task IDs, falling back to sorted materialized directories.

    The frozen set is intentionally non-contiguous, so callers must not infer
    membership from ``task_0000..task_N``.  A numeric prefix remains the final
    fallback for an output-only working directory that has neither a taskset
    file nor materialized task directories.
    """

    taskset_candidates = (
        project_root / "configs" / "taskset.txt",
        project_root / "benchmark" / "configs" / "taskset.txt",
    )
    for taskset_path in taskset_candidates:
        if not taskset_path.is_file():
            continue
        try:
            lines = taskset_path.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            continue
        task_ids = [
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
        # Ignore malformed/path-like entries rather than allowing a taskset
        # typo to escape the intended task root.
        task_ids = [
            task_id
            for task_id in task_ids
            if re.fullmatch(r"task_[A-Za-z0-9_-]+", task_id)
        ]
        if limit is not None:
            return task_ids[: max(0, limit)]
        return task_ids

    root = _task_root(project_root)
    task_ids = sorted(path.name for path in root.glob("task_*") if path.is_dir())
    if task_ids:
        return task_ids[: max(0, limit)] if limit is not None else task_ids
    if limit is None:
        return []
    return [f"task_{index:04d}" for index in range(max(0, limit))]


def _audit_pptx_resources(
    project_root: Path,
    task_id: str,
    pptx_path: Path,
) -> dict[str, Any]:
    """Audit approved resources and the image-similarity diagnostics."""

    task_root = _task_root(project_root) / task_id
    reference_path = task_root / "reference.png"
    resource_paths = sorted(
        (task_root / "resources").glob("resource_*.png")
    )
    try:
        report = extract_pptx_components(pptx_path)
        resource_audit = audit_pptx_resource_usage(report, resource_paths)
        approved_hashes = [
            str(row.get("sha256", ""))
            for row in resource_audit.get("approved_resources", [])
            if isinstance(row, dict) and row.get("sha256")
        ]
        if reference_path.is_file():
            similarity_gate = assess_raster_gate(
                reference_path,
                None,
                report,
                approved_resource_hashes=approved_hashes,
            )
        else:
            # Release audits can be run against an output-only checkout.  Keep
            # the resource audit useful and make the missing reference explicit
            # rather than silently claiming that the similarity gate passed.
            similarity_gate = {
                "status": "not_run",
                "reason": "reference image was not available",
                "triggered": False,
                "violations": [],
            }
        resource_audit["image_similarity_gate"] = similarity_gate
        resource_audit["violations"].extend(similarity_gate.get("violations", []))
        if similarity_gate.get("status") == "not_run":
            if resource_audit["status"] == "complete":
                resource_audit["status"] = "incomplete"
        elif similarity_gate.get("triggered"):
            resource_audit["status"] = "invalid"
        return resource_audit
    except (OSError, PptxError) as exc:
        return {
            "schema": "pptbench-pptx-resource-integrity",
            "status": "invalid",
            "image_similarity_gate": {
                "status": "not_run",
                "reason": "PPTX component extraction failed",
                "triggered": False,
                "violations": [],
            },
            "violations": [f"cannot audit PPTX raster resources: {exc}"],
        }


def _audit_task_resources(task_dir: Path, metadata: dict[str, Any]) -> tuple[int, list[str]]:
    """Validate one task's approved resource package from disk."""

    task_id = task_dir.name
    resource_root = task_dir / "resources"
    index_path = resource_root / "index.json"
    issues: list[str] = []
    if not index_path.is_file():
        return 0, [f"{task_id} is missing resources/index.json"]
    try:
        index = _json(index_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return 0, [f"{task_id} resource index is invalid: {exc}"]
    if index.get("schema") not in RESOURCE_INDEX_SCHEMAS:
        issues.append(f"{task_id} resource index has the wrong schema")
    if index.get("task_id") != task_id:
        issues.append(f"{task_id} resource index has the wrong task_id")
    if index.get("candidate_id") != metadata.get("candidate_id"):
        issues.append(f"{task_id} resource index has the wrong candidate_id")
    assets = index.get("assets")
    if not isinstance(assets, list):
        return 0, [*issues, f"{task_id} resource index assets is not a list"]
    resource_count = len(assets)
    if index.get("resource_count") != resource_count:
        issues.append(f"{task_id} resource_count does not match assets")
    if resource_count > MAXIMUM_RESOURCES_PER_TASK:
        issues.append(
            f"{task_id} has {resource_count} resources; maximum is {MAXIMUM_RESOURCES_PER_TASK}"
        )
    expected_names = [f"resource_{number:04d}.png" for number in range(1, resource_count + 1)]
    indexed_names = [
        str(asset.get("file", "")) if isinstance(asset, dict) else "" for asset in assets
    ]
    if indexed_names != expected_names:
        issues.append(f"{task_id} resource filenames are not contiguous")
    actual_names = sorted(path.name for path in resource_root.glob("resource_*.png"))
    if actual_names != expected_names:
        issues.append(f"{task_id} resource files do not match the index")
    for expected_name, asset in zip(expected_names, assets, strict=True):
        if not isinstance(asset, dict):
            issues.append(f"{task_id} {expected_name} index entry is not an object")
            continue
        path = resource_root / expected_name
        if not path.is_file():
            continue
        if _sha256(path) != asset.get("sha256"):
            issues.append(f"{task_id} {expected_name} SHA-256 mismatch")
        try:
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG":
                    issues.append(f"{task_id} {expected_name} is not a PNG")
                if image.width != asset.get("width") or image.height != asset.get("height"):
                    issues.append(f"{task_id} {expected_name} dimensions mismatch")
                if ("A" in image.getbands()) != asset.get("has_alpha"):
                    issues.append(f"{task_id} {expected_name} alpha metadata mismatch")
                expected_pixels = str(asset.get("pixel_sha256", ""))
                if expected_pixels and pixel_sha256(image) != expected_pixels:
                    issues.append(f"{task_id} {expected_name} pixel SHA-256 mismatch")
        except (OSError, ValueError) as exc:
            issues.append(f"{task_id} {expected_name} cannot be decoded: {exc}")
    return resource_count, issues


def _command_image_paths(command: list[str]) -> tuple[list[Path], list[str]]:
    images: list[Path] = []
    issues: list[str] = []
    index = 0
    while index < len(command):
        argument = command[index]
        if argument == "--image":
            if index + 1 >= len(command):
                issues.append("command has a dangling --image option")
                break
            images.append(Path(command[index + 1]))
            index += 2
            continue
        if argument.startswith("--image="):
            value = argument.partition("=")[2]
            if value:
                images.append(Path(value))
            else:
                issues.append("command has an empty --image option")
        index += 1
    return images, issues


def _agent_input_evidence_violations(task_id: str, result: dict[str, Any]) -> list[str]:
    """Check that the single-run agent saw the image contract, not source PDFs."""

    ledger_value = str(result.get("token_ledger_path", "")).strip()
    if not ledger_value:
        return ["cannot locate the agent workspace without a token ledger path"]
    agent_run = Path(ledger_value).parent
    host_run = agent_run.parent.parent
    workspace = agent_run.parent / "workspace"
    violations: list[str] = []
    try:
        run_config = _json(host_run / "run-config.json")
    except (OSError, ValueError, json.JSONDecodeError):
        violations.append("run config is missing or invalid")
    else:
        if run_config.get("include_caption") is not False:
            violations.append("run config does not enforce image-only input")
    if not workspace.is_dir():
        return [*violations, "agent workspace is missing"]
    exposed_pdfs = [
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".pdf"
    ]
    if exposed_pdfs:
        violations.append("agent workspace contains PDF files: " + ", ".join(exposed_pdfs))

    turns = result.get("turns")
    if not isinstance(turns, list) or len(turns) != 1 or not isinstance(turns[0], dict):
        return [*violations, "result does not contain exactly one agent turn"]
    turn = turns[0]
    if (turn.get("role"), turn.get("turn")) != ("generator", 0):
        violations.append("turn is not the initial reconstruction turn")
    expected_command = [Path("/work/reference.png")]
    artifacts_value = str(turn.get("turn_artifacts", "")).strip()
    if not artifacts_value:
        return [*violations, "turn has no artifact evidence"]
    artifacts = Path(artifacts_value)
    try:
        prompt = (artifacts / "prompt.md").read_text(encoding="utf-8")
    except OSError:
        violations.append("turn prompt evidence is missing")
    else:
        if "pdf" in prompt.casefold():
            violations.append("turn prompt mentions a PDF")
    try:
        command_value = json.loads((artifacts / "command.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        violations.append("turn command evidence is invalid")
        command_value = []
    if not isinstance(command_value, list) or not all(
        isinstance(argument, str) for argument in command_value
    ):
        violations.append("turn command evidence is not a list of strings")
        command_value = []
    command = list(command_value)
    if any(".pdf" in argument.casefold() for argument in command):
        violations.append("turn command contains a PDF argument")
    observed, image_issues = _command_image_paths(command)
    violations.extend(f"turn {issue}" for issue in image_issues)
    if observed != expected_command:
        violations.append("turn attachments do not contain exactly /work/reference.png")

    invocation_value = str(turn.get("invocation_path", "")).strip()
    try:
        invocation = _json(Path(invocation_value)) if invocation_value else {}
    except (OSError, ValueError, json.JSONDecodeError):
        invocation = {}
    isolation = invocation.get("isolation")
    expected_isolation = {
        "filesystem": "bubblewrap-sparse",
        "process_view": "private-proc-with-opaque-bind-sources",
        "evaluator_sources_mounted": False,
        "session_namespace": "generator",
        "identity_blind": True,
        "workspace_alias": "/work",
        "workspace_access": "read-write",
        "output_alias": "/turn-output",
    }
    if not isinstance(isolation, dict):
        violations.append("turn lacks filesystem isolation evidence")
    else:
        for field, expected in expected_isolation.items():
            if isolation.get(field) != expected:
                violations.append(f"turn isolation {field} is not {expected!r}")
    report_path = workspace / "current-pptx-components.json"
    try:
        report = _json(report_path)
    except (OSError, ValueError, json.JSONDecodeError):
        violations.append("agent-visible component report is missing or invalid")
    else:
        if report.get("source_path") != "/work/reconstruction.pptx":
            violations.append("agent-visible component report has a non-anonymous source_path")
        serialized = json.dumps(report, ensure_ascii=False).casefold()
        if "/mnt/" in serialized or task_id.casefold() in serialized:
            violations.append("agent-visible component report leaks host/task identity")
    return violations


def audit_tasks(project_root: Path, *, expected_count: int = 500) -> dict[str, Any]:
    """Validate task numbering, materialization records, resources, and PDFs."""

    task_root = _task_root(project_root)
    expected_ids = _task_ids(project_root, expected_count)
    actual_ids = sorted(path.name for path in task_root.glob("task_*"))
    violations: list[str] = []
    if len(expected_ids) != expected_count:
        violations.append(
            f"taskset contains {len(expected_ids)} usable IDs; expected {expected_count}"
        )
    if actual_ids != sorted(expected_ids):
        violations.append("task IDs do not match the frozen taskset")
    paper_ids: list[str] = []
    resource_counts: list[int] = []
    validated = 0
    for task_id in expected_ids:
        task_dir = task_root / task_id
        required = {
            "metadata": task_dir / "metadata.json",
            "reference": task_dir / "reference.png",
            "source": task_dir / "source.pdf",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            violations.append(f"{task_id} is missing: {', '.join(missing)}")
            continue
        metadata = _json(required["metadata"])
        if metadata.get("task_id") != task_id:
            violations.append(f"{task_id} metadata contains the wrong task_id")
        paper_id = str(metadata.get("arxiv_id", "")).strip()
        if not paper_id:
            violations.append(f"{task_id} metadata has no arxiv_id")
        paper_ids.append(paper_id)
        if _sha256(required["reference"]) != metadata.get("reference_sha256"):
            violations.append(f"{task_id} reference SHA-256 mismatch")
        if _sha256(required["source"]) != metadata.get("source_sha256"):
            violations.append(f"{task_id} source SHA-256 mismatch")
        try:
            with Image.open(required["reference"]) as image:
                image.load()
                if image.width < 1 or image.height < 1:
                    violations.append(f"{task_id} reference has invalid dimensions")
        except (OSError, ValueError) as exc:
            violations.append(f"{task_id} reference cannot be decoded: {exc}")
        if required["source"].read_bytes()[:5] != b"%PDF-":
            violations.append(f"{task_id} source is not a PDF")
        resource_count, resource_issues = _audit_task_resources(task_dir, metadata)
        resource_counts.append(resource_count)
        violations.extend(resource_issues)
        validated += 1
    manifest_path = task_root / "manifest.jsonl"
    if not manifest_path.is_file():
        violations.append("task manifest is missing")
        manifest_lines: list[str] = []
    else:
        manifest_lines = [
            line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    if len(manifest_lines) != expected_count:
        violations.append("task manifest does not contain the expected number of rows")
    paper_counts = Counter(paper_ids)
    if max(paper_counts.values(), default=0) > 1:
        violations.append("a paper contributes more than one task")
    return {
        "schema": "pptbench-task-audit",
        "expected_tasks": expected_count,
        "validated_tasks": validated,
        "manifest_rows": len(manifest_lines),
        "unique_papers": len(paper_counts),
        "papers_by_task_count": {
            str(count): sum(value == count for value in paper_counts.values())
            for count in sorted(set(paper_counts.values()))
        },
        "total_resources": sum(resource_counts),
        "resource_task_count": sum(count > 0 for count in resource_counts),
        "max_resources_per_task": max(resource_counts, default=0),
        "violations": violations,
        "status": "complete" if not violations else "invalid",
    }


def _token_usage_violations(result: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    ledger_value = str(result.get("token_ledger_path", "")).strip()
    summary_value = str(result.get("usage_summary_path", "")).strip()
    if not ledger_value or not Path(ledger_value).is_file():
        violations.append("token ledger path is missing")
    if not summary_value or not Path(summary_value).is_file():
        violations.append("usage summary path is missing")
    if violations:
        return violations
    try:
        ledger = json.loads(Path(ledger_value).read_text(encoding="utf-8"))
        summary = json.loads(Path(summary_value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"token accounting JSON is invalid: {exc}"]
    if not isinstance(ledger, list) or not all(isinstance(row, dict) for row in ledger):
        return ["token ledger is not a list of objects"]
    if not isinstance(summary, dict):
        return ["usage summary is not an object"]
    if len(ledger) != 1 or ledger[0].get("stage") != "G0":
        violations.append("token ledger does not contain exactly one initial call")
    try:
        recomputed = aggregate_usage_ledger(ledger)
    except (TypeError, ValueError) as exc:
        violations.append(f"token ledger cannot be aggregated: {exc}")
    else:
        if summary != recomputed:
            violations.append("usage summary does not match the token ledger")
    if summary.get("usage_complete") is not True:
        violations.append("usage summary is not complete")
    if result.get("usage") != summary:
        violations.append("result usage does not match usage-summary.json")
    return violations


def _rollout_contract_violations(
    project_root: Path,
    task_id: str,
    result: dict[str, Any],
) -> list[str]:
    """Validate one single-run result and its produced artifacts."""

    violations: list[str] = []
    if result.get("target") != "pptx":
        violations.append("result target is not pptx")
    if result.get("protocol") != ROLLOUT_PROTOCOL:
        violations.append(f"result protocol is not {ROLLOUT_PROTOCOL}")
    if result.get("harness") != "single-run":
        violations.append("result harness is not single-run")
    for key in ("final_pptx_path", "final_screenshot_path", "final_components_path"):
        value = str(result.get(key, "")).strip()
        if not value or not Path(value).is_file():
            violations.append(f"{key} is missing")
    metrics = result.get("final_metrics")
    if not isinstance(metrics, dict):
        violations.append("final metrics are missing")
    pptx_value = str(result.get("final_pptx_path", "")).strip()
    if pptx_value and Path(pptx_value).is_file():
        resource_audit = _audit_pptx_resources(project_root, task_id, Path(pptx_value))
        violations.extend(
            f"PPTX resource integrity: {issue}"
            for issue in resource_audit.get("violations", [])
        )
        violations.extend(check_deck(Path(pptx_value)))
    violations.extend(_agent_input_evidence_violations(task_id, result))
    violations.extend(_token_usage_violations(result))
    return violations


def audit_rollouts(
    project_root: Path,
    *,
    expected_tasks: int = 20,
    agent: str = "codex",
    model: str = "gpt-5.6-sol",
    reasoning_effort: str = "medium",
) -> dict[str, Any]:
    """Validate current-input single-run results for a bounded task prefix."""

    task_ids = _task_ids(project_root, expected_tasks)
    current_results = _load_complete_results(
        project_root,
        task_ids=task_ids,
        agent=agent,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    complete: dict[str, Path] = {}
    missing: list[str] = []
    contract_violations: dict[str, list[str]] = {}
    for task_id in task_ids:
        current = current_results.get(task_id)
        if current is None:
            missing.append(f"{task_id}:pptx")
            continue
        result_path, result = current
        issues = _rollout_contract_violations(project_root, task_id, result)
        if issues:
            contract_violations[task_id] = issues
            missing.append(f"{task_id}:pptx:contract")
        else:
            complete[task_id] = result_path
    return {
        "schema": "pptbench-rollout-audit",
        "target": "pptx",
        "expected_tasks": expected_tasks,
        "expected_cases": expected_tasks,
        "complete_cases": len(complete),
        "missing": missing,
        "contract_violations": contract_violations,
        "status": "complete" if not missing else "incomplete",
    }


def audit_collected_outputs(project_root: Path, *, expected_tasks: int = 20) -> dict[str, Any]:
    """Validate the review collection containing one final artifact per task."""

    final_root = project_root / "task_outputs" / "final"
    violations: list[str] = []
    validated_tasks = 0
    task_ids = _task_ids(project_root, expected_tasks)
    for task_id in task_ids:
        task_root = final_root / task_id
        required = {
            "result": task_root / "result.json",
            "token_ledger": task_root / "token-ledger.json",
            "usage_summary": task_root / "usage-summary.json",
            "final_pptx": task_root / "final" / "reconstruction.pptx",
            "final_screenshot": task_root / "final" / "screenshot.png",
            "final_components": task_root / "final" / "components.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            violations.append(f"{task_id} final outputs missing: {', '.join(missing)}")
            continue
        result = _json(required["result"])
        metadata_path = _task_root(project_root) / task_id / "metadata.json"
        metadata = _json(metadata_path)
        if result.get("status") != "complete":
            violations.append(f"{task_id} result is not complete")
        if result.get("candidate_id") != metadata.get("candidate_id"):
            violations.append(f"{task_id} result belongs to a stale candidate")
        violations.extend(
            f"{task_id} {issue}"
            for issue in _rollout_contract_violations(project_root, task_id, result)
        )
        violations.extend(
            f"{task_id} final {issue}"
            for issue in check_deck(required["final_pptx"])
        )
        validated_tasks += 1
    return {
        "schema": "pptbench-collected-output-audit",
        "target": "pptx",
        "expected_tasks": expected_tasks,
        "validated_tasks": validated_tasks,
        "violations": violations,
        "status": "complete" if not violations else "invalid",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--task-count", type=int, default=500)
    parser.add_argument("--rollout-count", type=int, default=20)
    parser.add_argument("--agent", choices=("codex", "claude-code", "opencode"), default="codex")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="medium")
    arguments = parser.parse_args(argv)
    project_root = arguments.project_root.resolve()
    report = {
        "tasks": audit_tasks(project_root, expected_count=arguments.task_count),
        "rollouts": audit_rollouts(
            project_root,
            expected_tasks=arguments.rollout_count,
            agent=arguments.agent,
            model=arguments.model,
            reasoning_effort=arguments.reasoning_effort,
        ),
        "collected_outputs": audit_collected_outputs(
            project_root,
            expected_tasks=arguments.rollout_count,
        ),
    }
    report["status"] = (
        "complete"
        if all(section["status"] == "complete" for section in report.values())
        else "incomplete"
    )
    destination = project_root / "task_outputs" / "release-audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
