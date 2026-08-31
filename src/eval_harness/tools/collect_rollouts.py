"""Collect completed single-run reconstruction artifacts for review."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
import shutil
from typing import Any

from eval_harness.paths import default_project_root
from eval_harness.usage import aggregate_usage_ledger
from .run_rollouts import RolloutJob, _existing_complete_result, _task_roots


ROLLOUT_PROTOCOL = "image-only-single-run"


def _load_complete_results(
    project_root: Path,
    *,
    task_ids: Iterable[str] | None = None,
    agent: str = "codex",
    model: str = "gpt-5.6-sol",
    reasoning_effort: str = "medium",
) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Load complete results for the requested current task inputs."""

    if task_ids is None:
        taskset = project_root / "configs" / "taskset.txt"
        if taskset.is_file():
            task_ids = [
                line.strip()
                for line in taskset.read_text(encoding="utf-8-sig").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        else:
            for root in _task_roots(project_root):
                discovered = sorted(path.name for path in root.glob("task_*") if path.is_dir())
                if discovered:
                    task_ids = discovered
                    break
            else:
                task_ids = []
    results: dict[str, tuple[Path, dict[str, Any]]] = {}
    for task_id in task_ids:
        path = _existing_complete_result(
            project_root,
            RolloutJob(task_id=task_id),
            agent=agent,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        if path is None:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("protocol") == ROLLOUT_PROTOCOL:
            results[task_id] = (path, value)
    return results


def _artifact_paths(result: dict[str, Any], prefix: str = "final") -> dict[str, Path]:
    """Return the artifact paths for one completed run."""

    return {
        "pptx": Path(str(result.get(f"{prefix}_pptx_path", ""))),
        "screenshot": Path(str(result.get(f"{prefix}_screenshot_path", ""))),
        "components": Path(str(result.get(f"{prefix}_components_path", ""))),
    }


def collect_rollouts(
    project_root: Path,
    *,
    task_count: int = 20,
    agent: str = "codex",
    model: str = "gpt-5.6-sol",
    reasoning_effort: str = "medium",
) -> dict[str, Any]:
    """Collect one completed PPTX and its diagnostics per task."""

    if task_count < 1:
        raise ValueError("task_count must be positive")
    project_root = project_root.resolve()
    taskset = project_root / "configs" / "taskset.txt"
    if taskset.is_file():
        task_ids = [
            line.strip()
            for line in taskset.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        task_ids = []
        for root in _task_roots(project_root):
            discovered = sorted(path.name for path in root.glob("task_*") if path.is_dir())
            if discovered:
                task_ids = discovered
                break
    task_ids = task_ids[:task_count]
    if len(task_ids) < task_count:
        raise ValueError(
            f"requested {task_count} tasks, but only {len(task_ids)} task IDs are available"
        )
    results = _load_complete_results(
        project_root,
        task_ids=task_ids,
        agent=agent,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    destination_root = project_root / "task_outputs" / "final"
    destination_root.mkdir(parents=True, exist_ok=True)
    collected: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid_artifacts: dict[str, list[str]] = {}
    all_ledger_rows: list[dict[str, Any]] = []

    for task_id in task_ids:
        current = results.get(task_id)
        if current is None:
            missing.append(task_id)
            continue
        result_path, result = current
        paths = _artifact_paths(result)
        absent = [name for name, path in paths.items() if not path.is_file()]
        ledger_path = Path(str(result.get("token_ledger_path", "")))
        usage_summary_path = Path(str(result.get("usage_summary_path", "")))
        if not ledger_path.is_file():
            absent.append("token-ledger")
        if not usage_summary_path.is_file():
            absent.append("usage-summary")
        if absent:
            missing.append(task_id)
            invalid_artifacts[task_id] = absent
            continue
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            usage_summary = json.loads(usage_summary_path.read_text(encoding="utf-8"))
            recomputed_usage = aggregate_usage_ledger(ledger)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            missing.append(task_id)
            invalid_artifacts[task_id] = ["invalid-token-accounting"]
            continue
        if not isinstance(ledger, list) or not isinstance(usage_summary, dict):
            missing.append(task_id)
            invalid_artifacts[task_id] = ["invalid-token-accounting"]
            continue
        if usage_summary != recomputed_usage or usage_summary.get("usage_complete") is not True:
            missing.append(task_id)
            invalid_artifacts[task_id] = ["incomplete-token-accounting"]
            continue

        task_destination = destination_root / task_id
        if task_destination.exists():
            shutil.rmtree(task_destination)
        task_destination.mkdir(parents=True)
        final_destination = task_destination / "final"
        final_destination.mkdir()
        shutil.copy2(paths["pptx"], final_destination / "reconstruction.pptx")
        shutil.copy2(paths["screenshot"], final_destination / "screenshot.png")
        shutil.copy2(paths["components"], final_destination / "components.json")
        shutil.copy2(result_path, task_destination / "result.json")
        shutil.copy2(ledger_path, task_destination / "token-ledger.json")
        shutil.copy2(usage_summary_path, task_destination / "usage-summary.json")
        all_ledger_rows.extend(ledger)
        collected.append(
            {
                "task_id": task_id,
                "target": "pptx",
                "result": str(result_path),
                "final_composite": result.get("final_metrics", {}).get("composite"),
                "usage": usage_summary,
            }
        )

    summary = {
        "schema": "pptbench-collected-rollouts",
        "target": "pptx",
        "expected_tasks": task_count,
        "complete_tasks": len(collected),
        "missing_tasks": missing,
        "invalid_artifacts": invalid_artifacts,
        "usage": aggregate_usage_ledger(all_ledger_rows),
        "mean_final_composite": (
            sum(float(row["final_composite"]) for row in collected) / len(collected)
            if collected and all(row["final_composite"] is not None for row in collected)
            else None
        ),
        "status": "complete" if not missing else "incomplete",
        "tasks": collected,
    }
    (destination_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pptbench-collect-rollouts",
        description=__doc__,
    )
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--task-count", type=int, default=20)
    parser.add_argument("--agent", choices=("codex", "claude-code", "opencode"), default="codex")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="medium")
    arguments = parser.parse_args(argv)
    summary = collect_rollouts(
        arguments.project_root,
        task_count=arguments.task_count,
        agent=arguments.agent,
        model=arguments.model,
        reasoning_effort=arguments.reasoning_effort,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
