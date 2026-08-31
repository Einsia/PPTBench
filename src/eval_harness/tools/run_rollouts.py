"""Run bounded batches of single-task PowerPoint reconstructions.

The batch driver is intentionally a thin scheduler around the public
single-run harness. Each task is launched with its own opaque run id, so a
failed case can be retried without changing the task contract or mixing
artifacts from another case.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

from eval_harness.config import load_config
from eval_harness.paths import default_project_root


HARNESS = "single-run"
TARGET = "pptx"
DEFAULT_AGENT = "codex"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"


@dataclass(frozen=True, slots=True)
class RolloutJob:
    """Identify one task to run."""

    task_id: str


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object for missing/corrupt data."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _task_roots(project_root: Path) -> tuple[Path, ...]:
    """Return the configured task root followed by compatible fallbacks."""

    candidates: list[Path] = []
    try:
        configured = load_config(project_root / "project.toml").benchmark.tasks_root
    except (OSError, KeyError, TypeError, ValueError):
        configured = None
    if configured is not None:
        candidates.append(Path(configured).resolve())
    candidates.extend(
        (
            (project_root / "data" / "pptbench-tasks").resolve(),
            (project_root / "tasks").resolve(),
            (project_root / "benchmark" / "tasks").resolve(),
        )
    )
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _metadata(project_root: Path, task_id: str) -> dict[str, Any]:
    """Load task metadata from the configured public or legacy local root."""

    for root in _task_roots(project_root):
        metadata_path = root / task_id / "metadata.json"
        if metadata_path.is_file():
            return _read_json(metadata_path)
    return {}


def _run_id(
    project_root: Path,
    job: RolloutJob,
    *,
    agent: str = DEFAULT_AGENT,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> str:
    """Return an opaque id bound to the task input and generation settings."""

    candidate = str(_metadata(project_root, job.task_id).get("candidate_id", ""))
    if not candidate:
        raise ValueError(f"task {job.task_id} has no current candidate id")
    identity = "\0".join((HARNESS, agent, model, reasoning_effort, job.task_id, candidate))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return f"rollout-pptx-{digest}"


def _run_root(
    project_root: Path,
    job: RolloutJob,
    *,
    agent: str = DEFAULT_AGENT,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> Path:
    """Return the canonical output directory for one single-run case."""

    config = load_config(project_root / "project.toml")
    return (
        config.benchmark.output_dir
        / "harnesses"
        / HARNESS
        / agent
        / _run_id(
            project_root,
            job,
            agent=agent,
            model=model,
            reasoning_effort=reasoning_effort,
        )
    )


def _result_path(
    project_root: Path,
    job: RolloutJob,
    *,
    agent: str = DEFAULT_AGENT,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> Path:
    return _run_root(
        project_root,
        job,
        agent=agent,
        model=model,
        reasoning_effort=reasoning_effort,
    ) / job.task_id / "agent_run" / "run.json"


def _existing_complete_result(
    project_root: Path,
    job: RolloutJob,
    *,
    agent: str = DEFAULT_AGENT,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> Path | None:
    """Return a complete result for the current task input, if one exists."""

    try:
        result_path = _result_path(
            project_root,
            job,
            agent=agent,
            model=model,
            reasoning_effort=reasoning_effort,
        )
    except (OSError, ValueError):
        return None
    value = _read_json(result_path)
    if (
        value.get("status") != "complete"
        or value.get("harness") != HARNESS
        or value.get("target") != TARGET
        or value.get("candidate_id") != _metadata(project_root, job.task_id).get("candidate_id")
    ):
        return None
    artifact = Path(str(value.get("final_pptx_path", "")))
    return result_path if artifact.is_file() else None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_job(
    project_root: Path,
    job: RolloutJob,
    *,
    agent: str,
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
    soffice_command: str | Path | None,
) -> dict[str, Any]:
    existing = _existing_complete_result(
        project_root,
        job,
        agent=agent,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    if existing is not None:
        return {
            "task_id": job.task_id,
            "target": TARGET,
            "status": "complete",
            "skipped": True,
            "result_path": str(existing),
        }

    config = load_config(project_root / "project.toml")
    run_id = _run_id(
        project_root,
        job,
        agent=agent,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    command = [
        sys.executable,
        "-m",
        "eval_harness.cli",
        "--config",
        str(project_root / "project.toml"),
        "harness-run",
        "--harness",
        HARNESS,
        "--agent",
        agent,
        "--model",
        model,
        "--reasoning-effort",
        reasoning_effort,
        "--sample",
        job.task_id,
        "--run-id",
        run_id,
        "--timeout",
        str(timeout_seconds),
    ]
    if soffice_command:
        command.extend(("--soffice-command", str(soffice_command)))
    log_dir = config.benchmark.output_dir / "rollout-driver"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.task_id}.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_seconds + 60,
            )
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        return {
            "task_id": job.task_id,
            "target": TARGET,
            "status": "failed",
            "skipped": False,
            "returncode": None,
            "log_path": str(log_path),
            "error": "harness process timed out",
        }
    except Exception as exc:  # keep the batch moving when one task fails
        return {
            "task_id": job.task_id,
            "target": TARGET,
            "status": "failed",
            "skipped": False,
            "error": repr(exc),
        }
    completed_result = _existing_complete_result(
        project_root,
        job,
        agent=agent,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    return {
        "task_id": job.task_id,
        "target": TARGET,
        "status": "complete" if completed_result is not None else "failed",
        "skipped": False,
        "returncode": returncode,
        "log_path": str(log_path),
        "result_path": str(completed_result) if completed_result is not None else "",
        "error": "" if completed_result is not None else f"harness exited with {returncode}",
    }


def run_all(
    *,
    project_root: Path | None = None,
    task_count: int = 500,
    concurrency: int = 12,
    agent: str = DEFAULT_AGENT,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    soffice_command: str | Path | None = None,
) -> dict[str, Any]:
    """Run one independent single-run reconstruction for each selected task."""

    if not 1 <= task_count <= 500:
        raise ValueError("task_count must be between one and 500")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if agent not in {"codex", "claude-code", "opencode"}:
        raise ValueError(f"unsupported agent: {agent}")
    if not model.strip() or not reasoning_effort.strip():
        raise ValueError("model and reasoning_effort cannot be empty")
    project_root = (project_root or default_project_root()).resolve()
    config = load_config(project_root / "project.toml")
    # Loading once validates the task manifest before any model call starts.
    from eval_harness.samples import load_samples

    samples = load_samples(config.benchmark, minimum_count=task_count)
    # The frozen task set is intentionally not numerically contiguous.  Derive
    # jobs from the materialized manifest rather than synthesizing task_0000,
    # task_0001, ...; doing the latter silently selects the wrong inputs after
    # curation removes a candidate.
    jobs = [RolloutJob(task_id=str(sample["sample_id"])) for sample in samples[:task_count]]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _run_job,
                project_root,
                job,
                agent=agent,
                model=model,
                reasoning_effort=reasoning_effort,
                timeout_seconds=1_200,
                soffice_command=soffice_command,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"[{record['status']}] {record['task_id']}", flush=True)

    records.sort(key=lambda row: str(row["task_id"]))
    driver_root = config.benchmark.output_dir / "rollout-driver"
    summary = {
        "schema": "pptbench-rollout-driver",
        "harness": HARNESS,
        "target": TARGET,
        "task_count": task_count,
        "requested_cases": len(records),
        "complete_cases": sum(row["status"] == "complete" for row in records),
        "failed_cases": sum(row["status"] != "complete" for row in records),
        "concurrency": concurrency,
        "agent": {"kind": agent, "model": model, "reasoning_effort": reasoning_effort},
        "cases": records,
    }
    _write_json(driver_root / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pptbench-rollout-batch",
        description=__doc__,
    )
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--task-count", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument(
        "--agent",
        choices=("codex", "claude-code", "opencode"),
        default=DEFAULT_AGENT,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--soffice-command")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = run_all(
        project_root=arguments.project_root,
        task_count=arguments.task_count,
        concurrency=arguments.concurrency,
        agent=arguments.agent,
        model=arguments.model,
        reasoning_effort=arguments.reasoning_effort,
        soffice_command=arguments.soffice_command,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
