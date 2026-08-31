"""Build a private candidate task tree from an already-downloaded manifest.

This maintainer-only command consumes a local candidate collection during curation.
The public ``pptbench-materialize`` command is the separate downloader that rebuilds
the frozen release tasks from their recorded arXiv provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import fitz
from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _resolve_candidate_path(
    row: dict[str, Any],
    dataset_root: Path,
    *,
    keys: tuple[str, ...],
) -> Path | None:
    for key in keys:
        value: Any = row
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if not value:
            continue
        path = Path(str(value))
        candidates = [path] if path.is_absolute() else [dataset_root / path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    return None


def _validate_source_pdf(path: Path) -> str | None:
    if path.suffix.casefold() != ".pdf" or path.read_bytes()[:5] != b"%PDF-":
        return "source-is-not-pdf"
    try:
        with fitz.open(path) as document:
            if document.page_count != 1:
                return f"source-has-{document.page_count}-pages"
    except (OSError, RuntimeError, ValueError) as exc:
        return f"source-pdf-invalid:{type(exc).__name__}"
    return None


def _validate_reference(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.load()
            if image.width < 1 or image.height < 1:
                return "reference-has-invalid-size"
    except (OSError, ValueError) as exc:
        return f"reference-invalid:{type(exc).__name__}"
    return None


def _candidate(
    row: dict[str, Any],
    dataset_root: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    source = _resolve_candidate_path(
        row,
        dataset_root,
        keys=("source_image_path", "figure.source_path", "source_path"),
    )
    if source is None:
        return None, "source-missing"
    source_issue = _validate_source_pdf(source)
    if source_issue:
        return None, source_issue
    reference = _resolve_candidate_path(
        row,
        dataset_root,
        keys=("image_path", "render_path"),
    )
    if reference is None:
        return None, "reference-missing"
    reference_issue = _validate_reference(reference)
    if reference_issue:
        return None, reference_issue
    candidate_id = str(row.get("candidate_id") or row.get("structured_id") or "").strip()
    paper = row.get("paper") if isinstance(row.get("paper"), dict) else {}
    arxiv_id = str(paper.get("arxiv_id", "")).strip()
    if not candidate_id:
        return None, "candidate-id-missing"
    if not arxiv_id:
        return None, "arxiv-id-missing"
    return {
        "row": row,
        "source": source,
        "reference": reference,
        "candidate_id": candidate_id,
        "paper": paper,
        "arxiv_id": arxiv_id,
    }, None


def _write_task(root: Path, task_index: int, item: dict[str, Any]) -> dict[str, Any]:
    task_id = f"task_{task_index:04d}"
    task_root = root / task_id
    task_root.mkdir(parents=True)
    source = task_root / "source.pdf"
    reference = task_root / "reference.png"
    shutil.copy2(item["source"], source)
    shutil.copy2(item["reference"], reference)
    row = item["row"]
    caption = str(
        row.get("description")
        or (row.get("figure") or {}).get("caption_text", "")
    )
    metadata = {
        "schema": "pptbench-task-v1",
        "task_id": task_id,
        "task_index": task_index,
        "candidate_id": item["candidate_id"],
        "arxiv_id": item["arxiv_id"],
        "caption": caption,
        "paper": item["paper"],
        "source_kind": "pdf",
        "source_origin": str(item["source"]),
        "source_sha256": _sha256(source),
        "reference_sha256": _sha256(reference),
        "complexity": {
            "subcomponent_count": int(row.get("subcomponent_count", 0) or 0),
            "connector_count": int(row.get("connector_count", 0) or 0),
        },
        "selection": {
            "collector_score": float(row.get("score", 0.0) or 0.0),
            "review_status": str(row.get("review_status", "")),
        },
    }
    (task_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    resources = task_root / "resources"
    resources.mkdir()
    (resources / "index.json").write_text(
        json.dumps(
            {
                "schema": "pptbench-task-resources-v1",
                "task_id": task_id,
                "candidate_id": item["candidate_id"],
                "resource_count": 0,
                "assets": [],
                "manual_qc": {
                    "status": "not-reviewed",
                    "maximum_resources": 12,
                    "removed_count": 0,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "task_id": task_id,
        "candidate_id": item["candidate_id"],
        "arxiv_id": item["arxiv_id"],
        "caption": caption,
        "collector_score": row.get("score", 0.0),
    }


def materialize_tasks(
    manifest: Path,
    output_root: Path,
    *,
    dataset_root: Path | None = None,
    start: int = 0,
    take: int | None = None,
    accepted_only: bool = False,
    replace: bool = False,
) -> dict[str, Any]:
    """Build an atomic harness task tree without mutating the source dataset."""

    manifest = manifest.resolve()
    dataset_root = (
        dataset_root.resolve()
        if dataset_root is not None
        else (manifest.parent.parent if manifest.parent.name == "candidates" else manifest.parent)
    )
    rows = _read_jsonl(manifest)
    rejected: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    seen_papers: set[str] = set()
    for row in rows:
        if accepted_only and str(row.get("review_status", "")).casefold() != "accept":
            rejected["not-accepted"] += 1
            continue
        item, issue = _candidate(row, dataset_root)
        if issue:
            rejected[issue] += 1
            continue
        if item is None:
            raise RuntimeError("candidate validation returned no issue and no candidate")
        if item["arxiv_id"] in seen_papers:
            rejected["duplicate-paper"] += 1
            continue
        seen_papers.add(item["arxiv_id"])
        eligible.append(item)
    selected = eligible[start : start + take if take is not None else None]
    if not selected:
        raise ValueError("no eligible single-page PDF candidates were selected")
    output_root = output_root.resolve()
    if output_root.exists() and not replace:
        raise FileExistsError(f"{output_root} already exists; pass --replace to rebuild it")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    backup = output_root.with_name(f".{output_root.name}.backup")
    try:
        manifest_rows = [
            _write_task(staging, index, item) for index, item in enumerate(selected)
        ]
        (staging / "manifest.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in manifest_rows
            ),
            encoding="utf-8",
        )
        if output_root.exists():
            if backup.exists():
                shutil.rmtree(backup)
            output_root.replace(backup)
        staging.replace(output_root)
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not output_root.exists():
            backup.replace(output_root)
        raise
    return {
        "schema": "pptbench-task-materialization-v1",
        "manifest": str(manifest),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "source_rows": len(rows),
        "eligible_rows": len(eligible),
        "materialized_tasks": len(selected),
        "rejected": dict(sorted(rejected.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pptbench-materialize-candidates",
        description=__doc__,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("tasks"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--take", type=int)
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start < 0:
        raise ValueError("--start cannot be negative")
    if args.take is not None and args.take < 1:
        raise ValueError("--take must be positive")
    summary = materialize_tasks(
        args.manifest,
        args.output_root,
        dataset_root=args.dataset_root,
        start=args.start,
        take=args.take,
        accepted_only=args.accepted_only,
        replace=args.replace,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
