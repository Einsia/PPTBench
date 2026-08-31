"""Materialize portable evaluation inputs from finalized PPTBench tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

from PIL import Image

from .config import BenchmarkConfig


SAMPLE_SCHEMA = "pptbench-runtime-sample-v1"
_TASK_ID_RE = re.compile(r"task_\d{4}")
_RESOURCE_NAME_RE = re.compile(r"resource_\d{4}\.png")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def normalize_reference(source: Path, destination: Path, width: int, height: int) -> None:
    """Letterbox one task reference onto the benchmark canvas."""

    with Image.open(source) as loaded:
        image = loaded.convert("RGB")
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(
            image,
            ((width - image.width) // 2, (height - image.height) // 2),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, optimize=True)


def _update_file_identity(digest: Any, path: Path) -> None:
    """Add a cheap local identity for an immutable task input to a digest."""

    digest.update(path.name.encode("utf-8"))
    digest.update(b"\0")
    if not path.is_file():
        digest.update(b"missing\0")
        return
    stat = path.stat()
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    digest.update(b"\0")


def _resource_paths(task_root: Path) -> tuple[Path | None, list[Path]]:
    resource_root = task_root / "resources"
    index_path = resource_root / "index.json"
    if not index_path.is_file():
        unindexed = sorted(resource_root.glob("resource_*.png"))
        if unindexed:
            raise ValueError(f"{task_root.name} has resources without resources/index.json")
        return None, []
    index = _load_json(index_path)
    assets = index.get("assets") or []
    if not isinstance(assets, list):
        raise ValueError(f"{index_path} assets must be a list")
    paths: list[Path] = []
    seen: set[str] = set()
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError(f"{index_path} contains a non-object asset")
        name = str(asset.get("file", ""))
        if not _RESOURCE_NAME_RE.fullmatch(name) or name in seen:
            raise ValueError(f"{index_path} contains an invalid resource filename: {name!r}")
        path = resource_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(name)
        paths.append(path)
    indexed = {path.name for path in paths}
    unindexed = {
        path.name for path in resource_root.glob("resource_*.png") if path.name not in indexed
    }
    if unindexed:
        raise ValueError(f"{task_root.name} has unindexed resources: {sorted(unindexed)}")
    return index_path, paths


def _task_specs(
    config: BenchmarkConfig,
    *,
    minimum_count: int | None = None,
    selectors: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows = _load_jsonl(config.tasks_manifest)
    requested = max(config.sample_count, minimum_count or 0)
    if requested < 1:
        raise ValueError("sample_count must be positive")
    if requested > len(rows):
        raise ValueError(
            f"requested {requested} samples, but {config.tasks_manifest} contains {len(rows)}"
        )
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest_index, manifest_row in enumerate(rows[:requested]):
        task_id = str(manifest_row.get("task_id", ""))
        if not _TASK_ID_RE.fullmatch(task_id) or task_id in seen:
            raise ValueError(f"invalid or duplicate task_id in manifest: {task_id!r}")
        seen.add(task_id)
        if selectors and not any(selector in task_id for selector in selectors):
            continue
        task_root = config.tasks_root / task_id
        metadata_path = task_root / "metadata.json"
        reference_path = task_root / "reference.png"
        source_path = task_root / "source.pdf"
        for required in (metadata_path, reference_path, source_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        metadata = _load_json(metadata_path)
        if metadata.get("task_id") not in {None, task_id}:
            raise ValueError(f"{metadata_path} contains the wrong task_id")
        candidate_id = str(metadata.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError(f"{metadata_path} contains no candidate_id")
        manifest_candidate = str(manifest_row.get("candidate_id", ""))
        if manifest_candidate and manifest_candidate != candidate_id:
            raise ValueError(f"{task_id} manifest and metadata candidate IDs differ")
        resource_index, resources = _resource_paths(task_root)
        component_manifest = task_root / "components.json"
        if not component_manifest.is_file():
            component_manifest = None

        digest = hashlib.sha256()
        digest.update(
            json.dumps(manifest_row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(f"\0{config.canvas_width}x{config.canvas_height}\0".encode("ascii"))
        for path in (
            metadata_path,
            reference_path,
            source_path,
            resource_index,
            *resources,
            component_manifest,
        ):
            if path is not None:
                _update_file_identity(digest, path)
        specs.append(
            {
                "manifest_index": manifest_index,
                "manifest": manifest_row,
                "task_id": task_id,
                "task_root": task_root,
                "metadata": metadata,
                "reference": reference_path,
                "source": source_path,
                "resource_index": resource_index,
                "resources": resources,
                "component_manifest": component_manifest,
                "fingerprint": digest.hexdigest(),
            }
        )
    return specs


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def _materialize_spec(config: BenchmarkConfig, spec: dict[str, Any]) -> dict[str, Any]:
    task_id = str(spec["task_id"])
    task_root = Path(spec["task_root"])
    sample_root = config.output_dir / "samples" / task_id
    sample_root.mkdir(parents=True, exist_ok=True)
    reference = sample_root / "reference.png"
    source_pdf = sample_root / "source.pdf"
    normalize_reference(
        Path(spec["reference"]),
        reference,
        config.canvas_width,
        config.canvas_height,
    )
    _copy_file(Path(spec["source"]), source_pdf)

    sample_resources = sample_root / "resources"
    if sample_resources.exists():
        shutil.rmtree(sample_resources)
    resource_paths: list[str] = []
    resources = [Path(path) for path in spec["resources"]]
    if resources:
        sample_resources.mkdir(parents=True)
        resource_index = spec["resource_index"]
        if resource_index is not None:
            _copy_file(Path(resource_index), sample_resources / "index.json")
        for resource in resources:
            destination = sample_resources / resource.name
            _copy_file(resource, destination)
            resource_paths.append(str(destination.resolve()))

    metadata = spec["metadata"]
    manifest_row = spec["manifest"]
    score = manifest_row.get("final_quality_score")
    if score is None:
        score = (metadata.get("selection") or {}).get("final_quality_score", 0.0)
    result: dict[str, Any] = {
        "schema": SAMPLE_SCHEMA,
        "sample_id": task_id,
        "candidate_id": metadata["candidate_id"],
        "task_manifest_index": spec["manifest_index"],
        "reference_path": str(reference.resolve()),
        "original_path": str((task_root / "reference.png").resolve()),
        "description": metadata.get("caption", manifest_row.get("caption", "")),
        "score": score,
        "paper": metadata.get("paper", {}),
        "source_kind": "pdf",
        "source_pdf_path": str(source_pdf.resolve()),
        "source_sha256": metadata.get("source_sha256", ""),
        "subcomponent_count": (metadata.get("complexity") or {}).get("subcomponent_count", 0),
        "resource_paths": resource_paths,
        "input_fingerprint": spec["fingerprint"],
    }
    component_manifest = spec["component_manifest"]
    if component_manifest is not None:
        result["component_manifest_path"] = str(Path(component_manifest).resolve())
    return result


def materialize_samples(
    config: BenchmarkConfig,
    *,
    minimum_count: int | None = None,
) -> list[dict[str, Any]]:
    """Build runtime sample assets from the leading finalized manifest tasks."""

    specs = _task_specs(config, minimum_count=minimum_count)
    rows = [_materialize_spec(config, spec) for spec in specs]
    destination = config.output_dir / "samples.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return rows


def _cached_samples(
    config: BenchmarkConfig,
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    manifest = config.output_dir / "samples.json"
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, list) or len(value) < len(specs):
        return None
    rows = value[: len(specs)]
    for row, spec in zip(rows, specs, strict=True):
        if not isinstance(row, dict):
            return None
        task_id = str(spec["task_id"])
        sample_root = config.output_dir / "samples" / task_id
        expected_reference = (sample_root / "reference.png").resolve()
        expected_source = (sample_root / "source.pdf").resolve()
        expected_resources = [
            str((sample_root / "resources" / Path(path).name).resolve())
            for path in spec["resources"]
        ]
        if (
            row.get("schema") != SAMPLE_SCHEMA
            or row.get("sample_id") != task_id
            or row.get("candidate_id") != spec["metadata"]["candidate_id"]
            or row.get("input_fingerprint") != spec["fingerprint"]
            or Path(str(row.get("reference_path", ""))).resolve() != expected_reference
            or Path(str(row.get("source_pdf_path", ""))).resolve() != expected_source
            or row.get("resource_paths") != expected_resources
            or not expected_reference.is_file()
            or not expected_source.is_file()
            or expected_source.stat().st_size != Path(spec["source"]).stat().st_size
        ):
            return None
        with Image.open(expected_reference) as image:
            if image.size != (config.canvas_width, config.canvas_height):
                return None
        if any(not Path(path).is_file() for path in expected_resources):
            return None
        component_manifest = spec["component_manifest"]
        expected_component = (
            str(Path(component_manifest).resolve()) if component_manifest is not None else None
        )
        if row.get("component_manifest_path") != expected_component:
            return None
    return rows


def load_samples(
    config: BenchmarkConfig,
    *,
    minimum_count: int | None = None,
    selectors: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load current runtime samples, rebuilding them when task inputs changed."""

    specs = _task_specs(config, minimum_count=minimum_count, selectors=selectors)
    current = _cached_samples(config, specs)
    if current is not None:
        return current
    if selectors:
        return [_materialize_spec(config, spec) for spec in specs]
    return materialize_samples(config, minimum_count=minimum_count)
