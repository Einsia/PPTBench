"""Materialize PPTBench task pixels locally from exact arXiv source versions.

The public repository contains provenance and deterministic rendering specifications,
not third-party paper figures.  This module downloads the source archive selected by
the user, extracts the PDF figure whose byte hash is frozen in task metadata, renders
the reference, and recreates approved native raster resources.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import fitz
from PIL import Image

from .http import CachedHttpClient
from .source import extract_source_archive


ARXIV_API_TERMS = "https://info.arxiv.org/help/api/tou.html"
ARXIV_REUSE_HELP = "https://info.arxiv.org/help/license/reuse.html"
DEFAULT_USER_AGENT = "PPTBench-materializer/1.0 (https://github.com/Einsia/PPTBench)"
MATERIALIZATION_SCHEMA = "pptbench-task-materialization-v1"
RESOURCE_INDEX_SCHEMA = "pptbench-task-resources-v2"


@dataclass(frozen=True)
class PdfImageBlock:
    mode: str
    width: int
    height: int
    pixel_sha256: str
    image: Image.Image


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(image: Image.Image) -> str:
    mode = "RGBA" if "A" in image.getbands() else "RGB"
    normalized = image.convert(mode)
    digest = hashlib.sha256()
    digest.update(mode.encode("ascii"))
    digest.update(b"\0")
    digest.update(normalized.width.to_bytes(8, "big"))
    digest.update(normalized.height.to_bytes(8, "big"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def reference_pixel_hashes(spec: dict[str, Any]) -> set[str]:
    """Accept only frozen renderer output or the frozen original reference."""
    return {str(spec[key]) for key in ("pixel_sha256", "historic_pixel_sha256") if spec.get(key)}


def render_reference(
    source_pdf: Path,
    *,
    maximum_scale: float = 2.0,
    maximum_side: int = 2048,
) -> Image.Image:
    with fitz.open(source_pdf) as document:
        if document.page_count != 1:
            raise ValueError(f"reference source must have one PDF page: {source_pdf}")
        page = document[0]
        page_side = max(page.rect.width, page.rect.height)
        if page_side <= 0:
            raise ValueError(f"reference source has invalid dimensions: {source_pdf}")
        scale = min(maximum_scale, maximum_side / page_side)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=True)
    rgba = Image.frombytes("RGBA", (pixmap.width, pixmap.height), pixmap.samples)
    return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")


def decode_pdf_image_blocks(source_pdf: Path) -> list[PdfImageBlock]:
    blocks: list[PdfImageBlock] = []
    with fitz.open(source_pdf) as document:
        if document.page_count != 1:
            raise ValueError(f"resource source must have one PDF page: {source_pdf}")
        raw_blocks = document[0].get_text("dict").get("blocks", [])
    for block in raw_blocks:
        if not isinstance(block, dict) or block.get("type") != 1 or not block.get("image"):
            continue
        try:
            base = Image.open(BytesIO(block["image"]))
            base.load()
        except (OSError, ValueError):
            continue
        image = base.convert("RGB")
        mask_payload = block.get("mask")
        if mask_payload:
            mask = Image.open(BytesIO(mask_payload)).convert("L")
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.Resampling.LANCZOS)
            image.putalpha(mask)
        blocks.append(
            PdfImageBlock(
                mode=image.mode,
                width=image.width,
                height=image.height,
                pixel_sha256=pixel_sha256(image),
                image=image,
            )
        )
    return blocks


def _find_source_pdf(
    archive_payload: bytes,
    *,
    source_filename: str,
    expected_sha256: str,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="pptbench-arxiv-source-") as directory:
        extracted = extract_source_archive(archive_payload, Path(directory))
        name_matches = [path for path in extracted if path.name == source_filename]
        candidates = name_matches or [path for path in extracted if path.suffix.lower() == ".pdf"]
        hash_matches = [path for path in candidates if file_sha256(path) == expected_sha256]
        if not hash_matches:
            raise ValueError(
                f"expected an archive member named {source_filename!r} with SHA-256 "
                f"{expected_sha256}, found {len(hash_matches)}"
            )
        # arXiv sources can carry identical figures in several directories.
        # The frozen byte hash identifies the input, not the number of copies.
        return hash_matches[0].read_bytes()


def _resource_from_spec(blocks: list[PdfImageBlock], spec: dict[str, Any]) -> Image.Image:
    source = spec["source_block"]
    matches = [
        block
        for block in blocks
        if block.pixel_sha256 == source["pixel_sha256"]
        and block.mode == source["mode"]
        and block.width == source["width"]
        and block.height == source["height"]
    ]
    if not matches:
        raise ValueError(f"native PDF image block is missing for {spec['file']}")
    left, top, right, bottom = (int(value) for value in spec["crop"])
    image = matches[0].image.crop((left, top, right, bottom))
    if image.mode != spec["mode"]:
        image = image.convert(spec["mode"])
    if image.size != (int(spec["width"]), int(spec["height"])):
        raise ValueError(f"materialized dimensions differ for {spec['file']}")
    if pixel_sha256(image) != spec["pixel_sha256"]:
        raise ValueError(f"materialized pixels differ for {spec['file']}")
    return image


def materialize_task_from_payload(
    *,
    task_spec_root: Path,
    task_spec: dict[str, Any],
    archive_payload: bytes,
    output_root: Path,
    license_record: dict[str, Any] | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    task_id = str(task_spec["task_id"])
    source_task = task_spec_root / task_id
    metadata_path = source_task / "metadata.json"
    resource_index_path = source_task / "resources" / "index.json"
    if not metadata_path.is_file() or not resource_index_path.is_file():
        raise FileNotFoundError(f"task specification is incomplete: {source_task}")

    task_output = output_root / task_id
    if task_output.exists() and not replace:
        raise FileExistsError(f"{task_output} exists; pass --replace to rebuild it")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{task_id}.staging-", dir=output_root))
    try:
        source_spec = task_spec["source"]
        source_bytes = _find_source_pdf(
            archive_payload,
            source_filename=str(source_spec["filename"]),
            expected_sha256=str(source_spec["sha256"]),
        )
        source_pdf = staging / "source.pdf"
        source_pdf.write_bytes(source_bytes)

        reference_spec = task_spec["reference"]
        reference = render_reference(
            source_pdf,
            maximum_scale=float(reference_spec["maximum_scale"]),
            maximum_side=int(reference_spec["maximum_side"]),
        )
        if reference.size != (
            int(reference_spec["width"]),
            int(reference_spec["height"]),
        ):
            raise ValueError(f"materialized reference dimensions differ for {task_id}")
        if pixel_sha256(reference) not in reference_pixel_hashes(reference_spec):
            raise ValueError(f"materialized reference pixels differ for {task_id}")
        reference_path = staging / "reference.png"
        reference.save(reference_path, format="PNG", optimize=True)

        source_index = json.loads(resource_index_path.read_text(encoding="utf-8"))
        resources = staging / "resources"
        resources.mkdir()
        blocks = decode_pdf_image_blocks(source_pdf) if task_spec["resources"] else []
        generated_assets: list[dict[str, Any]] = []
        original_assets = {
            str(asset.get("file")): asset
            for asset in source_index.get("assets", [])
            if isinstance(asset, dict)
        }
        for resource_spec in task_spec["resources"]:
            image = _resource_from_spec(blocks, resource_spec)
            destination = resources / str(resource_spec["file"])
            image.save(destination, format="PNG", optimize=True)
            original = original_assets.get(destination.name, {})
            generated_assets.append(
                {
                    "file": destination.name,
                    "width": image.width,
                    "height": image.height,
                    "has_alpha": "A" in image.getbands(),
                    "sha256": file_sha256(destination),
                    "pixel_sha256": pixel_sha256(image),
                    "historic_sha256": original.get("sha256", ""),
                }
            )
        generated_index = {
            **source_index,
            "schema": RESOURCE_INDEX_SCHEMA,
            "resource_count": len(generated_assets),
            "assets": generated_assets,
            "materialized_from": "benchmark/materialization_manifest.jsonl",
        }
        (resources / "index.json").write_text(
            json.dumps(generated_index, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(metadata_path, staging / "metadata.json")
        report = {
            "schema": "pptbench-materialized-task-report-v1",
            "task_id": task_id,
            "source_sha256": file_sha256(source_pdf),
            "reference_sha256": file_sha256(reference_path),
            "reference_pixel_sha256": pixel_sha256(reference),
            "resource_count": len(generated_assets),
            "license": license_record or {},
            "arxiv_terms": ARXIV_API_TERMS,
            "reuse_help": ARXIV_REUSE_HELP,
        }
        (staging / "materialization_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if task_output.exists():
            shutil.rmtree(task_output)
        staging.replace(task_output)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def verify_materialized_task(task_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Verify persisted bytes and pixels before resuming a materialization."""
    if file_sha256(task_root / "source.pdf") != spec["source"]["sha256"]:
        raise ValueError(f"source hash mismatch: {task_root.name}")
    with Image.open(task_root / "reference.png") as image:
        if pixel_sha256(image) not in reference_pixel_hashes(spec["reference"]):
            raise ValueError(f"reference hash mismatch: {task_root.name}")
    for resource in spec["resources"]:
        with Image.open(task_root / "resources" / resource["file"]) as image:
            if pixel_sha256(image) != resource["pixel_sha256"]:
                raise ValueError(f"resource hash mismatch: {task_root.name}/{resource['file']}")
    index = json.loads((task_root / "resources/index.json").read_text(encoding="utf-8"))
    expected = {resource["file"] for resource in spec["resources"]}
    if {asset["file"] for asset in index["assets"]} != expected:
        raise ValueError(f"resource index mismatch: {task_root.name}")
    metadata = json.loads((task_root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("task_id") != spec["task_id"]:
        raise ValueError(f"metadata task mismatch: {task_root.name}")
    report = json.loads((task_root / "materialization_report.json").read_text(encoding="utf-8"))
    if report.get("task_id") != spec["task_id"]:
        raise ValueError(f"report task mismatch: {task_root.name}")
    return report


def _selected_task_ids(args: argparse.Namespace, available: Iterable[str]) -> list[str]:
    selected = list(args.task_id or [])
    if args.task_file:
        selected.extend(
            line.strip()
            for line in args.task_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not selected:
        return sorted(available)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown task IDs: {', '.join(unknown)}")
    return list(dict.fromkeys(selected))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pptbench-materialize",
        description=__doc__,
    )
    parser.add_argument("--tasks-root", type=Path, default=Path("benchmark/tasks"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmark/materialization_manifest.jsonl")
    )
    parser.add_argument(
        "--license-audit", type=Path, default=Path("benchmark/source_license_audit.jsonl")
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/pptbench-tasks"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/arxiv-sources"))
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--task-file", type=Path)
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--replace", action="store_true")
    existing.add_argument("--resume", action="store_true", help="verify and reuse completed tasks")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--accept-arxiv-terms", action="store_true")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--min-interval", type=float, default=3.1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.offline and not args.accept_arxiv_terms:
        raise SystemExit(
            "Online materialization requires --accept-arxiv-terms after reviewing "
            f"{ARXIV_API_TERMS} and {ARXIV_REUSE_HELP}"
        )
    if args.min_interval < 3.0:
        raise SystemExit("--min-interval must be at least 3 seconds under the arXiv API terms")

    specs = {str(row["task_id"]): row for row in _read_jsonl(args.manifest)}
    licenses = {str(row["task_id"]): row for row in _read_jsonl(args.license_audit)}
    selected = _selected_task_ids(args, specs)
    client = CachedHttpClient(
        args.cache_dir,
        args.user_agent,
        timeout=120,
        min_interval=args.min_interval,
        max_attempts=4,
        max_response_bytes=500_000_000,
    )
    reports: list[dict[str, Any]] = []
    for index, task_id in enumerate(selected, 1):
        spec = specs[task_id]
        task_output = args.output_root / task_id
        if args.resume and task_output.exists():
            reports.append(verify_materialized_task(task_output, spec))
            print(f"verified {index}/{len(selected)} {task_id}", flush=True)
            continue
        source = spec["source"]
        cache_key = f"source-{str(source['versioned_id']).replace('/', '_')}"
        cache_path = args.cache_dir / f"{cache_key}.archive"
        if args.offline:
            if not cache_path.is_file():
                raise FileNotFoundError(f"offline source cache is missing: {cache_path}")
            payload = cache_path.read_bytes()
        else:
            payload = client.get(str(source["archive_url"]), cache_key, ".archive")
        reports.append(
            materialize_task_from_payload(
                task_spec_root=args.tasks_root,
                task_spec=spec,
                archive_payload=payload,
                output_root=args.output_root,
                license_record=licenses.get(task_id),
                replace=args.replace,
            )
        )
        print(f"materialized {index}/{len(selected)} {task_id}", flush=True)
    summary = {
        "schema": "pptbench-materialization-run-v1",
        "task_count": len(reports),
        "task_ids": [report["task_id"] for report in reports],
        "output_root": str(args.output_root.resolve()),
        "arxiv_terms_accepted": bool(args.accept_arxiv_terms or args.offline),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "materialization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
