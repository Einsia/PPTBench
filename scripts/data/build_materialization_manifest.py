"""Build the public downloader manifest from a private materialized task tree.

This maintainer-only utility records provenance, deterministic rendering settings,
and the locations of approved native image resources.  It writes metadata only;
PDF and image bytes never enter the resulting manifest.  OpenCV is required for
the exact crop-localization step and is installed with ``pip install -e
.[maintainer]``.

Typical use from the repository root::

    python scripts/data/build_materialization_manifest.py \
        --tasks-root /private/pptbench/tasks \
        --output benchmark/materialization_manifest.jsonl

The input tree must contain the fully materialized ``task_*/source.pdf``,
``reference.png``, and (when applicable) ``resources/resource_*.png`` files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image

from arxiv_flow_dataset.task_materializer import (
    MATERIALIZATION_SCHEMA,
    decode_pdf_image_blocks,
    pixel_sha256,
    render_reference,
)


def _find_origin(blocks: list[Any], target: Image.Image) -> dict[str, Any]:
    target = target.convert("RGBA" if "A" in target.getbands() else "RGB")
    target_array = np.asarray(target)
    for block in blocks:
        if block.mode == target.mode and block.image.size == target.size:
            if pixel_sha256(block.image) == pixel_sha256(target):
                return _origin(block, (0, 0, target.width, target.height), target)

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - maintainer-only dependency
        raise RuntimeError("install opencv-python-headless to build the manifest") from exc
    for block in blocks:
        if (
            block.mode != target.mode
            or block.width < target.width
            or block.height < target.height
        ):
            continue
        result = cv2.matchTemplate(
            np.asarray(block.image), target_array, cv2.TM_SQDIFF_NORMED
        )
        _, _, minimum_location, _ = cv2.minMaxLoc(result)
        left, top = minimum_location
        crop = (left, top, left + target.width, top + target.height)
        candidate = block.image.crop(crop)
        if pixel_sha256(candidate) == pixel_sha256(target):
            return _origin(block, crop, target)
    raise ValueError("resource pixels are not an exact crop of a native PDF image block")


def _origin(block: Any, crop: tuple[int, int, int, int], target: Image.Image) -> dict[str, Any]:
    return {
        "source_block": {
            "mode": block.mode,
            "width": block.width,
            "height": block.height,
            "pixel_sha256": block.pixel_sha256,
        },
        "crop": list(crop),
        "mode": target.mode,
        "width": target.width,
        "height": target.height,
        "pixel_sha256": pixel_sha256(target),
    }


def _difference(left: Image.Image, right: Image.Image) -> dict[str, float | int]:
    left_array = np.asarray(left.convert("RGB"), dtype=np.int16)
    right_array = np.asarray(right.convert("RGB"), dtype=np.int16)
    delta = np.abs(left_array - right_array)
    return {
        "changed_pixels": int(np.any(delta, axis=2).sum()),
        "maximum_channel_delta": int(delta.max(initial=0)),
        "mean_absolute_channel_delta": float(delta.mean()),
    }


def build_row(task_dir: Path) -> dict[str, Any]:
    metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
    source_pdf = task_dir / "source.pdf"
    historic_reference = Image.open(task_dir / "reference.png").convert("RGB")
    rendered_reference = render_reference(source_pdf)
    if rendered_reference.size != historic_reference.size:
        raise ValueError(f"reference dimensions differ for {task_dir.name}")
    blocks = decode_pdf_image_blocks(source_pdf)
    resource_specs: list[dict[str, Any]] = []
    for path in sorted((task_dir / "resources").glob("resource_*.png")):
        target = Image.open(path)
        target.load()
        resource_specs.append({"file": path.name, **_find_origin(blocks, target)})
    paper = metadata["paper"]
    return {
        "schema": MATERIALIZATION_SCHEMA,
        "task_id": task_dir.name,
        "source": {
            "archive_url": paper["source_url"],
            "versioned_id": paper["versioned_id"],
            "filename": metadata["source_filename"],
            "sha256": metadata["source_sha256"],
        },
        "reference": {
            "renderer": "PyMuPDF",
            "pymupdf_version": fitz.VersionBind,
            "maximum_scale": 2.0,
            "maximum_side": 2048,
            "mode": "RGB",
            "width": rendered_reference.width,
            "height": rendered_reference.height,
            "pixel_sha256": pixel_sha256(rendered_reference),
            "historic_pixel_sha256": pixel_sha256(historic_reference),
            "historic_file_sha256": metadata["reference_sha256"],
            "historic_difference": _difference(rendered_reference, historic_reference),
        },
        "resources": resource_specs,
    }


def main(argv: list[str] | None = None) -> int:
    """Create a downloader-only manifest from an explicit private task root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=Path("benchmark/tasks"),
        help="private task tree containing source.pdf and reference.png files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/materialization_manifest.jsonl"),
        help="metadata-only JSONL manifest to write",
    )
    args = parser.parse_args(argv)
    rows = [build_row(task_dir) for task_dir in sorted(args.tasks_root.glob("task_*"))]
    if len(rows) != 500:
        raise ValueError(f"expected 500 tasks, found {len(rows)}")
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
