"""Calibrate the PPTX image-similarity artifact gate on a small smoke matrix.

The smoke matrix is intentionally deterministic and does not alter benchmark
results.  It provides two empirical bounds: the lowest similarity observed for
reference-like raster copies and the highest similarity observed for permitted
local resources.  The separation midpoint is reported as a calibration
diagnostic when the bounds are separated; runtime uses the evaluator's
configured conservative threshold.  A real task reference and approved
resources can be supplied to replace the synthetic resource fixtures.

Example::

    python scripts/release/calibrate_image_similarity_gate.py \
      --reference data/pptbench-tasks/task_0000/reference.png \
      --resource data/pptbench-tasks/task_0012/resources/resource_0001.png \
      --output .tmp/image-gate-calibration.json
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFilter

from eval_harness.image_similarity_gate import (
    DEFAULT_SIMILARITY_THRESHOLD,
    compare_image_similarity,
)


def _synthetic_reference() -> Image.Image:
    image = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((90, 80, 1110, 620), radius=24, fill=(40, 107, 166), outline="black", width=7)
    draw.line((600, 80, 600, 620), fill="white", width=5)
    draw.rounded_rectangle((145, 190, 510, 500), radius=14, fill=(236, 242, 247), outline=(30, 50, 70), width=5)
    draw.rounded_rectangle((690, 190, 1055, 500), radius=14, fill=(236, 242, 247), outline=(30, 50, 70), width=5)
    draw.line((510, 345, 690, 345), fill="white", width=8)
    draw.polygon([(680, 333), (710, 345), (680, 357)], fill="white")
    draw.text((300, 325), "reference", fill=(30, 50, 70))
    draw.text((810, 325), "candidate", fill=(30, 50, 70))
    return image


def _score(reference: Image.Image, candidate: Image.Image) -> float:
    return compare_image_similarity(reference, candidate)["copy_similarity"]


def _jpeg_copy(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as loaded:
        return loaded.convert("RGB")


def _copy_fixtures(reference: Image.Image) -> tuple[dict[str, Image.Image], dict[str, Image.Image]]:
    """Return normalisation copies and deliberately stressed copies separately.

    A threshold must preserve ordinary export changes (resampling and mild
    compression), not promise to catch an arbitrarily tiny thumbnail.  Keeping
    the stress fixtures out of the copy floor prevents a pathological low
    bound from silently weakening the gate.
    """

    normal: dict[str, Image.Image] = {"exact": reference.copy()}
    stress: dict[str, Image.Image] = {}
    for fraction in (0.5,):
        reduced = reference.resize(
            (max(1, round(reference.width * fraction)), max(1, round(reference.height * fraction))),
            Image.Resampling.LANCZOS,
        )
        normal[f"downsample_{fraction:g}"] = reduced.resize(reference.size, Image.Resampling.LANCZOS)
    for fraction in (0.25, 0.10):
        reduced = reference.resize(
            (max(1, round(reference.width * fraction)), max(1, round(reference.height * fraction))),
            Image.Resampling.LANCZOS,
        )
        stress[f"downsample_{fraction:g}"] = reduced.resize(reference.size, Image.Resampling.LANCZOS)
    for quality in (95, 85, 75):
        normal[f"jpeg_{quality}"] = _jpeg_copy(reference, quality)
    normal["blur_1"] = reference.filter(ImageFilter.GaussianBlur(1))
    return normal, stress


def _synthetic_resource_fixtures(reference: Image.Image) -> dict[str, Image.Image]:
    width, height = reference.size
    return {
        "central_icon": reference.crop((width // 3, height // 3, width * 2 // 3, height * 2 // 3)),
        "left_panel": reference.crop((0, 0, width // 2, height)),
        "independent_icon": Image.new("RGBA", (160, 160), (255, 255, 255, 0)),
    }


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "median": round(statistics.median(ordered), 6),
        "max": round(ordered[-1], 6),
    }


def calibrate(reference_path: Path | None, resource_paths: list[Path]) -> dict[str, Any]:
    if reference_path:
        with Image.open(reference_path) as loaded:
            reference = loaded.convert("RGB")
    else:
        reference = _synthetic_reference()
    try:
        copies, stress_copies = _copy_fixtures(reference)
        copy_scores = {name: round(_score(reference, image), 6) for name, image in copies.items()}
        stress_scores = {
            name: round(_score(reference, image), 6) for name, image in stress_copies.items()
        }

        resources: dict[str, Image.Image] = {}
        for path in resource_paths:
            with Image.open(path) as image:
                resources[str(path)] = image.convert("RGBA")
        if not resources:
            resources = _synthetic_resource_fixtures(reference)
        resource_scores = {name: round(_score(reference, image), 6) for name, image in resources.items()}

        copy_floor = min(copy_scores.values())
        resource_ceiling = max(resource_scores.values(), default=0.0)
        separated = resource_ceiling < copy_floor
        midpoint = (resource_ceiling + copy_floor) / 2 if separated else DEFAULT_SIMILARITY_THRESHOLD
        threshold = round(midpoint, 6)
        return {
            "reference": {
                "path": str(reference_path) if reference_path else None,
                "size": [reference.width, reference.height],
            },
            "reference_copy_scores": copy_scores,
            "reference_copy_statistics": _stats(copy_scores.values()),
            "stress_copy_scores": stress_scores,
            "stress_copy_statistics": _stats(stress_scores.values()),
            "allowed_resource_scores": resource_scores,
            "allowed_resource_statistics": _stats(resource_scores.values()),
            "bounds": {
                "reference_copy_floor": round(copy_floor, 6),
                "allowed_resource_ceiling": round(resource_ceiling, 6),
                "separated": separated,
            },
            "separation_midpoint": threshold,
            "runtime_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            "stress_checks": {
                "small_reference_copy": {
                    "similarity": stress_scores.get("downsample_0.1"),
                    "would_trigger_at_runtime_threshold": (
                        stress_scores.get("downsample_0.1", 0.0) >= DEFAULT_SIMILARITY_THRESHOLD
                    ),
                    "interpretation": "stress diagnostic only; not used to set the threshold",
                }
            },
            "interpretation": (
                "The midpoint is a calibration diagnostic; runtime uses the configured conservative threshold. "
                "The artifact gate is driven by the per-image similarity signal rather than image area."
            ),
        }
    finally:
        reference.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--resource", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reference and not args.reference.is_file():
        parser.error(f"reference does not exist: {args.reference}")
    missing = [path for path in args.resource if not path.is_file()]
    if missing:
        parser.error("resource does not exist: " + ", ".join(map(str, missing)))
    result = calibrate(args.reference, args.resource)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
