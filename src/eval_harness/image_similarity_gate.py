"""Image-similarity raster gate for PPTBench.

This module is deliberately separate from :mod:`eval_harness.pptx`.  The
PPTX validator reports package-level structure; this gate checks whether a
candidate embeds a near-copy of the reference image as an unapproved raster.

Each unapproved embedded raster is compared independently with the reference
after outer-whitespace normalization; one raster above the calibrated
similarity threshold is sufficient to trigger the artifact gate.  Approved task
resources are excluded by exact image digest.  The gate deliberately uses image
evidence rather than slide-area heuristics: a large cropped copy and a small
full-slide copy are both evaluated by their visual similarity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
import io
from pathlib import Path
import math
from typing import Any, Iterable, Mapping, Sequence, TypeAlias
import zipfile

import numpy as np
from PIL import Image, ImageFilter, ImageOps


DEFAULT_METRIC_SIZE: tuple[int, int] = (800, 450)
DEFAULT_SIMILARITY_THRESHOLD = 0.90

# A raster source may be supplied directly by a caller, or loaded from the
# ``ppt/media`` members of the PPTX report's ``source_path``.  Keeping this
# type local means the JSON report remains small: image bytes are consumed only
# while the gate is running and are never written into its JSON output.
RasterSource: TypeAlias = Path | Image.Image | bytes | bytearray | memoryview


@dataclass(frozen=True)
class RasterGateConfig:
    """Thresholds for :func:`assess_raster_gate`.

    ``similarity_threshold`` controls the artifact gate.  The gate uses only
    per-image similarity; image area and native-object counts are not gate
    inputs.
    """

    # The threshold is calibrated against a small synthetic fixture matrix:
    # mild lossy reference copies remain above the threshold while ordinary
    # local assets remain below it.  It is configurable until a larger
    # human-labelled calibration set is available.
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    metric_size: tuple[int, int] = DEFAULT_METRIC_SIZE
    crop_outer_whitespace: bool = True
    background_threshold: int = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
        if len(self.metric_size) != 2 or min(self.metric_size) <= 0:
            raise ValueError("metric_size must contain two positive values")
        if not 0 <= self.background_threshold <= 255:
            raise ValueError("background_threshold must be in [0, 255]")


def _as_rgb(source: RasterSource) -> Image.Image:
    """Load a source and composite transparency onto the metric background."""

    if isinstance(source, Image.Image):
        opened = source
        close_opened = False
    else:
        if isinstance(source, (bytes, bytearray, memoryview)):
            source = io.BytesIO(bytes(source))
        opened = Image.open(source)
        close_opened = True
    try:
        loaded = ImageOps.exif_transpose(opened)
        if "A" in loaded.getbands():
            rgba = loaded.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            return background.convert("RGB")
        return loaded.convert("RGB")
    finally:
        if close_opened:
            opened.close()


def _crop_visible_content(image: Image.Image, threshold: int) -> Image.Image:
    """Crop only canvas-colored outer margins, preserving interior gutters."""

    if image.width == 0 or image.height == 0:
        return image.copy()
    array = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]), axis=0)
    background = np.median(border, axis=0)
    foreground = np.max(np.abs(array - background), axis=2) > threshold
    ys, xs = np.where(foreground)
    if not len(xs):
        return image.copy()
    left, top = int(xs.min()), int(ys.min())
    right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
    margin = max(4, round(max(right - left, bottom - top) * 0.02))
    box = (
        max(0, left - margin),
        max(0, top - margin),
        min(image.width, right + margin),
        min(image.height, bottom + margin),
    )
    return image.crop(box)


def normalize_image(
    source: RasterSource,
    size: tuple[int, int] = DEFAULT_METRIC_SIZE,
    *,
    crop_outer_whitespace: bool = True,
    background_threshold: int = 10,
) -> np.ndarray:
    """Return an RGB array in a letterboxed, content-normalized canvas.

    This does not stretch a non-matching aspect ratio.  The content is
    optionally cropped using the same border
    background rule as the Judge composite, then fitted with ``contain``.
    Thus outer-canvas whitespace is not a quality signal while aspect-ratio
    distortion remains visible.
    """

    if len(size) != 2 or min(size) <= 0:
        raise ValueError("size must contain two positive values")
    image = _as_rgb(source)
    if crop_outer_whitespace:
        image = _crop_visible_content(image, background_threshold)
    fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return np.asarray(canvas, dtype=np.float64)


def _block_ssim(reference: np.ndarray, candidate: np.ndarray, block: int = 8) -> float:
    left = np.dot(reference[..., :3], [0.299, 0.587, 0.114])
    right = np.dot(candidate[..., :3], [0.299, 0.587, 0.114])
    height = left.shape[0] - left.shape[0] % block
    width = left.shape[1] - left.shape[1] % block
    if height == 0 or width == 0:
        return 0.0
    left = left[:height, :width].reshape(height // block, block, width // block, block)
    right = right[:height, :width].reshape(height // block, block, width // block, block)
    mu_left = left.mean(axis=(1, 3))
    mu_right = right.mean(axis=(1, 3))
    var_left = left.var(axis=(1, 3))
    var_right = right.var(axis=(1, 3))
    covariance = ((left - mu_left[:, None, :, None]) * (right - mu_right[:, None, :, None])).mean(
        axis=(1, 3)
    )
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    score = ((2 * mu_left * mu_right + c1) * (2 * covariance + c2)) / (
        (mu_left**2 + mu_right**2 + c1) * (var_left + var_right + c2)
    )
    return float(np.clip(score.mean(), -1.0, 1.0))


def _edge_f1(reference: np.ndarray, candidate: np.ndarray) -> float:
    def edges(array: np.ndarray) -> np.ndarray:
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("L")
        image = image.filter(ImageFilter.FIND_EDGES)
        return np.asarray(image) > 28

    left = edges(reference)
    right = edges(candidate)
    left_dilated = np.asarray(Image.fromarray(left).filter(ImageFilter.MaxFilter(3))) > 0
    right_dilated = np.asarray(Image.fromarray(right).filter(ImageFilter.MaxFilter(3))) > 0
    precision = float((right & left_dilated).sum() / max(1, right.sum()))
    recall = float((left & right_dilated).sum() / max(1, left.sum()))
    return 2 * precision * recall / max(1e-9, precision + recall)


def compare_image_similarity(
    reference_path: RasterSource,
    candidate_path: RasterSource,
    *,
    size: tuple[int, int] = DEFAULT_METRIC_SIZE,
    crop_outer_whitespace: bool = True,
    background_threshold: int = 10,
) -> dict[str, float]:
    """Compare two images for near-copy detection.

    The returned ``copy_similarity`` is deliberately separate from the
    benchmark's quality score.  It is a signal for the raster-integrity gate,
    not a replacement for semantic/detail judging.
    """

    reference = normalize_image(
        reference_path,
        size,
        crop_outer_whitespace=crop_outer_whitespace,
        background_threshold=background_threshold,
    )
    candidate = normalize_image(
        candidate_path,
        size,
        crop_outer_whitespace=crop_outer_whitespace,
        background_threshold=background_threshold,
    )
    pixel = float(1.0 - np.abs(reference - candidate).mean() / 255.0)
    ssim = _block_ssim(reference, candidate)
    edge = _edge_f1(reference, candidate)
    copy_similarity = 0.60 * max(0.0, ssim) + 0.25 * max(0.0, pixel) + 0.15 * edge
    return {
        "pixel_similarity": round(pixel, 6),
        "ssim": round(ssim, 6),
        "edge_f1": round(edge, 6),
        "copy_similarity": round(copy_similarity, 6),
    }


def _component_bbox(
    component: Mapping[str, Any],
    report: Mapping[str, Any] | None = None,
) -> Sequence[float] | None:
    """Return a component box in normalized slide coordinates.

    ``extract_pptx_components`` emits both ``bbox_normalized`` and a canvas
    ``bbox``. A few downstream exports retain only one of them, so this helper
    accepts either spelling and converts the latter instead of silently
    treating pixel coordinates as unit-square coordinates.
    """

    value = component.get("bbox_normalized")
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return value
    value = component.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        numbers = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in numbers):
        return None
    # Future reports may use ``bbox`` for normalized coordinates.  Treat
    # values close to the unit square as normalized, including off-slide boxes.
    if max(abs(item) for item in numbers) <= 1.5:
        return numbers
    dimensions = (report or {}).get("slide_size", {})
    width = dimensions.get("width_emu") if isinstance(dimensions, Mapping) else None
    height = dimensions.get("height_emu") if isinstance(dimensions, Mapping) else None
    # The public extractor's bbox is in the fixed normalized canvas.  Only use
    # EMU dimensions when the values clearly exceed that canvas.
    if max(abs(item) for item in numbers) > 100_000 and width and height:
        scale_x, scale_y = float(width), float(height)
    else:
        canvas = (report or {}).get("canvas_size", (1600, 900))
        if not isinstance(canvas, (list, tuple)) or len(canvas) != 2:
            canvas = (1600, 900)
        scale_x, scale_y = float(canvas[0]), float(canvas[1])
    if scale_x <= 0 or scale_y <= 0:
        return None
    left, top, right, bottom = numbers
    return (left / scale_x, top / scale_y, right / scale_x, bottom / scale_y)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _picture_components(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    components = report.get("components", [])
    if not isinstance(components, Iterable) or isinstance(components, (str, bytes)):
        return result
    for component in components:
        if not isinstance(component, Mapping):
            continue
        kind = str(component.get("kind", "")).casefold()
        if kind == "picture" or component.get("editable") is False:
            result.append(component)
    return result


def embedded_raster_sources(report: Mapping[str, Any]) -> dict[str, bytes]:
    """Load embedded PPTX media keyed by SHA-256 when a source path is present.

    Component reports intentionally contain only image sizes and digests.  The
    gate can inspect actual media without changing that public
    JSON schema by reading ``ppt/media`` from the original package on demand.
    Invalid/missing packages return an empty mapping and are reported as an
    unavailable-source diagnostic by :func:`raster_similarity_diagnostics`.
    """

    raw_path = report.get("source_path")
    if not raw_path:
        return {}
    path = Path(str(raw_path)).expanduser()
    if not path.is_file() or not zipfile.is_zipfile(path):
        return {}
    result: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                name = info.filename.casefold()
                if not name.startswith("ppt/media/") or info.is_dir():
                    continue
                payload = archive.read(info)
                result.setdefault(_sha256_bytes(payload), payload)
    except (OSError, KeyError, zipfile.BadZipFile, RuntimeError):
        return {}
    return result


def _coerce_raster_source(value: Any) -> RasterSource | None:
    """Convert a JSON-friendly source value into a supported image source."""

    if isinstance(value, (Path, Image.Image, bytes, bytearray, memoryview)):
        return value
    if isinstance(value, str) and value:
        return Path(value).expanduser()
    return None


def _report_raster_sources(report: Mapping[str, Any]) -> dict[str, RasterSource]:
    """Read optional caller-provided source paths from a report mapping."""

    result: dict[str, RasterSource] = {}
    for field in ("raster_sources", "image_sources"):
        value = report.get(field)
        if not isinstance(value, Mapping):
            continue
        for key, source in value.items():
            normalized = _coerce_raster_source(source)
            if normalized is not None:
                text_key = str(key)
                result[text_key] = normalized
                result.setdefault(text_key.casefold(), normalized)
    return result


def _find_raster_source(
    component: Mapping[str, Any],
    index: int,
    *,
    explicit: Mapping[str, RasterSource],
    embedded: Mapping[str, bytes],
) -> tuple[RasterSource | None, str | None]:
    """Resolve a picture's bytes by component id, digest, or embedded media."""

    digest = str(component.get("image_sha256", "")).casefold()
    component_id = str(component.get("component_id", ""))
    keys = [
        key
        for key in (
            component_id,
            component_id.casefold(),
            digest,
            str(index),
            str(index + 1),
            f"picture-{index + 1}",
        )
        if key
    ]
    for key in keys:
        if key in explicit:
            return explicit[key], f"explicit:{key}"
    if digest and digest in embedded:
        return embedded[digest], "pptx-media:" + digest
    return None, None


def raster_similarity_diagnostics(
    reference_path: RasterSource,
    pptx_report: Mapping[str, Any],
    *,
    approved_resource_hashes: Iterable[str] = (),
    raster_sources: Mapping[str, RasterSource | str] | None = None,
    config: RasterGateConfig | None = None,
) -> list[dict[str, Any]]:
    """Compare every unapproved embedded raster with the normalized reference.

    The report normally contains only media digests, so sources are resolved in
    this order: explicit ``raster_sources`` keyed by component id/digest/index,
    optional source mappings embedded in the report, then ``ppt/media`` members
    of ``report['source_path']``.  Approved hashes are recorded as exempt and
    never decoded or compared.
    """

    settings = config or RasterGateConfig()
    approved = {str(value).casefold() for value in approved_resource_hashes if str(value)}
    explicit: dict[str, RasterSource] = _report_raster_sources(pptx_report)
    for key, source in (raster_sources or {}).items():
        normalized = _coerce_raster_source(source)
        if normalized is not None:
            text_key = str(key)
            explicit[text_key] = normalized
            explicit.setdefault(text_key.casefold(), normalized)
    embedded = embedded_raster_sources(pptx_report)
    rows: list[dict[str, Any]] = []
    decoded_cache: dict[str, tuple[str, dict[str, float] | None, str | None]] = {}
    for index, component in enumerate(_picture_components(pptx_report)):
        digest = str(component.get("image_sha256", "")).casefold()
        row: dict[str, Any] = {
            "index": index,
            "component_id": str(component.get("component_id", f"picture-{index + 1}")),
            "image_sha256": digest,
            "approved": bool(digest and digest in approved),
            "bbox_normalized": list(_component_bbox(component, pptx_report) or ()),
        }
        if row["approved"]:
            row.update({"source_status": "approved_exempt", "matched": False})
            rows.append(row)
            continue
        source, source_key = _find_raster_source(
            component,
            index,
            explicit=explicit,
            embedded=embedded,
        )
        if source is None:
            row.update(
                {
                    "source_status": "unavailable",
                    "matched": False,
                    "source_key": None,
                }
            )
            rows.append(row)
            continue
        row["source_key"] = source_key
        cache_key = digest or source_key or f"index:{index}"
        cached = decoded_cache.get(cache_key)
        if cached is not None:
            source_status, metrics, error = cached
            if metrics is not None:
                row.update(metrics, source_status=source_status, matched=metrics["copy_similarity"] >= settings.similarity_threshold)
            else:
                row.update(source_status=source_status, matched=False, error=error)
            rows.append(row)
            continue
        try:
            metrics = compare_image_similarity(
                reference_path,
                source,
                size=settings.metric_size,
                crop_outer_whitespace=settings.crop_outer_whitespace,
                background_threshold=settings.background_threshold,
            )
        except Exception as exc:  # malformed media is a diagnostic, not a judge outage
            error = f"{type(exc).__name__}: {exc}"
            decoded_cache[cache_key] = ("undecodable", None, error)
            row.update(
                {
                    "source_status": "undecodable",
                    "matched": False,
                    "error": error,
                }
            )
        else:
            decoded_cache[cache_key] = ("decoded", metrics, None)
            row.update(
                metrics,
                source_status="decoded",
                matched=metrics["copy_similarity"] >= settings.similarity_threshold,
            )
        rows.append(row)
    return rows


def assess_raster_gate(
    reference_path: RasterSource,
    candidate_path: RasterSource | None,
    pptx_report: Mapping[str, Any],
    *,
    approved_resource_hashes: Iterable[str] = (),
    raster_sources: Mapping[str, RasterSource | str] | None = None,
    config: RasterGateConfig | None = None,
) -> dict[str, Any]:
    """Assess whether any unapproved embedded raster copies the reference.

    The function never changes ``pptx_report``. The gate is driven only by
    per-image comparisons. ``candidate_path`` is retained for API compatibility
    and contributes a global similarity diagnostic when supplied.
    """

    settings = config or RasterGateConfig()
    approved = tuple(str(value).casefold() for value in approved_resource_hashes if str(value))
    similarity = (
        compare_image_similarity(
            reference_path,
            candidate_path,
            size=settings.metric_size,
            crop_outer_whitespace=settings.crop_outer_whitespace,
            background_threshold=settings.background_threshold,
        )
        if candidate_path is not None
        else None
    )
    per_raster = raster_similarity_diagnostics(
        reference_path,
        pptx_report,
        approved_resource_hashes=approved,
        raster_sources=raster_sources,
        config=settings,
    )
    similarity_match = any(bool(row.get("matched")) for row in per_raster)
    source_available = any(row.get("source_status") == "decoded" for row in per_raster)
    unapproved_rows = [row for row in per_raster if not row.get("approved")]
    violations: list[str] = []
    for row in per_raster:
        if row.get("matched"):
            violations.append(
                "unapproved raster "
                + str(row.get("component_id"))
                + " matches the reference above the similarity threshold"
            )
    triggered = similarity_match
    if triggered:
        reason = "unapproved raster media matches reference above similarity threshold"
    elif not per_raster:
        reason = "candidate contains no raster media"
    elif not unapproved_rows:
        reason = "all embedded raster media is approved"
    elif not source_available:
        reason = "no decodable unapproved raster source was available"
    else:
        reason = "no unapproved raster exceeded similarity threshold"
    return {
        "schema": "pptbench-image-similarity-raster-gate-v1",
        "policy": "per-unapproved-raster-similarity",
        "triggered": triggered,
        "passed": not triggered,
        "reason": reason,
        "similarity": similarity,
        "raster_similarity": per_raster,
        "signals": {
            "similarity_match": similarity_match,
            "source_available": source_available,
            "embedded_media_count": len(per_raster),
            "unapproved_media_count": len(unapproved_rows),
        },
        "thresholds": {
            "similarity_threshold": settings.similarity_threshold,
            "metric_size": list(settings.metric_size),
            "crop_outer_whitespace": settings.crop_outer_whitespace,
            "background_threshold": settings.background_threshold,
        },
        "violations": violations,
    }


def validate_image_similarity_gate(
    reference_path: RasterSource,
    candidate_path: RasterSource | None,
    pptx_report: Mapping[str, Any],
    *,
    approved_resource_hashes: Iterable[str] = (),
    raster_sources: Mapping[str, RasterSource | str] | None = None,
    config: RasterGateConfig | None = None,
) -> dict[str, Any]:
    """API-compatible alias for :func:`assess_raster_gate`."""

    return assess_raster_gate(
        reference_path,
        candidate_path,
        pptx_report,
        approved_resource_hashes=approved_resource_hashes,
        raster_sources=raster_sources,
        config=config,
    )


def raster_gate_violations(
    reference_path: RasterSource,
    candidate_path: RasterSource | None,
    pptx_report: Mapping[str, Any],
    *,
    approved_resource_hashes: Iterable[str] = (),
    raster_sources: Mapping[str, RasterSource | str] | None = None,
    config: RasterGateConfig | None = None,
) -> list[str]:
    """Return gate messages suitable for ``violations.extend(...)``."""

    return list(
        assess_raster_gate(
            reference_path,
            candidate_path,
            pptx_report,
            approved_resource_hashes=approved_resource_hashes,
            raster_sources=raster_sources,
            config=config,
        )["violations"]
    )


# Short aliases retain descriptive names for callers that prefer an explicit
# gate function.
image_similarity = compare_image_similarity
raster_gate = assess_raster_gate
compare = compare_image_similarity


__all__ = [
    "DEFAULT_METRIC_SIZE",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "RasterSource",
    "RasterGateConfig",
    "assess_raster_gate",
    "compare",
    "compare_image_similarity",
    "embedded_raster_sources",
    "image_similarity",
    "normalize_image",
    "raster_similarity_diagnostics",
    "raster_gate",
    "raster_gate_violations",
    "validate_image_similarity_gate",
]
