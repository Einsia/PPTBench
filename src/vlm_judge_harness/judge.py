"""Concurrent, blind, three-stage VLM judging for PPTBench renders."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps


DEDUCTION_KEYS = (
    "layout_and_composition",
    "text_and_typography",
    "local_graphics_and_nodes",
)
SCORING_CATEGORIES: dict[str, tuple[str, ...]] = {
    "layout_and_composition": (
        "panel_misalignment",
        "node_layout_deviation",
        "panel_style_error",
    ),
    "text_and_typography": (
        "spelling_and_omission",
        "text_overflow_or_edge_touch",
        "improper_typography",
    ),
    "local_graphics_and_nodes": (
        "node_shape_error",
        "node_style_error",
        "node_shadow_error",
        "vector_and_icon_confusion",
        "line_and_arrow_details",
    ),
}
ISSUE_TYPES: dict[str, tuple[str, ...]] = {
    "panel_misalignment": (
        "canvas_crop",
        "canvas_aspect_ratio",
        "canvas_orientation",
        "global_scale",
        "global_centering",
        "outer_whitespace",
        "missing_panel",
        "extra_panel",
        "duplicate_panel",
        "substituted_panel",
        "panel_order",
        "panel_position",
        "panel_size",
        "panel_proportion",
        "panel_spacing",
        "panel_alignment",
        "separator_position",
        "panel_overlap",
        "panel_clipping",
        "conspicuous_blank_region",
        "other_panel_layout",
    ),
    "node_layout_deviation": (
        "missing_node",
        "extra_node",
        "duplicate_node",
        "substituted_node",
        "node_position",
        "node_spacing",
        "node_alignment",
        "node_distribution",
        "reading_order",
        "grid_arrangement",
        "grouping",
        "nesting",
        "padding",
        "hierarchy",
        "node_overlap",
        "node_occlusion",
        "z_order",
        "local_scale",
        "local_mirroring",
        "local_rotation",
        "other_node_layout",
    ),
    "panel_style_error": (
        "panel_fill",
        "panel_border_color",
        "panel_border_width",
        "panel_border_pattern",
        "panel_opacity",
        "panel_gradient",
        "panel_texture",
        "panel_clipping_mask",
        "panel_corner_radius",
        "panel_shadow",
        "panel_glow",
        "panel_bevel",
        "panel_outer_frame",
        "panel_header_strip",
        "panel_title_tab",
        "panel_separator_style",
        "other_panel_style",
    ),
    "spelling_and_omission": (
        "missing_text",
        "extra_text",
        "substituted_text",
        "duplicated_text",
        "reordered_text",
        "incorrect_number",
        "incorrect_identifier",
        "capitalization",
        "punctuation",
        "abbreviation",
        "math_symbol",
        "greek_letter",
        "subscript",
        "superscript",
        "accent",
        "unit",
        "special_glyph",
        "wrong_label_assignment",
        "other_text_content",
    ),
    "text_overflow_or_edge_touch": (
        "edge_touch",
        "text_clipping",
        "text_truncation",
        "outside_text_box",
        "text_text_overlap",
        "text_node_overlap",
        "text_icon_overlap",
        "text_connector_overlap",
        "text_arrowhead_overlap",
        "text_cell_overlap",
        "cut_off_glyph",
        "insufficient_text_padding",
        "other_text_fit",
    ),
    "improper_typography": (
        "font_family",
        "font_size",
        "font_weight",
        "font_style",
        "text_color",
        "text_opacity",
        "text_contrast",
        "letter_spacing",
        "word_spacing",
        "line_height",
        "baseline",
        "kerning",
        "math_typesetting",
        "horizontal_alignment",
        "vertical_alignment",
        "text_anchoring",
        "internal_padding",
        "unexpected_wrap",
        "mid_word_break",
        "isolated_character_line",
        "hyphenation",
        "line_count",
        "line_order",
        "paragraph_width",
        "text_rotation",
        "text_orientation",
        "curved_text",
        "text_blur",
        "text_aliasing",
        "other_typography",
    ),
    "node_shape_error": (
        "shape_type",
        "node_aspect_ratio",
        "node_size",
        "node_rotation",
        "node_orientation",
        "node_skew",
        "node_symmetry",
        "node_corner_radius",
        "notch_geometry",
        "folded_corner",
        "cutout",
        "tab_geometry",
        "port_geometry",
        "handle_geometry",
        "tail_geometry",
        "pointer_geometry",
        "missing_shape_primitive",
        "extra_shape_primitive",
        "other_node_geometry",
    ),
    "node_style_error": (
        "node_fill",
        "node_stroke",
        "node_border_width",
        "node_border_pattern",
        "node_color",
        "node_opacity",
        "node_transparency",
        "node_gradient",
        "node_texture",
        "node_hatch",
        "node_highlight",
        "node_contrast",
        "state_color",
        "palette_mapping",
        "three_dimensional_treatment",
        "node_bevel",
        "node_gloss",
        "internal_z_order",
        "style_consistency",
        "other_node_style",
    ),
    "node_shadow_error": (
        "shadow_presence",
        "shadow_offset",
        "shadow_direction",
        "shadow_blur",
        "shadow_spread",
        "shadow_color",
        "shadow_opacity",
        "glow_presence",
        "glow_style",
        "hard_vs_soft_shadow",
        "other_shadow",
    ),
    "vector_and_icon_confusion": (
        "missing_icon",
        "extra_icon",
        "substituted_icon",
        "malformed_icon",
        "mirrored_icon",
        "rotated_icon",
        "distorted_icon",
        "icon_state",
        "legend_symbol",
        "legend_mapping",
        "vector_blur",
        "vector_pixelation",
        "jagged_vector",
        "rasterized_vector",
        "grid_dimensions",
        "grid_count",
        "cell_size",
        "cell_spacing",
        "cell_border",
        "cell_alignment",
        "primitive_count",
        "primitive_order",
        "primitive_position",
        "primitive_color",
        "primitive_fill",
        "primitive_pattern",
        "mini_diagram_topology",
        "chart_axes",
        "chart_ticks",
        "chart_legend",
        "chart_series",
        "chart_marker",
        "raster_crop",
        "raster_stretch",
        "raster_blur",
        "raster_replacement",
        "other_local_graphic",
    ),
    "line_and_arrow_details": (
        "missing_connector",
        "extra_connector",
        "wrong_source_attachment",
        "wrong_target_attachment",
        "wrong_port",
        "detached_endpoint",
        "endpoint_side",
        "connector_direction",
        "arrowhead_endpoint",
        "intermediate_arrowhead",
        "bidirectionality",
        "arrowhead_count",
        "arrowhead_shape",
        "arrowhead_size",
        "arrowhead_fill",
        "arrowhead_outline",
        "arrowhead_color",
        "arrowhead_angle",
        "route_shape",
        "bezier_curvature",
        "bend_count",
        "bend_location",
        "bend_radius",
        "loop_shape",
        "bypass_shape",
        "parallel_spacing",
        "line_pattern",
        "dash_sequence",
        "dash_gap",
        "line_width",
        "line_color",
        "line_opacity",
        "line_cap",
        "line_join",
        "line_blur",
        "line_glow",
        "line_shadow",
        "junction_dot",
        "split_merge_marker",
        "crossing_bridge",
        "intersection_treatment",
        "connector_overlap_text",
        "connector_overlap_node",
        "connector_clipping",
        "border_confusion",
        "other_connector",
    ),
}
PROTOCOL_ID = "pptbench-final"
RESULT_SCHEMA = "pptbench-judge-result"
GATE_REUSE_PROTOCOL_ID = PROTOCOL_ID

TOOL_INSPECTION_GUIDANCE = """\
Tool-assisted image inspection is available and expected:

- Three images are attached in this order: the normalized side-by-side
  `comparison.png`, the unmodified `reference.png`, and the unmodified
  `candidate.png`. All three files are also available in the current writable
  workspace. `comparison-metadata.json` records the crop and scale used to
  build the normalized comparison.
- The normalized comparison intentionally removes outer canvas whitespace and
  absolute placement against the outermost image edge. Never score outer-edge
  whitespace, global centering within the source canvas, or absolute content
  scale. Use the raw images only to recover detail lost by resizing and to
  verify that the merge did not create an apparent difference.
- Start from the full images. When any connector, arrowhead, text box, icon, or
  repeated module is too small or dense to judge confidently, use the terminal
  tools to write and run a small image-processing script (for example with
  Python and Pillow) that crops, enlarges, segments colors, extracts bounding
  boxes, or measures relative positions and padding.
- Choose crop boundaries adaptively from the actual diagram. Create as many or
  as few crops as needed; do not assume a fixed grid. Preserve the left
  reference/right candidate correspondence in every paired crop.
- Register corresponding panels or landmarks before comparing pixel extents.
  A direct pixel difference or raw bounding-box difference between unregistered
  images is not evidence of a layout defect. Express measurements as ratios
  inside the corresponding panel or node, not distances to the outer canvas.
- You may create scripts and derived images freely inside the workspace. Read
  and use the scripts' numerical output. If the runtime exposes an image-viewing
  tool, inspect generated crops with it; otherwise rely on the attached
  full-resolution originals plus quantitative measurements rather than
  claiming to have visually reopened a derived file.
- Keep all scripts and derived images inside the current workspace. Do not
  modify `comparison.png`, access unrelated files, or use the network.
- Return only the JSON required by the task after completing the inspection.
"""


def with_tool_inspection_guidance(prompt: str) -> str:
    """Append the stable self-directed crop protocol to a judge prompt."""
    return f"{prompt.rstrip()}\n\n{TOOL_INSPECTION_GUIDANCE.strip()}"


GATE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["gate_keeper"],
    "properties": {
        "gate_keeper": {
            "type": "object",
            "additionalProperties": False,
            "required": ["triggered", "reason"],
            "properties": {
                "triggered": {"type": "boolean"},
                "reason": {"type": "string"},
            },
        }
    },
}

SCORE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["deductions", "summary_feedback"],
    "properties": {
        "deductions": {
            "type": "object",
            "additionalProperties": False,
            "required": list(DEDUCTION_KEYS),
            "properties": {
                dimension: {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(categories),
                    "properties": {
                        category: {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["cases", "ratio"],
                            "properties": {
                                "cases": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": [
                                            "issue_type",
                                            "location",
                                            "candidate",
                                            "reference",
                                            "affected_count",
                                        ],
                                        "properties": {
                                            "issue_type": {
                                                "type": "string",
                                                "enum": list(ISSUE_TYPES[category]),
                                            },
                                            "location": {"type": "string", "minLength": 1},
                                            "candidate": {"type": "string", "minLength": 1},
                                            "reference": {"type": "string", "minLength": 1},
                                            "affected_count": {
                                                "type": "integer",
                                                "minimum": 1,
                                            },
                                        },
                                    },
                                },
                                "ratio": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                            },
                        }
                        for category in categories
                    },
                }
                for dimension, categories in SCORING_CATEGORIES.items()
            },
        },
        "summary_feedback": {"type": "string"},
    },
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    summary_path: Path


@dataclass(frozen=True)
class Judge:
    judge_id: str
    model: str
    effort: str


@dataclass(frozen=True)
class Job:
    task_id: str
    candidate: Candidate
    judge: Judge
    reference_path: Path
    candidate_path: Path
    composite_path: Path


def read_task_ids(path: Path) -> list[str]:
    task_ids: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.split("#", 1)[0].strip()
        if not value:
            continue
        if re.fullmatch(r"\d{1,4}", value):
            value = f"task_{int(value):04d}"
        if not re.fullmatch(r"task_\d{4}", value):
            raise ValueError(f"Invalid task ID in {path}: {value!r}")
        if value not in seen:
            task_ids.append(value)
            seen.add(value)
    return task_ids


def load_candidate_screenshots(summary_path: Path) -> dict[str, Path]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    screenshots: dict[str, Path] = {}
    records = payload.get("records")
    if records is None:
        records = payload.get("cases", [])
    for record in records:
        if record.get("status") != "complete":
            continue
        result_path = Path(record["result_path"])
        if not result_path.is_absolute():
            result_path = summary_path.parent / result_path
        case_dir = result_path.parent.parent
        screenshot = case_dir / "final" / "screenshot.png"
        if not screenshot.is_file():
            screenshot = case_dir / "single-run" / "screenshot.png"
        if screenshot.is_file():
            screenshots[str(record["task_id"])] = screenshot
    return screenshots


def _visible_content(
    image: Image.Image,
    threshold: int = 10,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    rgb = Image.alpha_composite(white, rgba).convert("RGB")
    array = np.asarray(rgb, dtype=np.int16)
    if array.size == 0:
        return rgb, (0, 0, rgb.width, rgb.height)

    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]), axis=0)
    background = np.median(border, axis=0)
    foreground = np.max(np.abs(array - background), axis=2) > threshold

    # Do not let a lone antialiasing/export speck determine the content box.
    # The mask is used only to find the crop (the original RGB pixels are kept),
    # so removing isolated pixels here cannot erase visible content inside it.
    # A real thin line still has an adjacent foreground pixel and remains eligible.
    padded = np.pad(foreground, 1, mode="constant", constant_values=False)
    adjacent = np.zeros_like(foreground, dtype=np.uint8)
    height, width = foreground.shape
    for y_offset in range(3):
        for x_offset in range(3):
            if x_offset == 1 and y_offset == 1:
                continue
            adjacent += padded[y_offset : y_offset + height, x_offset : x_offset + width]
    supported_foreground = foreground & (adjacent > 0)
    if supported_foreground.any():
        foreground = supported_foreground

    ys, xs = np.where(foreground)
    if not len(xs):
        return rgb, (0, 0, rgb.width, rgb.height)

    left, top = int(xs.min()), int(ys.min())
    right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
    margin = max(4, round(max(right - left, bottom - top) * 0.02))
    box = (
        max(0, left - margin),
        max(0, top - margin),
        min(rgb.width, right + margin),
        min(rgb.height, bottom + margin),
    )
    return rgb.crop(box), box


def make_blind_composite(
    reference_path: Path,
    candidate_path: Path,
    output_path: Path,
    *,
    panel_size: tuple[int, int] = (1200, 900),
) -> dict[str, Any]:
    """Crop pure canvas margins, normalize scale, and concatenate left/right."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_meta: list[dict[str, Any]] = []
    panels: list[Image.Image] = []
    for path in (reference_path, candidate_path):
        with Image.open(path) as original:
            original_size = original.size
            content, crop_box = _visible_content(original)
            normalized = ImageOps.contain(content, panel_size, Image.Resampling.LANCZOS)
            panel = Image.new("RGB", panel_size, "white")
            xy = (
                (panel_size[0] - normalized.width) // 2,
                (panel_size[1] - normalized.height) // 2,
            )
            panel.paste(normalized, xy)
            panels.append(panel)
            source_meta.append(
                {
                    "source_size": list(original_size),
                    "content_crop": list(crop_box),
                    "normalized_size": list(normalized.size),
                    "panel_offset": list(xy),
                }
            )

    divider = 4
    composite = Image.new("RGB", (panel_size[0] * 2 + divider, panel_size[1]), "white")
    composite.paste(panels[0], (0, 0))
    composite.paste(
        Image.new("RGB", (divider, panel_size[1]), (185, 185, 185)),
        (panel_size[0], 0),
    )
    composite.paste(panels[1], (panel_size[0] + divider, 0))
    composite.save(output_path, format="PNG", compress_level=1)
    return {
        "protocol": "crop-background-contain",
        "panel_size": list(panel_size),
        "composite_size": list(composite.size),
        "left": source_meta[0],
        "right": source_meta[1],
    }


def validate_gate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"gate_keeper"}:
        raise ValueError("Gate output must contain only gate_keeper")
    gate = payload["gate_keeper"]
    if not isinstance(gate, dict) or set(gate) != {"triggered", "reason"}:
        raise ValueError("Invalid gate_keeper")
    if type(gate["triggered"]) is not bool or not isinstance(gate["reason"], str):
        raise ValueError("Invalid gate_keeper values")
    if gate["triggered"] and not gate["reason"].strip():
        raise ValueError("A triggered gate must include a reason")
    return payload


def validate_score(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Output is not a JSON object")
    if set(payload) != {"deductions", "summary_feedback"}:
        raise ValueError("Scoring output keys do not match the required schema")
    deductions = payload["deductions"]
    if not isinstance(deductions, dict) or set(deductions) != set(DEDUCTION_KEYS):
        raise ValueError("Invalid deductions")
    for dimension, categories_expected in SCORING_CATEGORIES.items():
        categories = deductions[dimension]
        if not isinstance(categories, dict) or set(categories) != set(categories_expected):
            raise ValueError(f"Invalid deduction categories for {dimension}")
        for category, observation in categories.items():
            if not isinstance(observation, dict) or set(observation) != {"cases", "ratio"}:
                raise ValueError(f"Deduction category {category} must contain only cases and ratio")
            items = observation["cases"]
            ratio = observation["ratio"]
            if not isinstance(items, list):
                raise ValueError(f"Deduction category {category}.cases must be a list")
            required_case_keys = {
                "issue_type",
                "location",
                "candidate",
                "reference",
                "affected_count",
            }
            for item in items:
                if not isinstance(item, dict) or set(item) != required_case_keys:
                    raise ValueError(
                        f"Deduction category {category} contains an invalid structured case"
                    )
                if item["issue_type"] not in ISSUE_TYPES[category]:
                    raise ValueError(f"Invalid issue_type {item['issue_type']!r} for {category}")
                for key in ("location", "candidate", "reference"):
                    if not isinstance(item[key], str) or not item[key].strip():
                        raise ValueError(f"Deduction category {category}.{key} must be non-empty")
                if type(item["affected_count"]) is not int or item["affected_count"] < 1:
                    raise ValueError(
                        f"Deduction category {category}.affected_count must be a positive integer"
                    )
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise ValueError(f"Deduction category {category}.ratio must be numeric")
            if not math.isfinite(float(ratio)) or not 0.0 <= float(ratio) <= 1.0:
                raise ValueError(f"Deduction category {category}.ratio must be in [0, 1]")
            if abs(float(ratio) - round(float(ratio), 2)) > 1e-9:
                raise ValueError(
                    f"Deduction category {category}.ratio must have at most two decimals"
                )
            if not items and float(ratio) != 0.0:
                raise ValueError(
                    f"Deduction category {category}.ratio must be zero when cases is empty"
                )
    if not isinstance(payload["summary_feedback"], str):
        raise ValueError("summary_feedback must be a string")
    return payload


def _result_path(output_dir: Path, job: Job) -> Path:
    return (
        output_dir
        / "results"
        / job.judge.judge_id
        / job.candidate.candidate_id
        / f"{job.task_id}.json"
    )


def _reference_path(dataset_root: Path, task_id: str) -> Path:
    """Resolve a materialized reference under the documented dataset root."""

    candidates = (
        dataset_root / task_id / "reference.png",
        dataset_root / "tasks" / task_id / "reference.png",
        dataset_root / "benchmark" / "tasks" / task_id / "reference.png",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Return the canonical direct-root path so a missing-reference diagnostic is
    # stable and points users at the documented layout.
    return candidates[0]


def _load_reused_gate_stages(reuse_root: Path, job: Job) -> dict[str, Any]:
    """Load and validate gate decisions produced by the final protocol."""
    source_path = _result_path(reuse_root, job)
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot reuse gates from {source_path}: {exc}") from exc

    source_protocol_id = payload.get("protocol_id")
    if source_protocol_id != GATE_REUSE_PROTOCOL_ID:
        raise ValueError(
            f"Gate reuse protocol ID mismatch in {source_path}: "
            f"{source_protocol_id!r}, expected {GATE_REUSE_PROTOCOL_ID!r}"
        )

    expected_identity = {
        "task_id": job.task_id,
        "candidate_id": job.candidate.candidate_id,
        "judge_model": job.judge.model,
        "judge_effort": job.judge.effort,
    }
    for key, expected in expected_identity.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Gate reuse identity mismatch in {source_path}: "
                f"{key}={payload.get(key)!r}, expected {expected!r}"
            )
    composite_sha256 = hashlib.sha256(job.composite_path.read_bytes()).hexdigest()
    if payload.get("composite_sha256") != composite_sha256:
        raise ValueError(f"Gate reuse composite mismatch in {source_path}")

    stages = payload.get("stages")
    if not isinstance(stages, dict):
        raise ValueError(f"Gate reuse source has no stages object: {source_path}")
    semantic = stages.get("semantic_gate")
    if not isinstance(semantic, dict) or semantic.get("status") != "complete":
        raise ValueError(f"Gate reuse source has no complete semantic gate: {source_path}")
    validate_gate(semantic.get("output"))

    render = stages.get("render_gate")
    if not semantic["output"]["gate_keeper"]["triggered"]:
        if not isinstance(render, dict) or render.get("status") != "complete":
            raise ValueError(f"Gate reuse source has no complete render gate: {source_path}")
        validate_gate(render.get("output"))

    return {
        "source_path": str(source_path),
        "source_protocol_id": payload.get("protocol_id", ""),
        "semantic_gate": semantic,
        "render_gate": render,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not (
            payload.get("schema") == RESULT_SCHEMA
            and payload.get("protocol_id") == PROTOCOL_ID
            and payload.get("status") == "complete"
        ):
            return False
        # Numeric scores are produced only by issue_ranker; this result stores
        # the structured findings and gate decisions.
        if "score" in payload:
            return False
        stages = payload.get("stages")
        if not isinstance(stages, dict):
            return False
        final_status = payload.get("final_status")
        if final_status == "scored":
            semantic = stages.get("semantic_gate")
            render = stages.get("render_gate")
            scoring = stages.get("scoring")
            if not isinstance(semantic, dict) or not isinstance(render, dict):
                return False
            if semantic.get("status") != "complete" or render.get("status") != "complete":
                return False
            semantic_gate = validate_gate(semantic.get("output"))["gate_keeper"]
            render_gate = validate_gate(render.get("output"))["gate_keeper"]
            if semantic_gate["triggered"] or render_gate["triggered"]:
                return False
            if not isinstance(scoring, dict) or scoring.get("status") != "complete":
                return False
            validate_score(scoring.get("output"))
        elif final_status == "semantic_gate":
            semantic = stages.get("semantic_gate")
            if not isinstance(semantic, dict) or semantic.get("status") != "complete":
                return False
            if not validate_gate(semantic.get("output"))["gate_keeper"]["triggered"]:
                return False
        elif final_status == "render_gate":
            semantic = stages.get("semantic_gate")
            render = stages.get("render_gate")
            if not isinstance(semantic, dict) or not isinstance(render, dict):
                return False
            if semantic.get("status") != "complete" or render.get("status") != "complete":
                return False
            if validate_gate(semantic.get("output"))["gate_keeper"]["triggered"]:
                return False
            if not validate_gate(render.get("output"))["gate_keeper"]["triggered"]:
                return False
        elif final_status == "artifact_gate":
            # Missing candidates are represented by a synthetic, gate-only
            # result; no VLM stage is run and therefore no gate output exists.
            if payload.get("artifact_gate") is not True:
                return False
            if any(
                not isinstance(stage, dict) or stage.get("status") != "skipped"
                for stage in stages.values()
            ):
                return False
        else:
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
        return False


def _extract_usage(events: bytes) -> dict[str, int]:
    for raw in reversed(events.decode("utf-8", errors="replace").splitlines()):
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        usage = event.get("usage")
        if event.get("type") == "turn.completed" and isinstance(usage, dict):
            return {
                str(key): int(value)
                for key, value in usage.items()
                if isinstance(value, (int, float))
            }
    return {}


async def _run_stage(
    job: Job,
    *,
    stage: str,
    input_path: Path,
    prompt: str,
    schema_path: Path,
    validator: Callable[[Any], dict[str, Any]],
    output_dir: Path,
    codex_bin: str,
    timeout_seconds: int,
    max_attempts: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    stage_prompt = with_tool_inspection_guidance(prompt)
    cache_identity = {
        "protocol_id": PROTOCOL_ID,
        "judge_model": job.judge.model,
        "judge_effort": job.judge.effort,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(stage_prompt.encode("utf-8")).hexdigest(),
    }
    raw_dir = (
        output_dir / "raw" / job.judge.judge_id / job.candidate.candidate_id / job.task_id / stage
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    cached_path = raw_dir / "stage-result.json"
    if cached_path.is_file():
        try:
            cached = json.loads(cached_path.read_text(encoding="utf-8"))
            if cached.get("status") == "complete" and all(
                cached.get(key) == expected for key, expected in cache_identity.items()
            ):
                validator(cached["output"])
                return cached
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    workspace_id = hashlib.sha256(
        f"{job.task_id}\0{job.candidate.candidate_id}\0{job.judge.judge_id}".encode()
    ).hexdigest()[:20]
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        workspace = output_dir / "workspaces" / workspace_id / stage / f"attempt-{attempt}"
        workspace.mkdir(parents=True, exist_ok=True)
        comparison_path = workspace / "comparison.png"
        if not comparison_path.is_file():
            shutil.copy2(input_path, comparison_path)
        reference_path = workspace / "reference.png"
        if not reference_path.is_file():
            shutil.copy2(job.reference_path, reference_path)
        candidate_path = workspace / "candidate.png"
        if not candidate_path.is_file():
            shutil.copy2(job.candidate_path, candidate_path)
        comparison_metadata_path = workspace / "comparison-metadata.json"
        source_metadata_path = job.composite_path.with_suffix(".json")
        if source_metadata_path.is_file() and not comparison_metadata_path.is_file():
            shutil.copy2(source_metadata_path, comparison_metadata_path)
        response_path = workspace / "response.json"
        event_path = raw_dir / f"attempt-{attempt}-events.jsonl"
        stderr_path = raw_dir / f"attempt-{attempt}-stderr.log"
        command = [
            codex_bin,
            "exec",
            "--model",
            job.judge.model,
            "--config",
            f'model_reasoning_effort="{job.judge.effort}"',
            "--config",
            'service_tier="priority"',
            "--config",
            'approval_policy="never"',
            "--strict-config",
            "--ephemeral",
            "--ignore-rules",
            "--sandbox",
            "workspace-write",
            "--cd",
            str(workspace),
            "--skip-git-repo-check",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(response_path),
            "--json",
            "--image",
            str(comparison_path),
            str(reference_path),
            str(candidate_path),
            "--",
            stage_prompt,
        ]
        async with semaphore:
            started = time.monotonic()
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=workspace,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                stdout, stderr = await process.communicate()
                last_error = f"Timed out after {timeout_seconds}s"
            else:
                last_error = f"Codex exited with status {process.returncode}"
            duration = time.monotonic() - started

        event_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        # Some Codex CLI builds can emit a complete, schema-valid last message
        # and still exit non-zero after stdin/terminal cleanup. The structured
        # artifact is authoritative; retry only when it is absent or invalid.
        if response_path.is_file():
            try:
                output = validator(json.loads(response_path.read_text(encoding="utf-8")))
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = f"Invalid structured output: {exc}"
            else:
                stage_result = {
                    **cache_identity,
                    "status": "complete",
                    "attempt": attempt,
                    "duration_seconds": round(duration, 3),
                    "duration_kind": "wall_call",
                    "usage": _extract_usage(stdout),
                    "output": output,
                    "workspace_id": workspace_id,
                    "workspace_stage": stage,
                    "codex_returncode": process.returncode,
                }
                _atomic_json(cached_path, stage_result)
                return stage_result
        _atomic_json(
            raw_dir / f"attempt-{attempt}-error.json",
            {
                "attempt": attempt,
                "duration_seconds": round(duration, 3),
                "error": last_error,
                "returncode": process.returncode,
            },
        )
        if attempt < max_attempts:
            failure_text = (stdout + b"\n" + stderr).decode("utf-8", errors="replace").lower()
            if any(
                marker in failure_text
                for marker in ("at capacity", "service unavailable", "status 503")
            ):
                retry_delay = min(30 * attempt, 120) + random.uniform(0, 10)
            elif "rate limit" in failure_text or "too many requests" in failure_text:
                retry_delay = min(15 * (2 ** (attempt - 1)), 120) + random.uniform(0, 10)
            else:
                retry_delay = min(2**attempt, 8)
            await asyncio.sleep(retry_delay)

    return {"status": "failed", "error": last_error}


async def _run_semantic_gate(
    job: Job,
    *,
    prompt: str,
    schema_path: Path,
    output_dir: Path,
    codex_bin: str,
    timeout_seconds: int,
    max_attempts: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    return await _run_stage(
        job,
        stage="semantic_gate",
        input_path=job.composite_path,
        prompt=prompt,
        schema_path=schema_path,
        validator=validate_gate,
        output_dir=output_dir,
        codex_bin=codex_bin,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        semaphore=semaphore,
    )


async def _run_render_gate(
    job: Job,
    *,
    prompt: str,
    schema_path: Path,
    output_dir: Path,
    codex_bin: str,
    timeout_seconds: int,
    max_attempts: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    return await _run_stage(
        job,
        stage="render_gate",
        input_path=job.composite_path,
        prompt=prompt,
        schema_path=schema_path,
        validator=validate_gate,
        output_dir=output_dir,
        codex_bin=codex_bin,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        semaphore=semaphore,
    )


async def _run_job(
    job: Job,
    *,
    prompts: dict[str, str],
    schema_paths: dict[str, Path],
    output_dir: Path,
    codex_bin: str,
    timeout_seconds: int,
    max_attempts: int,
    semaphore: asyncio.Semaphore,
    reuse_gates_from: Path | None,
) -> str:
    result_path = _result_path(output_dir, job)
    if _is_complete(result_path):
        return "skipped"

    stages: dict[str, Any] = {}
    reused: dict[str, Any] | None = None
    if reuse_gates_from is not None:
        reused = _load_reused_gate_stages(reuse_gates_from, job)
        semantic = {
            **reused["semantic_gate"],
            "reused_from": reused["source_path"],
            "source_protocol_id": reused["source_protocol_id"],
        }
    else:
        semantic = await _run_semantic_gate(
            job,
            prompt=prompts["semantic_gate"],
            schema_path=schema_paths["gate"],
            output_dir=output_dir,
            codex_bin=codex_bin,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            semaphore=semaphore,
        )
    stages["semantic_gate"] = semantic
    if semantic["status"] != "complete":
        final_status = "failed"
    elif semantic["output"]["gate_keeper"]["triggered"]:
        stages["render_gate"] = {"status": "skipped", "reason": "semantic_gate_triggered"}
        stages["scoring"] = {"status": "skipped", "reason": "semantic_gate_triggered"}
        final_status = "semantic_gate"
    else:
        if reused is not None:
            render = {
                **reused["render_gate"],
                "reused_from": reused["source_path"],
                "source_protocol_id": reused["source_protocol_id"],
            }
        else:
            render = await _run_render_gate(
                job,
                prompt=prompts["render_gate"],
                schema_path=schema_paths["gate"],
                output_dir=output_dir,
                codex_bin=codex_bin,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
                semaphore=semaphore,
            )
        stages["render_gate"] = render
        if render["status"] != "complete":
            final_status = "failed"
        elif render["output"]["gate_keeper"]["triggered"]:
            stages["scoring"] = {"status": "skipped", "reason": "render_gate_triggered"}
            final_status = "render_gate"
        else:
            scoring = await _run_stage(
                job,
                stage="scoring",
                input_path=job.composite_path,
                prompt=prompts["scoring"],
                schema_path=schema_paths["score"],
                validator=validate_score,
                output_dir=output_dir,
                codex_bin=codex_bin,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
                semaphore=semaphore,
            )
            stages["scoring"] = scoring
            if scoring["status"] != "complete":
                final_status = "failed"
            else:
                final_status = "scored"
                # The Judge only enumerates findings. Numeric dimensions and
                # the final score are computed later by issue_ranker and
                # consensus_ranker; no per-round score is stored.

    complete = final_status != "failed"
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "complete" if complete else "failed",
        "final_status": final_status,
        "task_id": job.task_id,
        "candidate_id": job.candidate.candidate_id,
        "judge_model": job.judge.model,
        "judge_effort": job.judge.effort,
        "composite_id": job.composite_path.stem,
        "composite_sha256": hashlib.sha256(job.composite_path.read_bytes()).hexdigest(),
        "stages": stages,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if final_status == "failed":
        payload["error"] = next(
            (stage["error"] for stage in stages.values() if stage.get("status") == "failed"),
            "Unknown stage failure",
        )
    _atomic_json(result_path, payload)
    return "complete" if complete else "failed"


def _parse_assignment(value: str, kind: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{kind} must use ID=VALUE syntax")
    identifier, assigned = value.split("=", 1)
    if not identifier or not assigned:
        raise argparse.ArgumentTypeError(f"{kind} must use non-empty ID=VALUE syntax")
    return identifier, assigned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--semantic-gate-prompt", type=Path, required=True)
    parser.add_argument("--render-gate-prompt", type=Path, required=True)
    parser.add_argument("--score-prompt", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, metavar="ID=SUMMARY_JSON")
    parser.add_argument("--judge", action="append", required=True, metavar="ID=MODEL[:EFFORT]")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codex-bin", default=shutil.which("codex") or "codex")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--reuse-gates-from",
        type=Path,
        help="Reuse validated semantic/render gate decisions from another final-protocol run",
    )
    return parser


async def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reuse_gates_from = args.reuse_gates_from.resolve() if args.reuse_gates_from else None
    schema_paths = {
        "gate": output_dir / "gate-output-schema.json",
        "score": output_dir / "score-output-schema.json",
    }
    _atomic_json(schema_paths["gate"], GATE_SCHEMA)
    _atomic_json(schema_paths["score"], SCORE_SCHEMA)
    prompts = {
        "semantic_gate": args.semantic_gate_prompt.read_text(encoding="utf-8").strip(),
        "render_gate": args.render_gate_prompt.read_text(encoding="utf-8").strip(),
        "scoring": args.score_prompt.read_text(encoding="utf-8").strip(),
    }
    task_ids = read_task_ids(args.tasks)
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    candidates = [
        Candidate(identifier, Path(value).resolve())
        for identifier, value in (_parse_assignment(item, "candidate") for item in args.candidate)
    ]
    judges: list[Judge] = []
    for identifier, value in (_parse_assignment(item, "judge") for item in args.judge):
        model, separator, effort = value.partition(":")
        judges.append(Judge(identifier, model, effort if separator else "medium"))

    candidate_maps = {
        candidate.candidate_id: load_candidate_screenshots(candidate.summary_path)
        for candidate in candidates
    }
    composites_dir = output_dir / "composites"
    jobs: list[Job] = []
    missing: list[dict[str, str]] = []
    artifact_zero_count = 0
    for task_id in task_ids:
        reference = _reference_path(args.dataset_root, task_id)
        for candidate in candidates:
            candidate_path = candidate_maps[candidate.candidate_id].get(task_id)
            if not reference.is_file() or candidate_path is None:
                missing.append(
                    {
                        "task_id": task_id,
                        "candidate_id": candidate.candidate_id,
                        "reason": (
                            "missing_reference" if not reference.is_file() else "missing_candidate"
                        ),
                    }
                )
                if reference.is_file() and candidate_path is None:
                    for judge in judges:
                        synthetic_job = Job(
                            task_id,
                            candidate,
                            judge,
                            reference,
                            reference,
                            Path(),
                        )
                        synthetic_path = _result_path(output_dir, synthetic_job)
                        if not _is_complete(synthetic_path):
                            _atomic_json(
                                synthetic_path,
                                {
                                    "schema": RESULT_SCHEMA,
                                    "protocol_id": PROTOCOL_ID,
                                    "status": "complete",
                                    "final_status": "artifact_gate",
                                    "task_id": task_id,
                                    "candidate_id": candidate.candidate_id,
                                    "judge_model": judge.model,
                                    "judge_effort": judge.effort,
                                    "artifact_gate": True,
                                    "stages": {
                                        "semantic_gate": {
                                            "status": "skipped",
                                            "reason": "artifact_gate",
                                        },
                                        "render_gate": {
                                            "status": "skipped",
                                            "reason": "artifact_gate",
                                        },
                                        "scoring": {
                                            "status": "skipped",
                                            "reason": "artifact_gate",
                                        },
                                    },
                                    "completed_at": datetime.now(timezone.utc).isoformat(),
                                },
                            )
                        artifact_zero_count += 1
                continue

            opaque = hashlib.sha256(
                f"{task_id}\0{candidate.candidate_id}\0blind-input".encode()
            ).hexdigest()[:20]
            composite_path = composites_dir / f"{opaque}.png"
            metadata_path = composites_dir / f"{opaque}.json"
            if not composite_path.is_file():
                metadata = make_blind_composite(
                    reference,
                    candidate_path,
                    composite_path,
                )
                _atomic_json(metadata_path, metadata)
            for judge in judges:
                jobs.append(
                    Job(
                        task_id,
                        candidate,
                        judge,
                        reference,
                        candidate_path,
                        composite_path,
                    )
                )

    _atomic_json(output_dir / "missing.json", missing)
    reused_gate_counts = {
        "semantic_gate": 0,
        "render_gate": 0,
        "scoring": 0,
    }
    if reuse_gates_from is not None:
        for job in jobs:
            reused = _load_reused_gate_stages(reuse_gates_from, job)
            semantic_gate = reused["semantic_gate"]["output"]["gate_keeper"]
            if semantic_gate["triggered"]:
                reused_gate_counts["semantic_gate"] += 1
                continue
            render_gate = reused["render_gate"]["output"]["gate_keeper"]
            if render_gate["triggered"]:
                reused_gate_counts["render_gate"] += 1
                continue
            reused_gate_counts["scoring"] += 1
    _atomic_json(
        output_dir / "run-manifest.json",
        {
            "schema": "pptbench-judge-manifest",
            "protocol_id": PROTOCOL_ID,
            "task_count": len(task_ids),
            "candidate_count": len(candidates),
            "judge_count": len(judges),
            "expected_result_count": len(task_ids) * len(candidates) * len(judges),
            "codex_job_count": len(jobs),
            "maximum_codex_call_count": (
                reused_gate_counts["scoring"] if reuse_gates_from is not None else len(jobs) * 3
            ),
            "artifact_zero_count": artifact_zero_count,
            "missing": missing,
            "normalization": "crop-background-contain",
            "ignored_canvas_properties": [
                "outer_whitespace",
                "global_centering_within_source_canvas",
                "absolute_content_scale_within_source_canvas",
            ],
            "blind_input": True,
            "tool_workspace": "isolated_workspace_write_with_agent_directed_analysis",
            "semantic_views": "single_full_composite_with_agent_directed_crops",
            "render_views": "single_full_composite_with_agent_directed_crops",
            "image_inspection": {
                "fixed_harness_tiles": False,
                "agent_may_write_crop_scripts": True,
                "agent_may_write_measurement_scripts": True,
                "agent_uses_numeric_output_for_unviewable_crops": True,
                "attached_images": [
                    "comparison.png",
                    "reference.png",
                    "candidate.png",
                ],
                "comparison_metadata": "comparison-metadata.json",
            },
            "gate_reuse": (
                {
                    "enabled": True,
                    "source_run": str(reuse_gates_from),
                    "validation": "identity_and_composite_sha256",
                    "outcomes": reused_gate_counts,
                }
                if reuse_gates_from is not None
                else {"enabled": False}
            ),
            "scoring": {
                "method": "deterministic_union_max_ratio",
                "dimension_maximum": {
                    "layout_and_composition": 30,
                    "text_and_typography": 40,
                    "local_graphics_and_nodes": 30,
                },
                "finding_merge_key": ["dimension", "category", "issue_type"],
                "affected_ratio": "maximum_across_non_gated_rounds",
                "affected_count": "diagnostic_only",
            },
        },
    )
    if args.prepare_only:
        print(json.dumps({"prepared_jobs": len(jobs), "missing": len(missing)}))
        return 0

    random.Random(args.seed).shuffle(jobs)
    semaphore = asyncio.Semaphore(args.concurrency)
    counts = {"complete": 0, "skipped": 0, "failed": 0}
    lock = asyncio.Lock()

    async def execute(job: Job) -> None:
        status = await _run_job(
            job,
            prompts=prompts,
            schema_paths=schema_paths,
            output_dir=output_dir,
            codex_bin=args.codex_bin,
            timeout_seconds=args.timeout_seconds,
            max_attempts=args.max_attempts,
            semaphore=semaphore,
            reuse_gates_from=reuse_gates_from,
        )
        async with lock:
            counts[status] += 1
            finished = sum(counts.values())
            print(
                json.dumps(
                    {
                        "finished": finished,
                        "total": len(jobs),
                        "status": status,
                        "task_id": job.task_id,
                        "candidate_id": job.candidate.candidate_id,
                        "judge_id": job.judge.judge_id,
                        "counts": counts,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    await asyncio.gather(*(execute(job) for job in jobs))
    _atomic_json(output_dir / "completion-summary.json", counts)
    return 1 if counts["failed"] else 0


def main() -> int:
    args = build_parser().parse_args()
    if args.concurrency < 1 or args.max_attempts < 1:
        raise SystemExit("concurrency and max-attempts must be positive")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
