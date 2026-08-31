"""Deterministic 100-point ranker for structured VLM issue enumerations."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RANKER_ID = "pptbench-final-scoring"
RANKED_RESULT_SCHEMA = "pptbench-ranked-result"

DIMENSIONS = {
    "layout_and_composition": {
        "score_key": "layout_and_composition_score",
        "maximum": 30.0,
    },
    "text_and_typography": {
        "score_key": "text_and_typography_score",
        "maximum": 40.0,
    },
    "local_graphics_and_nodes": {
        "score_key": "local_graphics_and_nodes_score",
        "maximum": 30.0,
    },
}

CATEGORY_DIMENSIONS = {
    "panel_misalignment": "layout_and_composition",
    "node_layout_deviation": "layout_and_composition",
    "panel_style_error": "layout_and_composition",
    "spelling_and_omission": "text_and_typography",
    "text_overflow_or_edge_touch": "text_and_typography",
    "improper_typography": "text_and_typography",
    "node_shape_error": "local_graphics_and_nodes",
    "node_style_error": "local_graphics_and_nodes",
    "node_shadow_error": "local_graphics_and_nodes",
    "vector_and_icon_confusion": "local_graphics_and_nodes",
    "line_and_arrow_details": "local_graphics_and_nodes",
}

SEVERITY_POINTS = {
    "high": 3.0,
    "medium": 1.5,
    "low": 0.5,
    "diagnostic": 0.0,
}

# Defaults cover future subtypes without silently dropping them. Overrides make
# semantically important, explicitly cosmetic, and intentionally ignored cases
# stable across prompt revisions.
CATEGORY_DEFAULT_SEVERITY = {
    "panel_misalignment": "medium",
    "node_layout_deviation": "medium",
    "panel_style_error": "low",
    "spelling_and_omission": "high",
    "text_overflow_or_edge_touch": "high",
    "improper_typography": "low",
    "node_shape_error": "medium",
    "node_style_error": "low",
    "node_shadow_error": "diagnostic",
    "vector_and_icon_confusion": "medium",
    "line_and_arrow_details": "medium",
}

HIGH_ISSUE_TYPES = {
    # Panels and macro layout.
    "missing_panel",
    "extra_panel",
    "duplicate_panel",
    "substituted_panel",
    "panel_order",
    "panel_overlap",
    "panel_clipping",
    "conspicuous_blank_region",
    # Nodes and hierarchy.
    "missing_node",
    "extra_node",
    "duplicate_node",
    "substituted_node",
    "reading_order",
    "grouping",
    "nesting",
    "hierarchy",
    "node_overlap",
    "node_occlusion",
    # Text content and fit.
    "missing_text",
    "extra_text",
    "substituted_text",
    "duplicated_text",
    "reordered_text",
    "incorrect_number",
    "incorrect_identifier",
    "math_symbol",
    "greek_letter",
    "wrong_label_assignment",
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
    # Shape identity.
    "missing_shape_primitive",
    "extra_shape_primitive",
    # Connector semantics which survived the semantic gate.
    "missing_connector",
    "extra_connector",
    "wrong_source_attachment",
    "wrong_target_attachment",
    "wrong_port",
    "connector_direction",
    "bidirectionality",
}

MEDIUM_ISSUE_TYPES = {
    "panel_position",
    "panel_size",
    "panel_proportion",
    "panel_spacing",
    "panel_alignment",
    "separator_position",
    "node_position",
    "node_spacing",
    "node_alignment",
    "node_distribution",
    "grid_arrangement",
    "z_order",
    "local_scale",
    "edge_touch",
    "insufficient_text_padding",
    "font_family",
    "font_weight",
    "text_color",
    "text_contrast",
    "line_height",
    "baseline",
    "math_typesetting",
    "horizontal_alignment",
    "vertical_alignment",
    "text_anchoring",
    "unexpected_wrap",
    "mid_word_break",
    "isolated_character_line",
    "line_count",
    "line_order",
    "paragraph_width",
    "text_rotation",
    "text_orientation",
    "node_aspect_ratio",
    "shape_type",
    "node_size",
    "node_rotation",
    "node_orientation",
    "node_skew",
    "node_symmetry",
    "notch_geometry",
    "folded_corner",
    "cutout",
    "tab_geometry",
    "port_geometry",
    "handle_geometry",
    "tail_geometry",
    "pointer_geometry",
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
    "rasterized_vector",
    "grid_dimensions",
    "grid_count",
    "primitive_count",
    "primitive_order",
    "mini_diagram_topology",
    "chart_axes",
    "chart_ticks",
    "chart_legend",
    "chart_series",
    "raster_crop",
    "raster_stretch",
    "raster_replacement",
    "detached_endpoint",
    "endpoint_side",
    "arrowhead_endpoint",
    "intermediate_arrowhead",
    "arrowhead_count",
    "arrowhead_shape",
    "route_shape",
    "bezier_curvature",
    "bend_count",
    "bend_location",
    "bend_radius",
    "loop_shape",
    "bypass_shape",
    "junction_dot",
    "split_merge_marker",
    "crossing_bridge",
    "intersection_treatment",
    "connector_overlap_text",
    "connector_overlap_node",
    "connector_clipping",
    "border_confusion",
}

LOW_ISSUE_TYPES = {
    "capitalization",
    "punctuation",
    "abbreviation",
    "accent",
    "unit",
    "font_size",
    "font_style",
    "text_opacity",
    "letter_spacing",
    "word_spacing",
    "kerning",
    "internal_padding",
    "hyphenation",
    "node_corner_radius",
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
    "style_consistency",
    "panel_corner_radius",
    "cell_size",
    "cell_spacing",
    "cell_border",
    "cell_alignment",
    "primitive_position",
    "primitive_color",
    "primitive_fill",
    "primitive_pattern",
    "vector_blur",
    "vector_pixelation",
    "jagged_vector",
    "raster_blur",
    "arrowhead_size",
    "arrowhead_fill",
    "arrowhead_outline",
    "arrowhead_color",
    "arrowhead_angle",
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
}

DIAGNOSTIC_ISSUE_TYPES = {
    "global_scale",
    "global_centering",
    "outer_whitespace",
    "panel_shadow",
    "line_shadow",
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
}


@dataclass(frozen=True)
class Stair:
    upper: float | None
    multiplier: float


RATIO_STAIRS = (
    Stair(0.02, 1.0),
    Stair(0.10, 1.25),
    Stair(0.25, 1.5),
    Stair(0.50, 2.0),
    Stair(None, 2.5),
)


def _stair_multiplier(value: float, stairs: tuple[Stair, ...]) -> float:
    for stair in stairs:
        if stair.upper is None or value <= stair.upper:
            return stair.multiplier
    raise AssertionError("stair table must end with an open interval")


def severity_for(category: str, issue_type: str) -> str:
    if category not in CATEGORY_DEFAULT_SEVERITY:
        raise ValueError(f"Unknown issue category: {category}")
    if issue_type in DIAGNOSTIC_ISSUE_TYPES:
        return "diagnostic"
    if issue_type in HIGH_ISSUE_TYPES:
        return "high"
    if issue_type in MEDIUM_ISSUE_TYPES:
        return "medium"
    if issue_type in LOW_ISSUE_TYPES:
        return "low"
    return CATEGORY_DEFAULT_SEVERITY[category]


def score_enumeration(enumeration: dict[str, Any]) -> dict[str, Any]:
    deductions = enumeration.get("deductions")
    if not isinstance(deductions, dict):
        raise ValueError("Enumeration must contain a deductions object")

    dimensions: dict[str, dict[str, Any]] = {}
    total_score = 0.0
    total_penalty = 0.0
    for dimension, config in DIMENSIONS.items():
        categories = deductions.get(dimension)
        if not isinstance(categories, dict):
            raise ValueError(f"Missing enumeration dimension: {dimension}")
        expected_categories = {
            category
            for category, category_dimension in CATEGORY_DIMENSIONS.items()
            if category_dimension == dimension
        }
        if set(categories) != expected_categories:
            raise ValueError(
                f"Unexpected categories for {dimension}: "
                f"{sorted(set(categories) ^ expected_categories)}"
            )
        cases: list[dict[str, Any]] = []
        for category, observation in categories.items():
            if CATEGORY_DIMENSIONS.get(category) != dimension:
                raise ValueError(f"Unexpected category {category!r} in {dimension}")
            if not isinstance(observation, dict):
                raise ValueError(f"Invalid observation for {category}")
            affected_ratio = float(observation.get("ratio", 0.0))
            if not 0.0 <= affected_ratio <= 1.0:
                raise ValueError(f"Invalid ratio for {category}: {affected_ratio}")
            raw_cases = observation.get("cases")
            if not isinstance(raw_cases, list):
                raise ValueError(f"Invalid cases for {category}")
            for raw_case in raw_cases:
                if not isinstance(raw_case, dict):
                    raise ValueError(f"Invalid case for {category}")
                affected_count = raw_case["affected_count"]
                if type(affected_count) is not int or affected_count < 1:
                    raise ValueError("affected_count must be positive")
                issue_type = str(raw_case["issue_type"])
                severity = severity_for(category, issue_type)
                base_points = SEVERITY_POINTS[severity]
                ratio_multiplier = _stair_multiplier(affected_ratio, RATIO_STAIRS)
                points_lost = round(base_points * ratio_multiplier, 2)
                cases.append(
                    {
                        "category": category,
                        "issue_type": issue_type,
                        "severity": severity,
                        "base_points": base_points,
                        "affected_count": affected_count,
                        "affected_ratio": affected_ratio,
                        "ratio_multiplier": ratio_multiplier,
                        "points_lost": points_lost,
                        "location": str(raw_case.get("location", "")),
                        "candidate": str(raw_case.get("candidate", "")),
                        "reference": str(raw_case.get("reference", "")),
                    }
                )
        raw_penalty = round(sum(case["points_lost"] for case in cases), 2)
        maximum = float(config["maximum"])
        applied_penalty = min(maximum, raw_penalty)
        score = round(maximum - applied_penalty, 2)
        dimensions[dimension] = {
            "score_key": config["score_key"],
            "maximum": maximum,
            "raw_penalty": raw_penalty,
            "applied_penalty": applied_penalty,
            "score": score,
            "case_count": len(cases),
            "cases": cases,
        }
        total_score += score
        total_penalty += applied_penalty

    return {
        "ranker_id": RANKER_ID,
        "maximum_score": 100.0,
        "dimension_weights": {
            dimension: config["maximum"] / 100.0
            for dimension, config in DIMENSIONS.items()
        },
        "dimensions": dimensions,
        "total_penalty": round(total_penalty, 2),
        "total_score": round(total_score, 2),
    }


def rank_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "complete":
        raise ValueError("Only complete judge results can be ranked")
    final_status = str(result.get("final_status", ""))
    # New raw Judge exports contain findings only; gate metadata lives in the
    # corresponding stage output.  Keep a narrow fallback for older private
    # exports so historical rows can still be re-ranked during migration.
    gate: dict[str, Any] = {}
    legacy_score = result.get("score")
    if isinstance(legacy_score, dict) and isinstance(legacy_score.get("gate_keeper"), dict):
        gate = dict(legacy_score["gate_keeper"])
    stages = result.get("stages", {})
    if isinstance(stages, dict):
        stage_name = {
            "artifact_gate": None,
            "semantic_gate": "semantic_gate",
            "render_gate": "render_gate",
        }.get(final_status)
        if stage_name is not None:
            stage = stages.get(stage_name)
            stage_output = stage.get("output") if isinstance(stage, dict) else None
            stage_gate = stage_output.get("gate_keeper") if isinstance(stage_output, dict) else None
            if isinstance(stage_gate, dict):
                gate = dict(stage_gate)
        elif final_status == "artifact_gate":
            gate = {
                "triggered": True,
                "reason": str(result.get("artifact_gate_reason", "Artifact is missing or invalid.")),
            }
    gated = bool(gate.get("triggered")) or final_status in {
        "artifact_gate",
        "semantic_gate",
        "render_gate",
    }
    if gated:
        dimensions = {
            dimension: {
                "score_key": config["score_key"],
                "maximum": float(config["maximum"]),
                "raw_penalty": float(config["maximum"]),
                "applied_penalty": float(config["maximum"]),
                "score": 0.0,
                "case_count": 0,
                "cases": [],
            }
            for dimension, config in DIMENSIONS.items()
        }
        ranking = {
            "ranker_id": RANKER_ID,
            "maximum_score": 100.0,
            "dimension_weights": {
                dimension: config["maximum"] / 100.0
                for dimension, config in DIMENSIONS.items()
            },
            "dimensions": dimensions,
            "total_penalty": 100.0,
            "total_score": 0.0,
        }
    else:
        scoring = result.get("stages", {}).get("scoring", {})
        enumeration = scoring.get("output")
        if final_status != "scored" or scoring.get("status") != "complete":
            raise ValueError("A non-gated result must contain a complete scoring enumeration")
        if not isinstance(enumeration, dict):
            raise ValueError("Missing scoring enumeration")
        ranking = score_enumeration(enumeration)

    return {
        "schema": RANKED_RESULT_SCHEMA,
        "ranker_id": RANKER_ID,
        "task_id": result["task_id"],
        "candidate_id": result["candidate_id"],
        "judge_model": result["judge_model"],
        "judge_effort": result["judge_effort"],
        "source_protocol_id": result.get("protocol_id"),
        "source_final_status": final_status,
        "gate_keeper": {
            "triggered": gated,
            "reason": str(gate.get("reason", "")),
        },
        "ranking": ranking,
    }


def _result_paths(source: Path) -> Iterable[Path]:
    if source.is_file():
        yield source
        return
    yield from sorted((source / "results").glob("*/*/task_*.json"))


def _write_summary(path: Path, ranked: list[dict[str, Any]]) -> None:
    fields = [
        "task_id",
        "candidate_id",
        "judge_model",
        "judge_effort",
        "source_final_status",
        "gate_triggered",
        "layout_score",
        "text_score",
        "local_graphics_score",
        "total_score",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for payload in ranked:
            dimensions = payload["ranking"]["dimensions"]
            writer.writerow(
                {
                    "task_id": payload["task_id"],
                    "candidate_id": payload["candidate_id"],
                    "judge_model": payload["judge_model"],
                    "judge_effort": payload["judge_effort"],
                    "source_final_status": payload["source_final_status"],
                    "gate_triggered": payload["gate_keeper"]["triggered"],
                    "layout_score": dimensions["layout_and_composition"]["score"],
                    "text_score": dimensions["text_and_typography"]["score"],
                    "local_graphics_score": dimensions["local_graphics_and_nodes"]["score"],
                    "total_score": payload["ranking"]["total_score"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Judge run directory or one result JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ranked: list[dict[str, Any]] = []
    for source_path in _result_paths(args.source.resolve()):
        result = json.loads(source_path.read_text(encoding="utf-8"))
        payload = rank_result(result)
        ranked.append(payload)
        target = (
            args.output_dir
            / "results"
            / f"{payload['judge_model']}__{payload['judge_effort']}"
            / payload["candidate_id"]
            / f"{payload['task_id']}.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if not ranked:
        raise SystemExit(f"No judge result JSON files found under {args.source}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(args.output_dir / "scores-detail.csv", ranked)
    print(json.dumps({"ranked": len(ranked), "output_dir": str(args.output_dir.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
