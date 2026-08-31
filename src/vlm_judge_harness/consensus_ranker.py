"""Three-round issue-union ranker using maximum affected ratios only."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from .issue_ranker import DIMENSIONS, RATIO_STAIRS, _stair_multiplier
except ImportError:  # Allow a copied module to run against an installed source tree.
    from vlm_judge_harness.issue_ranker import (
        DIMENSIONS,
        RATIO_STAIRS,
        _stair_multiplier,
    )


CONSENSUS_SCHEMA = "pptbench-case-result"


def _round_id(payload: dict[str, Any], index: int) -> str:
    value = payload.get("judge_round")
    return str(value) if value else f"round-{index + 1}"


def _is_gated(payload: dict[str, Any]) -> bool:
    gate = payload.get("gate_keeper", {})
    return bool(gate.get("triggered")) or str(payload.get("source_final_status", "")) in {
        "artifact_gate",
        "semantic_gate",
        "render_gate",
    }


def _ratio_multiplier(ratio: float) -> float:
    return _stair_multiplier(ratio, RATIO_STAIRS)


def merge_ranked_results(
    results: list[dict[str, Any]], *, expected_rounds: int = 3
) -> dict[str, Any]:
    """Merge repeated ranked results for one task/candidate and score the union.

    Issues are canonicalized by dimension, category, and issue type. Repeated
    observations take the maximum affected ratio. Affected counts remain in
    provenance but never influence the penalty.
    """
    if len(results) != expected_rounds:
        raise ValueError(f"Expected {expected_rounds} rounds, found {len(results)}")
    task_ids = {str(item.get("task_id")) for item in results}
    candidates = {str(item.get("candidate_id")) for item in results}
    if len(task_ids) != 1 or len(candidates) != 1:
        raise ValueError("Consensus inputs must share task_id and candidate_id")

    gate_rounds: list[str] = []
    gate_reasons: list[dict[str, str]] = []
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    round_ids: list[str] = []

    for index, payload in enumerate(results):
        round_id = _round_id(payload, index)
        round_ids.append(round_id)
        if _is_gated(payload):
            gate_rounds.append(round_id)
            gate_reasons.append(
                {
                    "round": round_id,
                    "reason": str(payload.get("gate_keeper", {}).get("reason", "")),
                }
            )
            continue
        dimensions = payload.get("ranking", {}).get("dimensions", {})
        for dimension, dimension_payload in dimensions.items():
            if dimension not in DIMENSIONS:
                continue
            for raw_case in dimension_payload.get("cases", []):
                category = str(raw_case["category"])
                issue_type = str(raw_case["issue_type"])
                key = (dimension, category, issue_type)
                ratio = float(raw_case.get("affected_ratio", 0.0))
                if not 0.0 <= ratio <= 1.0:
                    raise ValueError(f"Invalid affected ratio {ratio} for {key}")
                severity = str(raw_case["severity"])
                base_points = float(raw_case["base_points"])
                item = merged.setdefault(
                    key,
                    {
                        "dimension": dimension,
                        "category": category,
                        "issue_type": issue_type,
                        "severity": severity,
                        "base_points": base_points,
                        "affected_ratio": ratio,
                        "round_support": set(),
                        "observations": [],
                    },
                )
                if base_points > float(item["base_points"]):
                    item["base_points"] = base_points
                    item["severity"] = severity
                item["affected_ratio"] = max(float(item["affected_ratio"]), ratio)
                item["round_support"].add(round_id)
                observation = {
                    "round": round_id,
                    "affected_ratio": ratio,
                    "affected_count": int(raw_case.get("affected_count", 0)),
                    "location": str(raw_case.get("location", "")),
                    "candidate": str(raw_case.get("candidate", "")),
                    "reference": str(raw_case.get("reference", "")),
                }
                if observation not in item["observations"]:
                    item["observations"].append(observation)

    majority_gate = len(gate_rounds) >= expected_rounds // 2 + 1
    if majority_gate:
        dimensions = {
            dimension: {
                "maximum": float(config["maximum"]),
                "raw_penalty": float(config["maximum"]),
                "applied_penalty": float(config["maximum"]),
                "score": 0.0,
                "issue_count": 0,
                "issues": [],
            }
            for dimension, config in DIMENSIONS.items()
        }
        total_score = 0.0
    else:
        by_dimension: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in merged.values():
            ratio = float(item["affected_ratio"])
            multiplier = _ratio_multiplier(ratio)
            points_lost = round(float(item["base_points"]) * multiplier, 2)
            round_support = sorted(item["round_support"])
            by_dimension[str(item["dimension"])].append(
                {
                    "category": item["category"],
                    "issue_type": item["issue_type"],
                    "severity": item["severity"],
                    "base_points": item["base_points"],
                    "affected_ratio": ratio,
                    "ratio_multiplier": multiplier,
                    "points_lost": points_lost,
                    "round_support": len(round_support),
                    "round_ids": round_support,
                    "observations": item["observations"],
                }
            )

        dimensions = {}
        total_score = 0.0
        for dimension, config in DIMENSIONS.items():
            issues = sorted(
                by_dimension.get(dimension, []),
                key=lambda item: (
                    -float(item["points_lost"]),
                    str(item["category"]),
                    str(item["issue_type"]),
                ),
            )
            raw_penalty = round(sum(float(item["points_lost"]) for item in issues), 2)
            maximum = float(config["maximum"])
            applied_penalty = min(maximum, raw_penalty)
            score = round(maximum - applied_penalty, 2)
            dimensions[dimension] = {
                "maximum": maximum,
                "raw_penalty": raw_penalty,
                "applied_penalty": applied_penalty,
                "score": score,
                "issue_count": len(issues),
                "issues": issues,
            }
            total_score += score
        total_score = round(total_score, 2)

    issue_count = sum(int(value["issue_count"]) for value in dimensions.values())
    single_round_issue_count = sum(
        1
        for value in dimensions.values()
        for issue in value["issues"]
        if int(issue["round_support"]) == 1
    )
    return {
        "schema": CONSENSUS_SCHEMA,
        "task_id": next(iter(task_ids)),
        "candidate_id": next(iter(candidates)),
        "round_ids": sorted(round_ids),
        "round_count": len(results),
        "gate_count": len(gate_rounds),
        "majority_gate": majority_gate,
        "gate_rounds": sorted(gate_rounds),
        "gate_reasons": gate_reasons,
        "dimensions": dimensions,
        "issue_count": issue_count,
        "single_round_issue_count": single_round_issue_count,
        "total_score": total_score,
    }


def _result_paths(source: Path) -> Iterable[Path]:
    yield from sorted(source.glob("results/*/*/task_*.json"))


def load_consensus(source: Path, expected_rounds: int = 3) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in _result_paths(source):
        payload = json.loads(path.read_text(encoding="utf-8"))
        grouped[(str(payload["task_id"]), str(payload["candidate_id"]))].append(payload)
    return [
        merge_ranked_results(items, expected_rounds=expected_rounds)
        for _, items in sorted(grouped.items())
    ]


def aggregate_models(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["candidate_id"])].append(case)
    rows: list[dict[str, Any]] = []
    for candidate, items in sorted(grouped.items()):
        count = len(items)
        issues = sum(int(item["issue_count"]) for item in items)
        single = sum(int(item["single_round_issue_count"]) for item in items)
        row: dict[str, Any] = {
            "candidate_id": candidate,
            "task_count": count,
            "majority_gate_count": sum(bool(item["majority_gate"]) for item in items),
            "majority_gate_rate": sum(bool(item["majority_gate"]) for item in items) / count,
            "mean_layout_score": sum(
                float(item["dimensions"]["layout_and_composition"]["score"])
                for item in items
            )
            / count,
            "mean_text_score": sum(
                float(item["dimensions"]["text_and_typography"]["score"])
                for item in items
            )
            / count,
            "mean_local_graphics_score": sum(
                float(item["dimensions"]["local_graphics_and_nodes"]["score"])
                for item in items
            )
            / count,
            "mean_final_score": sum(float(item["total_score"]) for item in items) / count,
            "mean_issue_count": issues / count,
            "single_round_issue_rate": single / issues if issues else 0.0,
        }
        rows.append(row)
    rows.sort(key=lambda item: (-float(item["mean_final_score"]), str(item["candidate_id"])))
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    return rows


def aggregate_issues(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str, str], dict[str, Any]] = {}
    for case in cases:
        if case["majority_gate"]:
            continue
        for dimension, dimension_payload in case["dimensions"].items():
            for issue in dimension_payload["issues"]:
                key = (dimension, str(issue["category"]), str(issue["issue_type"]))
                row = totals.setdefault(
                    key,
                    {
                        "dimension": dimension,
                        "category": issue["category"],
                        "issue_type": issue["issue_type"],
                        "severity": issue["severity"],
                        "case_count": 0,
                        "ratio_sum": 0.0,
                        "penalty_sum": 0.0,
                        "single_round_count": 0,
                    },
                )
                row["case_count"] += 1
                row["ratio_sum"] += float(issue["affected_ratio"])
                row["penalty_sum"] += float(issue["points_lost"])
                row["single_round_count"] += int(issue["round_support"] == 1)
    rows = []
    for row in totals.values():
        count = int(row["case_count"])
        rows.append(
            {
                "dimension": row["dimension"],
                "category": row["category"],
                "issue_type": row["issue_type"],
                "severity": row["severity"],
                "case_count": count,
                "mean_max_ratio": float(row["ratio_sum"]) / count,
                "total_penalty": float(row["penalty_sum"]),
                "single_round_rate": int(row["single_round_count"]) / count,
            }
        )
    rows.sort(key=lambda item: (-float(item["total_penalty"]), str(item["issue_type"])))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_outputs(
    output: Path,
    cases: list[dict[str, Any]],
    models: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "case-results.jsonl").open("w", encoding="utf-8") as handle:
        for item in cases:
            public = dict(item)
            for key in ("round_ids", "gate_rounds", "judge_model", "judge_effort", "aggregation"):
                public.pop(key, None)
            public["schema"] = CONSENSUS_SCHEMA
            public["gate_reasons"] = [
                {"reason": str(reason.get("reason", ""))}
                for reason in public.get("gate_reasons", [])
            ]
            for dimension in public.get("dimensions", {}).values():
                for issue in dimension.get("issues", []):
                    issue.pop("round_ids", None)
                    for observation in issue.get("observations", []):
                        observation.pop("round", None)
            handle.write(json.dumps(public, ensure_ascii=False) + "\n")
    case_rows = []
    for item in cases:
        case_rows.append(
            {
                "task_id": item["task_id"],
                "candidate_id": item["candidate_id"],
                "majority_gate": item["majority_gate"],
                "gate_count": item["gate_count"],
                "layout_score": item["dimensions"]["layout_and_composition"]["score"],
                "text_score": item["dimensions"]["text_and_typography"]["score"],
                "local_graphics_score": item["dimensions"]["local_graphics_and_nodes"]["score"],
                "total_score": item["total_score"],
                "issue_count": item["issue_count"],
                "single_round_issue_count": item["single_round_issue_count"],
            }
        )
    _write_csv(
        output / "scores.csv",
        case_rows,
        list(case_rows[0]),
    )
    _write_csv(
        output / "leaderboard.csv",
        models,
        [
            "rank",
            "candidate_id",
            "task_count",
            "majority_gate_count",
            "majority_gate_rate",
            "mean_layout_score",
            "mean_text_score",
            "mean_local_graphics_score",
            "mean_final_score",
            "mean_issue_count",
            "single_round_issue_rate",
        ],
    )
    _write_csv(output / "issue-summary.csv", issues, list(issues[0]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rounds", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = load_consensus(args.ranked_dir, expected_rounds=args.expected_rounds)
    if not cases:
        raise SystemExit("No ranked result JSON files found")
    models = aggregate_models(cases)
    issues = aggregate_issues(cases)
    _write_outputs(args.output_dir, cases, models, issues)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "models": len(models),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
