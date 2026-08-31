#!/usr/bin/env python3
"""Bootstrap task-count requirements for a matrix of per-task model scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_score_matrix(score_root: Path, score_column: str) -> tuple[list[str], list[str], np.ndarray]:
    by_model: dict[str, dict[str, float]] = {}
    for path in sorted(score_root.glob("*/batch-summary.csv")):
        rows: dict[str, float] = {}
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows[row["task_id"]] = float(row[score_column])
        by_model[path.parent.name] = rows
    if len(by_model) < 2:
        raise ValueError(f"need at least two model score files under {score_root}")
    common = set.intersection(*(set(rows) for rows in by_model.values()))
    tasks = sorted(common)
    models = sorted(by_model)
    matrix = np.asarray([[by_model[model][task] for task in tasks] for model in models])
    return models, tasks, matrix


def load_score_csv(
    path: Path,
    score_column: str,
    *,
    model_column: str = "candidate_id",
    task_column: str = "task_id",
) -> tuple[list[str], list[str], np.ndarray]:
    """Load a complete task-by-model matrix from a long-form score CSV."""

    by_model: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row[model_column]
            task = row[task_column]
            model_scores = by_model.setdefault(model, {})
            if task in model_scores:
                raise ValueError(f"duplicate score for {model} / {task}")
            model_scores[task] = float(row[score_column])
    if len(by_model) < 2:
        raise ValueError(f"need at least two models in {path}")
    common = set.intersection(*(set(rows) for rows in by_model.values()))
    if any(set(rows) != common for rows in by_model.values()):
        raise ValueError("long-form score CSV is not a complete task-by-model matrix")
    tasks = sorted(common)
    models = sorted(by_model)
    matrix = np.asarray([[by_model[model][task] for task in tasks] for model in models])
    return models, tasks, matrix


def kendall_tau(order_a: np.ndarray, order_b: np.ndarray) -> float:
    positions_a = np.empty(len(order_a), dtype=int)
    positions_b = np.empty(len(order_b), dtype=int)
    positions_a[order_a] = np.arange(len(order_a))
    positions_b[order_b] = np.arange(len(order_b))
    concordant = 0
    discordant = 0
    for left in range(len(order_a)):
        for right in range(left + 1, len(order_a)):
            same = (positions_a[left] - positions_a[right]) * (
                positions_b[left] - positions_b[right]
            )
            concordant += same > 0
            discordant += same < 0
    return (concordant - discordant) / max(1, concordant + discordant)


def analyze(
    models: list[str],
    tasks: list[str],
    matrix: np.ndarray,
    *,
    sample_sizes: list[int],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    full_means = matrix.mean(axis=1)
    full_order = np.argsort(-full_means, kind="stable")
    top1 = full_order[0]
    top2 = set(full_order[:2].tolist())
    top3 = set(full_order[:3].tolist())
    top5 = set(full_order[:5].tolist())
    result: dict[str, Any] = {
        "schema": "pptbench-bootstrap",
        "seed": seed,
        "iterations": iterations,
        "task_count": len(tasks),
        "model_count": len(models),
        "full_ranking": [
            {"rank": rank, "model": models[index], "mean_score": float(full_means[index])}
            for rank, index in enumerate(full_order, start=1)
        ],
        "sample_sizes": [],
    }
    for size in sample_sizes:
        sampled_means = np.empty((iterations, len(models)), dtype=float)
        taus = np.empty(iterations, dtype=float)
        top1_exact = 0
        top2_exact = 0
        top3_exact = 0
        top5_exact = 0
        top3_membership = np.zeros(len(models), dtype=int)
        for iteration in range(iterations):
            indices = rng.integers(0, len(tasks), size=size)
            means = matrix[:, indices].mean(axis=1)
            sampled_means[iteration] = means
            order = np.argsort(-means, kind="stable")
            taus[iteration] = kendall_tau(full_order, order)
            top1_exact += order[0] == top1
            top2_exact += set(order[:2].tolist()) == top2
            observed_top3 = set(order[:3].tolist())
            top3_exact += observed_top3 == top3
            top5_exact += set(order[:5].tolist()) == top5
            for index in observed_top3:
                top3_membership[index] += 1
        intervals = []
        for index, model in enumerate(models):
            low, high = np.quantile(sampled_means[:, index], [0.025, 0.975])
            intervals.append(
                {
                    "model": model,
                    "mean": float(full_means[index]),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "ci_width": float(high - low),
                    "top3_probability": float(top3_membership[index] / iterations),
                }
            )
        adjacent = []
        for upper, lower in zip(full_order[:-1], full_order[1:]):
            differences = sampled_means[:, upper] - sampled_means[:, lower]
            low, high = np.quantile(differences, [0.025, 0.975])
            adjacent.append(
                {
                    "upper_model": models[upper],
                    "lower_model": models[lower],
                    "full_gap": float(full_means[upper] - full_means[lower]),
                    "difference_ci_low": float(low),
                    "difference_ci_high": float(high),
                    "significant_95pct": bool(low > 0 or high < 0),
                    "rank_flip_probability": float(np.mean(differences < 0)),
                }
            )
        result["sample_sizes"].append(
            {
                "sample_size": size,
                "kendall_tau_mean": float(taus.mean()),
                "kendall_tau_p05": float(np.quantile(taus, 0.05)),
                "kendall_tau_p95": float(np.quantile(taus, 0.95)),
                "top1_exact_probability": float(top1_exact / iterations),
                "top2_exact_set_probability": float(top2_exact / iterations),
                "top3_exact_set_probability": float(top3_exact / iterations),
                "top5_exact_set_probability": float(top5_exact / iterations),
                "mean_model_ci_width": float(np.mean([item["ci_width"] for item in intervals])),
                "max_model_ci_width": float(np.max([item["ci_width"] for item in intervals])),
                "model_intervals": intervals,
                "adjacent_pairs": adjacent,
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--score-root", type=Path)
    source.add_argument("--scores-csv", type=Path)
    parser.add_argument("--score-column", default="total_score")
    parser.add_argument("--sample-sizes", default="50,100,200,300,400,500")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--source-task-count", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.scores_csv:
        models, tasks, matrix = load_score_csv(args.scores_csv, args.score_column)
    else:
        models, tasks, matrix = load_score_matrix(args.score_root, args.score_column)
    if args.source_task_count is not None:
        if not 2 <= args.source_task_count <= len(tasks):
            raise SystemExit("--source-task-count must be between 2 and the available task count")
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(len(tasks), size=args.source_task_count, replace=False))
        tasks = [tasks[index] for index in selected]
        matrix = matrix[:, selected]
    payload = analyze(
        models,
        tasks,
        matrix,
        sample_sizes=[int(value) for value in args.sample_sizes.split(",")],
        iterations=args.iterations,
        seed=args.seed,
    )
    payload["available_task_count"] = len(tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "bootstrap.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"tasks": len(tasks), "models": len(models)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
