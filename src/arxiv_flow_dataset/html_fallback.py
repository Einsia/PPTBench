from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from PIL import Image

from .images import difference_hash, file_sha256
from .models import Paper
from .structured import (
    _write_gallery,
    discover_structured_figures,
    structured_score,
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def build_arxiv_html_fallbacks(
    source_dataset: Path,
    output_dir: Path,
    node: Path,
    node_modules: Path,
    *,
    max_per_paper: int = 4,
    min_similarity: float = 0.5,
    dedupe_hamming_distance: int = 4,
    refresh_existing: bool = False,
) -> dict:
    source_dataset = source_dataset.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    papers = [
        Paper(**row)
        for row in _read_jsonl(source_dataset / "work" / "papers.jsonl")
    ]
    paper_by_id = {paper.arxiv_id: paper for paper in papers}
    rows_by_id = {
        row["structured_id"]: row
        for row in [
            *_read_jsonl(output_dir / "manifest.jsonl"),
            *_read_jsonl(output_dir / "manifest.partial.jsonl"),
        ]
        if (output_dir / row.get("render_path", "")).is_file()
    }
    rows = list(rows_by_id.values())
    errors_by_id = {
        row["structured_id"]: row
        for row in _read_jsonl(output_dir / "errors.jsonl")
        if row.get("structured_id")
    }
    refresh_rows: dict[str, dict] = {}
    if refresh_existing:
        for row in list(rows_by_id.values()):
            if row.get("render_method") != "arxiv-html-svg-fallback":
                continue
            key = row["structured_id"]
            refresh_rows[key] = row
            rows_by_id.pop(key, None)
            errors_by_id[key] = {
                "structured_id": key,
                "paper_id": row["paper"]["arxiv_id"],
                "source_file": row.get("source_origin", ""),
                "error": "refreshing arXiv HTML SVG fallback",
            }
        rows = list(rows_by_id.values())
    paper_counts: dict[str, int] = {}
    for row in rows:
        paper_id = row["paper"]["arxiv_id"]
        paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1
    source_hashes = {row["source_sha256"] for row in rows}
    render_hashes = [row["perceptual_hash"] for row in rows]

    tasks: list[dict] = []
    task_figures: dict[str, tuple[Paper, object, str]] = {}
    source_root = source_dataset / "work" / "sources"
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for paper_id in sorted({row.get("paper_id", "") for row in errors_by_id.values()}):
        paper = paper_by_id.get(paper_id)
        paper_dir = source_root / paper_id.replace("/", "_").replace(".", "-")
        if paper is None or not paper_dir.is_dir():
            continue
        selected_for_paper = paper_counts.get(paper_id, 0)
        for figure in discover_structured_figures(paper_dir, paper_id):
            if selected_for_paper >= max_per_paper:
                break
            if structured_score(figure)[0] < 0 or not figure.caption_text:
                continue
            digest = file_sha256(figure.source_file)[:10]
            key = (
                f"{paper_dir.name}_sf{figure.figure_index:03d}_g{figure.graphic_index:02d}_"
                f"{digest}"
            )
            if key not in errors_by_id or key in rows_by_id:
                continue
            source_digest = hashlib.sha256(figure.source_code.encode("utf-8")).hexdigest()
            if source_digest in source_hashes:
                errors_by_id.pop(key, None)
                continue
            sample_dir = samples_dir / key
            sample_dir.mkdir(parents=True, exist_ok=True)
            tasks.append(
                {
                    "key": key,
                    "arxiv_id": paper_id,
                    "caption": figure.caption_text,
                    "output_png": str((sample_dir / "render.png").resolve()),
                    "output_svg": str((sample_dir / "arxiv-html.svg").resolve()),
                }
            )
            task_figures[key] = (paper, figure, source_digest)
            source_hashes.add(source_digest)
            selected_for_paper += 1

    if not tasks:
        return {"attempted": 0, "recovered": 0, "remaining_errors": len(errors_by_id)}

    tasks_path = output_dir / "html-fallback-tasks.json"
    results_path = output_dir / "html-fallback-results.json"
    tasks_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "scripts" / "render_arxiv_html_fallback.mjs"
    environment = os.environ.copy()
    environment["RUNTIME_NODE_MODULES"] = str(node_modules.resolve())
    environment["NODE_PATH"] = str((node_modules / ".pnpm" / "node_modules").resolve())
    result = subprocess.run(
        [
            str(node.resolve()),
            str(script),
            "--tasks",
            str(tasks_path),
            "--results",
            str(results_path),
            "--min-similarity",
            str(min_similarity),
        ],
        cwd=script.parent.parent,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(180, len(tasks) * 100),
        check=False,
    )
    if result.returncode or not results_path.exists():
        raise RuntimeError((result.stdout + "\n" + result.stderr)[-6000:])

    fallback_results = {
        row["key"]: row for row in json.loads(results_path.read_text(encoding="utf-8"))
    }
    recovered = 0
    duplicates = 0
    for key, (paper, figure, source_digest) in task_figures.items():
        fallback = fallback_results.get(key, {"status": "missing-result"})
        sample_dir = samples_dir / key
        render_path = sample_dir / "render.png"
        svg_path = sample_dir / "arxiv-html.svg"
        if fallback.get("status") != "ok" or not render_path.is_file() or not svg_path.is_file():
            previous = refresh_rows.get(key)
            if previous is not None:
                rows.append(previous)
                rows_by_id[key] = previous
                errors_by_id.pop(key, None)
            elif sample_dir.is_dir() and output_dir in sample_dir.resolve().parents:
                shutil.rmtree(sample_dir, ignore_errors=True)
            continue
        render_hash = difference_hash(render_path)
        if any(
            _hamming(render_hash, previous) <= dedupe_hamming_distance
            for previous in render_hashes
        ):
            duplicates += 1
            errors_by_id.pop(key, None)
            shutil.rmtree(sample_dir, ignore_errors=True)
            continue
        source_path = sample_dir / "source.tex"
        source_path.write_text(
            figure.source_code,
            encoding="utf-8",
            newline="\n",
        )
        graph_path = sample_dir / "graph.json"
        graph_path.write_text(
            json.dumps(figure.graph, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (sample_dir / "compile.log").write_text(
            "TikZ standalone compilation failed; render recovered from arXiv HTML SVG\n"
            + json.dumps(fallback, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with Image.open(render_path) as image:
            width, height = image.size

        def relative(path: Path) -> str:
            return path.relative_to(output_dir).as_posix()

        score, reasons = structured_score(figure)
        row = {
            "structured_id": key,
            "candidate_id": key,
            "description": figure.caption_text,
            "paper": {field: getattr(paper, field) for field in Paper.__dataclass_fields__},
            "source_kind": figure.source_kind,
            "source_path": relative(source_path),
            "source_origin": str(figure.source_file.relative_to(source_dataset)),
            "graph_path": relative(graph_path),
            "image_path": relative(render_path),
            "render_path": relative(render_path),
            "html_svg_path": relative(svg_path),
            "render_method": "arxiv-html-svg-fallback",
            "render_source_url": fallback.get("url", ""),
            "render_caption_similarity": fallback.get("similarity", 0),
            "caption_latex": figure.caption_latex,
            "label": figure.label,
            "node_count": figure.graph["node_count"],
            "connector_count": figure.graph["connector_count"],
            "embedded_raster_count": 0,
            "width": width,
            "height": height,
            "score": score,
            "score_reasons": [*reasons, "arxiv-html-render-fallback"],
            "source_sha256": source_digest,
            "render_sha256": file_sha256(render_path),
            "perceptual_hash": render_hash,
        }
        rows.append(row)
        rows_by_id[key] = row
        render_hashes.append(render_hash)
        paper_counts[paper.arxiv_id] = paper_counts.get(paper.arxiv_id, 0) + 1
        errors_by_id.pop(key, None)
        recovered += 1

    rows.sort(key=lambda item: (item["score"], item["node_count"]), reverse=True)
    _write_jsonl(output_dir / "manifest.partial.jsonl", rows)
    _write_jsonl(output_dir / "manifest.jsonl", rows)
    _write_jsonl(output_dir / "errors.jsonl", list(errors_by_id.values()))
    _write_gallery(output_dir, rows)
    tasks_path.unlink(missing_ok=True)
    return {
        "attempted": len(tasks),
        "recovered": recovered,
        "duplicates": duplicates,
        "remaining_errors": len(errors_by_id),
        "rendered_samples": len(rows),
    }
