from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import math
from pathlib import Path
import shutil
import sys
from urllib.error import HTTPError

from .config import Config
from .http import CachedHttpClient, ResponseTooLargeError
from .io import read_jsonl
from .pipeline import _safe_id, discover_papers
from .source import UnsafeArchiveError, extract_source_archive, inspect_source_archive
from .structured import discover_structured_figures, structured_score


FILTER_VERSION = 5


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _remove_extracted_dir(path: Path, source_root: Path) -> None:
    resolved = path.resolve()
    root = source_root.resolve()
    if root not in resolved.parents:
        raise ValueError(f"refusing to remove source directory outside workspace: {resolved}")
    shutil.rmtree(resolved, ignore_errors=True)


def _scan_paper(
    paper_index: int,
    paper,
    client: CachedHttpClient,
    source_root: Path,
) -> dict:
    safe_id = _safe_id(paper.arxiv_id)
    paper_root = source_root / safe_id
    row: dict = {
        "arxiv_id": paper.arxiv_id,
        "paper_index": paper_index,
        "status": "error",
        "tikz_files": 0,
        "svg_files": 0,
        "structured_figures": 0,
        "qualifying_count": 0,
        "filter_version": FILTER_VERSION,
        "error": "",
    }
    try:
        source_version = paper.versioned_id or paper.arxiv_id
        source_url = f"https://arxiv.org/src/{source_version}"
        payload = client.get(source_url, f"source-{safe_id}", ".bin")
        markers = inspect_source_archive(payload)
        row.update(
            tikz_files=markers["tikz_files"],
            svg_files=markers["svg_files"],
        )
        if not markers["has_structured_source"]:
            row["status"] = "no-structured-marker"
        else:
            if not (paper_root / ".extracted").exists():
                if paper_root.exists():
                    _remove_extracted_dir(paper_root, source_root)
                extract_source_archive(payload, paper_root)
                (paper_root / ".extracted").write_text("ok\n", encoding="utf-8")
            figures = discover_structured_figures(paper_root, paper.arxiv_id)
            qualifying = [figure for figure in figures if structured_score(figure)[0] >= 0]
            row["structured_figures"] = len(figures)
            row["qualifying_count"] = len(qualifying)
            row["status"] = "qualified" if qualifying else "rejected-after-parse"
            if not qualifying:
                _remove_extracted_dir(paper_root, source_root)
    except ResponseTooLargeError as exc:
        row["status"] = "archive-too-large"
        row["error"] = str(exc)
        if paper_root.exists():
            _remove_extracted_dir(paper_root, source_root)
    except HTTPError as exc:
        row["status"] = "unavailable" if exc.code in {403, 404, 410} else "error"
        row["error"] = str(exc)
        if paper_root.exists():
            _remove_extracted_dir(paper_root, source_root)
    except (OSError, UnsafeArchiveError, ValueError) as exc:
        row["error"] = str(exc)
        if paper_root.exists():
            _remove_extracted_dir(paper_root, source_root)
    return row


def scan_structured_sources(
    config: Config,
    *,
    target_figures: int | None = None,
    max_per_paper: int | None = None,
    workers: int = 2,
) -> dict:
    root = config.dataset.output_dir
    root.mkdir(parents=True, exist_ok=True)
    papers = discover_papers(config)
    target = target_figures or config.dataset.target_figures
    desired_pool = math.ceil(target * config.dataset.oversample_factor)
    per_paper = max_per_paper or config.selection.max_figures_per_paper
    scan_path = root / "work" / "structured-scan.jsonl"
    summary_path = root / "work" / "structured-scan-summary.json"
    scans = {row["arxiv_id"]: row for row in read_jsonl(scan_path)}
    source_root = root / "work" / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    client = CachedHttpClient(
        root / "cache" / "archives",
        config.arxiv.user_agent,
        timeout=config.arxiv.request_timeout_seconds,
        min_interval=config.arxiv.download_delay_seconds,
        max_attempts=12,
        max_response_bytes=12_500_000,
    )

    for paper_id, row in list(scans.items()):
        if row.get("status") != "qualified":
            continue
        paper_root = source_root / _safe_id(paper_id)
        if not paper_root.is_dir():
            continue
        figures = discover_structured_figures(paper_root, paper_id)
        new_count = sum(structured_score(figure)[0] >= 0 for figure in figures)
        if new_count == int(row.get("qualifying_count", 0)):
            continue
        revised = dict(row)
        revised["qualifying_count"] = new_count
        revised["filter_version"] = FILTER_VERSION
        revised["status"] = "qualified" if new_count else "rejected-after-parse"
        scans[paper_id] = revised
        _append_jsonl(scan_path, revised)

    qualified = sum(min(int(row.get("qualifying_count", 0)), per_paper) for row in scans.values())
    tasks = [
        (paper_index, paper)
        for paper_index, paper in enumerate(papers, start=1)
        if paper.arxiv_id not in scans
        or scans[paper.arxiv_id].get("status") == "error"
        or (
            scans[paper.arxiv_id].get("status") == "rejected-after-parse"
            and scans[paper.arxiv_id].get("filter_version") != FILTER_VERSION
        )
    ]
    tasks.sort(
        key=lambda item: (
            0
            if scans.get(item[1].arxiv_id, {}).get("status") == "rejected-after-parse"
            and scans.get(item[1].arxiv_id, {}).get("svg_files", 0)
            else 1
            if scans.get(item[1].arxiv_id, {}).get("status") == "rejected-after-parse"
            else 2,
            item[0],
        )
    )
    task_iterator = iter(tasks)
    worker_count = max(1, min(workers, 16))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending: dict[Future, tuple[int, object]] = {}

        def submit_next() -> bool:
            try:
                paper_index, paper = next(task_iterator)
            except StopIteration:
                return False
            future = executor.submit(_scan_paper, paper_index, paper, client, source_root)
            pending[future] = (paper_index, paper)
            return True

        for _ in range(worker_count):
            submit_next()
        while pending and qualified < desired_pool:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                paper_index, paper = pending.pop(future)
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "arxiv_id": paper.arxiv_id,
                        "paper_index": paper_index,
                        "status": "error",
                        "tikz_files": 0,
                        "svg_files": 0,
                        "structured_figures": 0,
                        "qualifying_count": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                scans[paper.arxiv_id] = row
                _append_jsonl(scan_path, row)
                qualified += min(int(row.get("qualifying_count", 0)), per_paper)
                if row["status"] == "qualified" or paper_index % 25 == 0:
                    print(
                        f"[structured-scan] papers={len(scans)}/{len(papers)} "
                        f"pool={qualified}/{desired_pool} status={row['status']} "
                        f"tikz={row['tikz_files']} svg={row['svg_files']}",
                        file=sys.stderr,
                        flush=True,
                    )
                if qualified < desired_pool:
                    submit_next()
        for future in pending:
            future.cancel()

    summary = {
        "papers_discovered": len(papers),
        "papers_scanned": len(scans),
        "papers_with_structured_marker": sum(
            bool(row.get("tikz_files") or row.get("svg_files")) for row in scans.values()
        ),
        "papers_with_qualified_flow": sum(
            row.get("status") == "qualified" for row in scans.values()
        ),
        "qualifying_pool_capped_per_paper": qualified,
        "desired_pool": desired_pool,
        "target_figures": target,
        "tikz_marker_files": sum(int(row.get("tikz_files", 0)) for row in scans.values()),
        "svg_marker_files": sum(int(row.get("svg_files", 0)) for row in scans.values()),
        "errors": sum(row.get("status") == "error" for row in scans.values()),
        "archives_too_large": sum(
            row.get("status") == "archive-too-large" for row in scans.values()
        ),
        "complete": qualified >= desired_pool,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
