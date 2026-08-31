from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
import sys
from typing import Iterable

from .arxiv_api import fetch_license, search_papers, with_submitted_date_filter
from .config import Config
from .http import CachedHttpClient
from .images import (
    UnsupportedImageError,
    difference_hash,
    file_sha256,
    hamming_distance,
    normalize_image,
)
from .io import candidate_from_dict, paper_to_dict, read_jsonl, write_jsonl
from .latex import parse_figures
from .models import Candidate, Paper
from .review import copy_release_image, read_review_status, write_review_bundle
from .scoring import passes_hard_filters, passes_text_filters, score_figure
from .source import UnsafeArchiveError, extract_source_archive


def _safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_").replace(".", "-")


def _interleave(groups: list[list[Paper]]) -> list[Paper]:
    seen: set[str] = set()
    result: list[Paper] = []
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group) and group[index].arxiv_id not in seen:
                seen.add(group[index].arxiv_id)
                result.append(group[index])
    return result


def discover_papers(config: Config, *, refresh: bool = False) -> list[Paper]:
    root = config.dataset.output_dir
    manifest = root / "work" / "papers.jsonl"
    existing = read_jsonl(manifest)
    if existing and not refresh:
        return [Paper(**row) for row in existing]

    client = CachedHttpClient(
        root / "cache" / "api",
        config.arxiv.user_agent,
        timeout=config.arxiv.request_timeout_seconds,
        min_interval=config.arxiv.api_delay_seconds,
    )
    groups = [
        search_papers(
            client,
            with_submitted_date_filter(
                query,
                config.arxiv.submitted_after,
                config.arxiv.submitted_before,
            ),
            config.arxiv.results_per_query,
            config.arxiv.page_size,
            query_number,
        )
        for query_number, query in enumerate(config.arxiv.queries, start=1)
    ]
    papers = _interleave(groups)
    write_jsonl(manifest, (paper_to_dict(paper) for paper in papers))
    return papers


def _license_allowed(url: str, allowed_urls: list[str]) -> bool:
    normalized = url.strip().lower().replace("https://", "http://").rstrip("/")
    allowed = {
        item.strip().lower().replace("https://", "http://").rstrip("/")
        for item in allowed_urls
    }
    return bool(normalized and normalized in allowed)


def _deduplicate(candidates: Iterable[Candidate], max_distance: int) -> list[Candidate]:
    kept: list[Candidate] = []
    sha_seen: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if candidate.sha256 in sha_seen:
            continue
        if any(
            hamming_distance(candidate.perceptual_hash, other.perceptual_hash) <= max_distance
            for other in kept
        ):
            continue
        sha_seen.add(candidate.sha256)
        kept.append(candidate)
    return kept


def collect(config: Config) -> dict[str, int]:
    root = config.dataset.output_dir
    root.mkdir(parents=True, exist_ok=True)
    papers = discover_papers(config)
    source_client = CachedHttpClient(
        root / "cache" / "archives",
        config.arxiv.user_agent,
        timeout=config.arxiv.request_timeout_seconds,
        min_interval=config.arxiv.download_delay_seconds,
    )
    license_client = CachedHttpClient(
        root / "cache" / "licenses",
        config.arxiv.user_agent,
        timeout=config.arxiv.request_timeout_seconds,
        min_interval=config.arxiv.api_delay_seconds,
    )
    candidate_dir = root / "candidates" / "images"
    source_root = root / "work" / "sources"
    logs_dir = root / "work" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []
    candidates: list[Candidate] = []
    desired_pool = round(config.dataset.target_figures * config.dataset.oversample_factor)

    for paper_index, paper in enumerate(papers, start=1):
        deduplicated_count = len(
            _deduplicate(candidates, config.selection.dedupe_hamming_distance)
        )
        if deduplicated_count >= desired_pool:
            break
        if paper_index == 1 or paper_index % 20 == 0:
            print(
                f"[collect] papers={paper_index - 1}/{len(papers)} "
                f"candidates={deduplicated_count}/{desired_pool}",
                file=sys.stderr,
                flush=True,
            )
        safe_id = _safe_id(paper.arxiv_id)
        paper_root = source_root / safe_id
        try:
            if not (paper_root / ".extracted").exists():
                source_version = paper.versioned_id or paper.arxiv_id
                payload = source_client.get(
                    f"https://arxiv.org/src/{source_version}",
                    f"source-{safe_id}",
                    ".bin",
                )
                extract_source_archive(payload, paper_root)
                (paper_root / ".extracted").write_text("ok\n", encoding="utf-8")
            figures = parse_figures(paper_root)
        except (OSError, UnsafeArchiveError, ValueError) as exc:
            errors.append({"arxiv_id": paper.arxiv_id, "stage": "source", "error": str(exc)})
            continue

        paper_candidates: list[Candidate] = []
        for figure in figures:
            text_passes, _ = passes_text_filters(figure, config.selection)
            if not text_passes:
                continue
            source_path = Path(figure.source_path)
            candidate_id = (
                f"{safe_id}_f{figure.figure_index:03d}_g{figure.graphic_index:02d}"
            )
            image_path = candidate_dir / f"{candidate_id}.png"
            try:
                width, height = normalize_image(source_path, image_path)
            except (OSError, RuntimeError, UnsupportedImageError, ValueError) as exc:
                errors.append(
                    {"arxiv_id": paper.arxiv_id, "stage": "image", "error": str(exc)}
                )
                continue
            passes, reason = passes_hard_filters(
                figure, width, height, config.selection
            )
            if not passes:
                image_path.unlink(missing_ok=True)
                continue
            score, reasons = score_figure(
                figure, width, height, source_path.suffix, config.selection
            )
            if score < config.selection.min_score:
                image_path.unlink(missing_ok=True)
                continue
            paper_candidates.append(
                Candidate(
                    candidate_id=candidate_id,
                    paper=paper,
                    figure=figure,
                    image_path=str(image_path.resolve()),
                    source_image_path=str(source_path.resolve()),
                    width=width,
                    height=height,
                    score=score,
                    score_reasons=reasons,
                    perceptual_hash=difference_hash(image_path),
                    sha256=file_sha256(image_path),
                )
            )

        if not paper_candidates:
            continue
        paper.license_url = fetch_license(license_client, paper)
        allowed = _license_allowed(paper.license_url, config.license.allowed_urls)
        for candidate in sorted(paper_candidates, key=lambda item: item.score, reverse=True)[
            : config.selection.max_figures_per_paper
        ]:
            candidate.paper.license_url = paper.license_url
            candidate.license_allowed = allowed
            candidates.append(candidate)

    candidates = _deduplicate(candidates, config.selection.dedupe_hamming_distance)
    candidates = candidates[:desired_pool]
    previous_review = root / "review" / "review.csv"
    if previous_review.exists():
        statuses = read_review_status(previous_review)
        for candidate in candidates:
            candidate.review_status, candidate.review_notes = statuses.get(
                candidate.candidate_id, ("pending", "")
            )
    selected_paths = {Path(candidate.image_path).resolve() for candidate in candidates}
    for image_path in candidate_dir.glob("*.png"):
        if image_path.resolve() not in selected_paths:
            image_path.unlink()
    manifest = root / "candidates" / "manifest.jsonl"
    write_jsonl(manifest, (candidate.to_dict(root) for candidate in candidates))
    write_jsonl(logs_dir / "errors.jsonl", errors)
    write_review_bundle(root, candidates)
    summary = {
        "papers_discovered": len(papers),
        "candidates": len(candidates),
        "license_allowed": sum(item.license_allowed for item in candidates),
        "errors": len(errors),
    }
    (root / "candidates" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[collect] complete candidates={len(candidates)} errors={len(errors)}",
        file=sys.stderr,
        flush=True,
    )
    return summary


def finalize(
    config: Config,
    *,
    include_unknown_license: bool = False,
    take: int | None = None,
) -> dict[str, int]:
    root = config.dataset.output_dir
    raw = read_jsonl(root / "candidates" / "manifest.jsonl")
    candidates = [candidate_from_dict(row, root) for row in raw]
    review_path = root / "review" / "review.csv"
    statuses = read_review_status(review_path)
    release_dir = root / "release"
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True)

    accepted: list[Candidate] = []
    excluded_license = 0
    for candidate in candidates:
        status, notes = statuses.get(candidate.candidate_id, ("pending", ""))
        candidate.review_status = status
        candidate.review_notes = notes
        if status != "accept":
            continue
        if not candidate.license_allowed and not include_unknown_license:
            excluded_license += 1
            continue
        accepted.append(candidate)
    accepted.sort(key=lambda item: item.score, reverse=True)
    accepted = accepted[: take or config.dataset.target_figures]

    release_rows: list[dict] = []
    for candidate in accepted:
        destination = copy_release_image(candidate, release_dir)
        row = candidate.to_dict()
        row["image_path"] = str(destination.relative_to(release_dir)).replace("\\", "/")
        row["source_image_path"] = ""
        release_rows.append(row)
    write_jsonl(release_dir / "manifest.jsonl", release_rows)
    summary = {
        "released": len(accepted),
        "target": take or config.dataset.target_figures,
        "accepted_but_license_excluded": excluded_license,
    }
    (release_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def dataset_stats(config: Config) -> dict:
    root = config.dataset.output_dir
    rows = read_jsonl(root / "candidates" / "manifest.jsonl")
    categories = Counter(
        row["paper"].get("primary_category", "unknown") or "unknown" for row in rows
    )
    return {
        "candidate_count": len(rows),
        "license_allowed": sum(bool(row.get("license_allowed")) for row in rows),
        "review_status": dict(Counter(row.get("review_status", "pending") for row in rows)),
        "primary_categories": dict(categories.most_common()),
    }
