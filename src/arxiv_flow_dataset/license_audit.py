"""Audit whether frozen arXiv-derived task assets may be redistributed.

The audit is deliberately stored beside, rather than inside, frozen task metadata so
that verifying licenses does not change task hashes.  License data is version-specific:
the exact ``paper.abs_url`` recorded by each task is therefore the source of truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Iterable

from .http import CachedHttpClient


ARXIV_LICENSE_HELP = "https://info.arxiv.org/help/license/index.html"
ARXIV_REUSE_HELP = "https://info.arxiv.org/help/license/reuse.html"
ARXIV_API_TERMS = "https://info.arxiv.org/help/api/tou.html"
DEFAULT_USER_AGENT = "PPTBench-license-audit/1.0 (https://github.com/Einsia/PPTBench)"


@dataclass(frozen=True)
class LicenseDecision:
    family: str
    release_status: str
    redistribution_grant: bool
    conditions: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class AuditRow:
    task_id: str
    arxiv_id: str
    versioned_id: str
    title: str
    authors: tuple[str, ...]
    abs_url: str
    license_url: str
    license_family: str
    release_status: str
    redistribution_grant: bool
    conditions: tuple[str, ...]
    rationale: str
    evidence_source: str
    verified_on: str


class _AbsLicenseParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._license_depth = 0
        self.license_url = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        classes = set(values.get("class", "").split())
        if tag.lower() == "div" and "abs-license" in classes:
            self._license_depth = 1
            return
        if self._license_depth:
            self._license_depth += 1
            if tag.lower() == "a" and values.get("href") and not self.license_url:
                self.license_url = values["href"].strip()

    def handle_endtag(self, tag: str) -> None:
        if self._license_depth:
            self._license_depth -= 1


def extract_license_url(html: str) -> str:
    parser = _AbsLicenseParser()
    parser.feed(html)
    return parser.license_url


def _normalized_license_url(url: str) -> str:
    value = url.strip().lower().replace("https://", "http://")
    value = re.sub(r"(?:license\.html)?/?$", "", value)
    return value


def classify_license(url: str) -> LicenseDecision:
    """Classify a version license for a conservative public benchmark release.

    ``release-ready`` means the license itself grants redistribution of the task
    figure.  Attribution and ShareAlike still have to be satisfied.  Non-commercial
    and NoDerivatives licenses are not treated as release-ready because PPTBench is
    intended as a generally reusable benchmark and ``reference.png`` is a derived
    rendering/crop of the source figure.
    """

    normalized = _normalized_license_url(url)
    if not normalized:
        return LicenseDecision(
            "unknown",
            "review-required",
            False,
            (),
            "No version-specific license evidence was found.",
        )
    if (
        "arxiv.org/licenses/nonexclusive-distrib" in normalized
        or "arxiv.org/licenses/assumed-1991-2003" in normalized
    ):
        return LicenseDecision(
            "arxiv-nonexclusive",
            "exclude-assets",
            False,
            (),
            "The arXiv non-exclusive license grants distribution rights to arXiv, not to PPTBench.",
        )
    if "creativecommons.org/publicdomain/zero" in normalized:
        return LicenseDecision(
            "cc0",
            "release-ready",
            True,
            (),
            "CC0 permits redistribution and adaptation without license conditions.",
        )
    if "creativecommons.org/publicdomain/mark" in normalized:
        return LicenseDecision(
            "public-domain-mark",
            "release-ready",
            True,
            (),
            "The source marks the work as being in the public domain.",
        )
    if "creativecommons.org/licenses/by-nc-nd/" in normalized:
        return LicenseDecision(
            "cc-by-nc-nd",
            "exclude-assets",
            False,
            ("attribution", "noncommercial", "no-derivatives"),
            "The derived reference rendering is not safely redistributable under NoDerivatives.",
        )
    if "creativecommons.org/licenses/by-nc-sa/" in normalized:
        return LicenseDecision(
            "cc-by-nc-sa",
            "review-required",
            True,
            ("attribution", "noncommercial", "share-alike"),
            "Redistribution is conditional and incompatible with an unrestricted benchmark release.",
        )
    if "creativecommons.org/licenses/by-nc/" in normalized:
        return LicenseDecision(
            "cc-by-nc",
            "review-required",
            True,
            ("attribution", "noncommercial"),
            "Redistribution is limited to non-commercial use.",
        )
    if "creativecommons.org/licenses/by-nd/" in normalized:
        return LicenseDecision(
            "cc-by-nd",
            "exclude-assets",
            False,
            ("attribution", "no-derivatives"),
            "The derived reference rendering is not safely redistributable under NoDerivatives.",
        )
    if "creativecommons.org/licenses/by-sa/" in normalized:
        return LicenseDecision(
            "cc-by-sa",
            "release-ready",
            True,
            ("attribution", "share-alike"),
            "Redistribution and adaptation are permitted with attribution and ShareAlike.",
        )
    if "creativecommons.org/licenses/by/" in normalized:
        return LicenseDecision(
            "cc-by",
            "release-ready",
            True,
            ("attribution",),
            "Redistribution and adaptation are permitted with attribution.",
        )
    return LicenseDecision(
        "other",
        "review-required",
        False,
        (),
        "The detected license is not covered by the automatic release policy.",
    )


def _read_tasks(tasks_root: Path) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for task_dir in sorted(tasks_root.glob("task_*")):
        metadata_path = task_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing task metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("task_id") != task_dir.name:
            raise ValueError(f"task ID mismatch in {metadata_path}")
        tasks.append(metadata)
    if not tasks:
        raise ValueError(f"no task directories found under {tasks_root}")
    return tasks


def _row_from_metadata(
    metadata: dict[str, object],
    *,
    license_url: str,
    evidence_source: str,
    verified_on: str,
) -> AuditRow:
    paper = metadata.get("paper")
    if not isinstance(paper, dict):
        raise ValueError(f"{metadata.get('task_id', '<unknown>')} has no paper metadata")
    decision = classify_license(license_url)
    authors = paper.get("authors", [])
    return AuditRow(
        task_id=str(metadata.get("task_id", "")),
        arxiv_id=str(paper.get("arxiv_id") or metadata.get("arxiv_id") or ""),
        versioned_id=str(paper.get("versioned_id", "")),
        title=str(paper.get("title", "")),
        authors=tuple(str(author) for author in authors) if isinstance(authors, list) else (),
        abs_url=str(paper.get("abs_url", "")),
        license_url=license_url,
        license_family=decision.family,
        release_status=decision.release_status,
        redistribution_grant=decision.redistribution_grant,
        conditions=decision.conditions,
        rationale=decision.rationale,
        evidence_source=evidence_source,
        verified_on=verified_on,
    )


def audit_source_licenses(
    tasks_root: Path,
    *,
    cache_dir: Path,
    online: bool,
    user_agent: str = DEFAULT_USER_AGENT,
    min_interval: float = 3.1,
    verified_on: str | None = None,
) -> list[AuditRow]:
    tasks = _read_tasks(tasks_root)
    client = CachedHttpClient(
        cache_dir,
        user_agent,
        timeout=60,
        min_interval=min_interval,
        max_attempts=4,
        max_response_bytes=2_000_000,
    )
    rows: list[AuditRow] = []
    audit_date = verified_on or date.today().isoformat()
    for index, metadata in enumerate(tasks, start=1):
        paper = metadata["paper"]
        if not isinstance(paper, dict):
            raise ValueError(
                f"task metadata at index {index} must contain a paper object"
            )
        arxiv_id = str(paper.get("arxiv_id") or metadata.get("arxiv_id") or "")
        versioned_id = str(paper.get("versioned_id") or arxiv_id)
        license_url = str(paper.get("license_url", "")).strip()
        evidence_source = "task-metadata"
        if not license_url and online:
            abs_url = str(paper.get("abs_url") or f"https://arxiv.org/abs/{versioned_id}")
            payload = client.get(abs_url, f"abs-{versioned_id.replace('/', '_')}", ".html")
            license_url = extract_license_url(payload.decode("utf-8", errors="replace"))
            evidence_source = "arxiv-abstract-page"
        rows.append(
            _row_from_metadata(
                metadata,
                license_url=license_url,
                evidence_source=evidence_source,
                verified_on=audit_date,
            )
        )
        if online and (index % 10 == 0 or index == len(tasks)):
            print(f"audited {index}/{len(tasks)} tasks", flush=True)
    return rows


def summarize(rows: Iterable[AuditRow]) -> dict[str, object]:
    materialized = list(rows)
    status_counts = Counter(row.release_status for row in materialized)
    family_counts = Counter(row.license_family for row in materialized)
    task_ids_by_status = {
        status: [row.task_id for row in materialized if row.release_status == status]
        for status in sorted(status_counts)
    }
    task_ids_by_family = {
        family: [row.task_id for row in materialized if row.license_family == family]
        for family in sorted(family_counts)
    }
    return {
        "schema": "pptbench-source-license-audit-summary-v1",
        "verified_on": sorted({row.verified_on for row in materialized}),
        "task_count": len(materialized),
        "unique_papers": len({row.arxiv_id for row in materialized}),
        "release_status_counts": dict(sorted(status_counts.items())),
        "license_family_counts": dict(sorted(family_counts.items())),
        "task_ids_by_release_status": task_ids_by_status,
        "task_ids_by_license_family": task_ids_by_family,
        "release_ready": bool(materialized) and status_counts.get("release-ready", 0) == len(materialized),
        "policy": {
            "release_ready": ["cc-by", "cc-by-sa", "cc0", "public-domain-mark"],
            "review_required": ["cc-by-nc", "cc-by-nc-sa", "other", "unknown"],
            "exclude_assets": ["cc-by-nd", "cc-by-nc-nd", "arxiv-nonexclusive"],
        },
        "evidence": {
            "arxiv_license_help": ARXIV_LICENSE_HELP,
            "arxiv_reuse_help": ARXIV_REUSE_HELP,
            "arxiv_api_terms": ARXIV_API_TERMS,
        },
    }


def render_report(summary: dict[str, object]) -> str:
    status_counts = summary["release_status_counts"]
    family_counts = summary["license_family_counts"]
    if not isinstance(status_counts, dict):
        raise TypeError("summary.release_status_counts must be a mapping")
    if not isinstance(family_counts, dict):
        raise TypeError("summary.license_family_counts must be a mapping")
    ready = bool(summary["release_ready"])
    lines = [
        "# Source data license audit",
        "",
        f"Audit date: `{date.today().isoformat()}`",
        "",
        "This is a reproducible provenance and license check, not legal advice. The audit uses the",
        "license attached to the exact arXiv version recorded by each frozen task. It does not infer",
        "reuse rights from free access, a DOI, or a publisher's open-access label.",
        "",
        "## Release decision",
        "",
        (
            "**BUNDLED-ASSET RELEASE: PASS.** All source assets have a recognized redistribution grant."
            if ready
            else "**BUNDLED-ASSET RELEASE: BLOCKED.** Use the downloader-only release; do not commit paper pixels."
        ),
        "",
        "The public PPTBench package is downloader-only: it ships task specifications and hashes,",
        "not source PDFs, reference renderings, or raster resource payloads.",
        "",
        f"- Tasks audited: **{summary['task_count']}**",
        f"- Unique source papers: **{summary['unique_papers']}**",
        f"- Release-ready: **{status_counts.get('release-ready', 0)}**",
        f"- Manual review required: **{status_counts.get('review-required', 0)}**",
        f"- Assets to exclude or replace with download instructions: **{status_counts.get('exclude-assets', 0)}**",
        "",
        "## Detected licenses",
        "",
        "| License family | Tasks |",
        "|---|---:|",
    ]
    for family, count in sorted(family_counts.items()):
        lines.append(f"| `{family}` | {count} |")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- `release-ready`: CC BY, CC BY-SA, CC0, or a public-domain mark. Required",
            "  attribution and ShareAlike conditions still apply per task.",
            "- `review-required`: unknown, unrecognized, or non-commercial licenses. These are not",
            "  included in an unrestricted public release without separate review/permission.",
            "- `exclude-assets`: arXiv non-exclusive or NoDerivatives licenses. Keep only provenance",
            "  metadata and provide a deterministic downloader, unless separate permission is recorded.",
            "",
            "The full per-task evidence, title, authors, exact arXiv version, license URL, conditions,",
            "and decision are in `benchmark/source_license_audit.jsonl`. The machine-readable summary",
            "is in `benchmark/source_license_audit_summary.json`.",
            "",
            "## Authoritative references",
            "",
            f"- [arXiv license information]({ARXIV_LICENSE_HELP})",
            f"- [arXiv permissions and reuse FAQ]({ARXIV_REUSE_HELP})",
            f"- [arXiv API terms and rate limits]({ARXIV_API_TERMS})",
            "",
            "arXiv states that its default non-exclusive license grants distribution rights to arXiv",
            "and does not itself grant third parties permission to mirror e-prints or figures.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_audit(
    rows: list[AuditRow], output: Path, summary_output: Path, report_output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output)
    summary = summarize(rows)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_output.write_text(render_report(summary), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pptbench-license-audit",
        description=__doc__,
    )
    parser.add_argument("--tasks-root", type=Path, default=Path("benchmark/tasks"))
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark/source_license_audit.jsonl")
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("benchmark/source_license_audit_summary.json"),
    )
    parser.add_argument(
        "--report-output", type=Path, default=Path("SOURCE_LICENSE_AUDIT.md")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/source-license-audit")
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not fetch missing licenses; useful for checking committed metadata only",
    )
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument(
        "--min-interval",
        type=float,
        default=3.1,
        help="seconds between requests; arXiv requires at least 3 seconds",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.offline and args.min_interval < 3.0:
        raise SystemExit("--min-interval must be at least 3 seconds under the arXiv API terms")
    rows = audit_source_licenses(
        args.tasks_root,
        cache_dir=args.cache_dir,
        online=not args.offline,
        user_agent=args.user_agent,
        min_interval=args.min_interval,
    )
    write_audit(rows, args.output, args.summary_output, args.report_output)
    summary = summarize(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["release_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
