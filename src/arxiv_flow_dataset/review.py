from __future__ import annotations

import csv
from html import escape
from pathlib import Path
import shutil

from PIL import Image

from .models import Candidate


REVIEW_FIELDS = [
    "candidate_id",
    "status",
    "notes",
    "score",
    "license_allowed",
    "license_url",
    "arxiv_id",
    "title",
    "description",
    "image_path",
    "abs_url",
]


def write_review_bundle(output_dir: Path, candidates: list[Candidate]) -> Path:
    review_dir = output_dir / "review"
    thumbs_dir = review_dir / "thumbnails"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = review_dir / "review.csv"

    rows: list[dict[str, str]] = []
    cards: list[str] = []
    selected_thumbnails: set[Path] = set()
    for candidate in candidates:
        image_path = Path(candidate.image_path)
        thumb_path = thumbs_dir / f"{candidate.candidate_id}.jpg"
        selected_thumbnails.add(thumb_path.resolve())
        if not thumb_path.exists():
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((520, 320), Image.Resampling.LANCZOS)
                image.save(thumb_path, quality=88, optimize=True)
        relative_image = Path("..") / image_path.relative_to(output_dir)
        row = {
            "candidate_id": candidate.candidate_id,
            "status": candidate.review_status,
            "notes": candidate.review_notes,
            "score": f"{candidate.score:.2f}",
            "license_allowed": str(candidate.license_allowed).lower(),
            "license_url": candidate.paper.license_url,
            "arxiv_id": candidate.paper.arxiv_id,
            "title": candidate.paper.title,
            "description": candidate.figure.caption_text,
            "image_path": str(relative_image).replace("\\", "/"),
            "abs_url": candidate.paper.abs_url,
        }
        rows.append(row)
        cards.append(
            "<article>"
            f'<a href="{escape(row["image_path"])}"><img loading="lazy" '
            f'src="thumbnails/{escape(thumb_path.name)}"></a>'
            f'<h2>{escape(candidate.candidate_id)} · {candidate.score:.1f}</h2>'
            f'<p class="paper">{escape(candidate.paper.title)}</p>'
            f'<p>{escape(candidate.figure.caption_text)}</p>'
            f'<p><a href="{escape(candidate.paper.abs_url)}">paper</a> · '
            f'license: {escape(candidate.paper.license_url or "unknown")}</p>'
            "</article>"
        )

    for thumb_path in thumbs_dir.glob("*.jpg"):
        if thumb_path.resolve() not in selected_thumbnails:
            thumb_path.unlink()

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    html = """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Figure review</title>
<style>
body{font:14px/1.45 system-ui,sans-serif;margin:24px;background:#f5f6f8;color:#18202a}
header{max-width:1100px;margin:auto auto 20px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px}
article{background:white;padding:14px;border-radius:10px;box-shadow:0 1px 5px #0002}img{width:100%;height:280px;object-fit:contain;background:#fff}
h2{font-size:15px;margin:10px 0 5px}.paper{font-weight:650}p{margin:7px 0}a{color:#075ccf}
</style><header><h1>arXiv flow figure review</h1>
<p>Edit <code>review.csv</code> and set status to <code>accept</code> or <code>reject</code></p></header><main>"""
    html += "\n".join(cards) + "</main></html>"
    (review_dir / "gallery.html").write_text(html, encoding="utf-8")
    return csv_path


def read_review_status(path: Path) -> dict[str, tuple[str, str]]:
    statuses: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            statuses[row["candidate_id"]] = (
                row.get("status", "pending").strip().lower(),
                row.get("notes", "").strip(),
            )
    return statuses


def copy_release_image(candidate: Candidate, release_dir: Path) -> Path:
    destination = release_dir / "images" / f"{candidate.candidate_id}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate.image_path, destination)
    return destination
