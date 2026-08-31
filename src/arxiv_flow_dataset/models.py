from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: str
    updated: str
    abs_url: str
    pdf_url: str
    source_url: str
    doi: str = ""
    journal_ref: str = ""
    license_url: str = ""
    query: str = ""
    versioned_id: str = ""


@dataclass(slots=True)
class FigureRef:
    tex_file: str
    source_path: str
    caption_latex: str
    caption_text: str
    label: str
    figure_index: int
    graphic_index: int
    graphics_in_environment: int


@dataclass(slots=True)
class Candidate:
    candidate_id: str
    paper: Paper
    figure: FigureRef
    image_path: str
    source_image_path: str
    width: int
    height: int
    score: float
    score_reasons: list[str] = field(default_factory=list)
    perceptual_hash: str = ""
    sha256: str = ""
    license_allowed: bool = False
    review_status: str = "pending"
    review_notes: str = ""

    def to_dict(self, root: Path | None = None) -> dict[str, Any]:
        result = asdict(self)
        result["description"] = self.figure.caption_text
        if root:
            for key in ("image_path", "source_image_path"):
                value = result[key]
                if value:
                    try:
                        result[key] = str(Path(value).resolve().relative_to(root.resolve()))
                    except ValueError:
                        pass
        return result
