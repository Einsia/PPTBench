from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

from .models import Candidate, Paper


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def paper_to_dict(paper: Paper) -> dict:
    return asdict(paper)


def candidate_from_dict(value: dict, root: Path | None = None) -> Candidate:
    from .models import FigureRef

    paper = Paper(**value["paper"])
    figure = FigureRef(**value["figure"])
    image_path = value["image_path"]
    source_image_path = value["source_image_path"]
    if root:
        if image_path and not Path(image_path).is_absolute():
            image_path = str((root / image_path).resolve())
        if source_image_path and not Path(source_image_path).is_absolute():
            source_image_path = str((root / source_image_path).resolve())
    fields = Candidate.__dataclass_fields__.keys()
    kwargs = {key: value[key] for key in fields if key in value}
    kwargs.update(paper=paper, figure=figure, image_path=image_path, source_image_path=source_image_path)
    return Candidate(**kwargs)
