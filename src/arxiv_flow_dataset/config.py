from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


DEFAULT_POSITIVE = [
    "architecture",
    "pipeline",
    "workflow",
    "flowchart",
    "framework",
    "overview",
    "schematic",
    "system diagram",
]

DEFAULT_NEGATIVE = [
    "confusion matrix",
    "ablation",
    "accuracy",
    "loss curve",
    "roc curve",
    "screenshot",
    "histogram",
]


@dataclass(slots=True)
class DatasetConfig:
    name: str = "pptbench-collection"
    output_dir: Path = Path("data/pptbench-collection")
    target_figures: int = 200
    oversample_factor: float = 2.0
    random_seed: int = 20260710


@dataclass(slots=True)
class ArxivConfig:
    queries: list[str] = field(default_factory=list)
    results_per_query: int = 160
    page_size: int = 80
    api_delay_seconds: float = 3.0
    download_delay_seconds: float = 1.0
    request_timeout_seconds: int = 60
    user_agent: str = "PPTBench/1.0 (contact@example.org)"
    submitted_after: str = ""
    submitted_before: str = ""


@dataclass(slots=True)
class SelectionConfig:
    min_width: int = 700
    min_height: int = 350
    min_area: int = 400_000
    min_aspect_ratio: float = 0.55
    max_aspect_ratio: float = 4.5
    min_caption_chars: int = 35
    min_score: float = 4.0
    require_positive_keyword: bool = True
    max_figures_per_paper: int = 2
    require_single_graphic: bool = True
    dedupe_hamming_distance: int = 5
    positive_keywords: list[str] = field(default_factory=lambda: DEFAULT_POSITIVE.copy())
    negative_keywords: list[str] = field(default_factory=lambda: DEFAULT_NEGATIVE.copy())


@dataclass(slots=True)
class LicenseConfig:
    allowed_urls: list[str] = field(default_factory=list)
    allow_unknown_for_local_research: bool = True


@dataclass(slots=True)
class Config:
    dataset: DatasetConfig
    arxiv: ArxivConfig
    selection: SelectionConfig
    license: LicenseConfig


def _known_fields(cls: type, values: dict) -> dict:
    names = cls.__dataclass_fields__.keys()
    return {key: value for key, value in values.items() if key in names}


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    dataset_values = _known_fields(DatasetConfig, raw.get("dataset", {}))
    output = Path(dataset_values.get("output_dir", DatasetConfig().output_dir))
    if not output.is_absolute():
        output = (config_path.parent / output).resolve()
    dataset_values["output_dir"] = output

    return Config(
        dataset=DatasetConfig(**dataset_values),
        arxiv=ArxivConfig(**_known_fields(ArxivConfig, raw.get("arxiv", {}))),
        selection=SelectionConfig(**_known_fields(SelectionConfig, raw.get("selection", {}))),
        license=LicenseConfig(**_known_fields(LicenseConfig, raw.get("license", {}))),
    )
