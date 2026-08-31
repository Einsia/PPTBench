from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(slots=True)
class BenchmarkConfig:
    tasks_manifest: Path
    tasks_root: Path
    output_dir: Path
    sample_count: int = 10
    canvas_width: int = 1600
    canvas_height: int = 900


@dataclass(slots=True)
class Config:
    benchmark: BenchmarkConfig


def _resolve(base: Path, value: str) -> Path:
    """Resolve a path relative to a configuration file."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> Config:
    """Load benchmark paths and canvas settings from a TOML file."""
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    values = raw["benchmark"]
    benchmark = BenchmarkConfig(
        tasks_manifest=_resolve(config_path.parent, values["tasks_manifest"]),
        tasks_root=_resolve(config_path.parent, values["tasks_root"]),
        output_dir=_resolve(config_path.parent, values["output_dir"]),
        sample_count=int(values.get("sample_count", 10)),
        canvas_width=int(values.get("canvas_width", 1600)),
        canvas_height=int(values.get("canvas_height", 900)),
    )
    return Config(benchmark=benchmark)
