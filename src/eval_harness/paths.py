"""Repository path discovery shared by installed harness commands."""

from __future__ import annotations

from pathlib import Path


def default_project_root(start: Path | None = None) -> Path:
    """Find the nearest PPTBench project rooted by ``project.toml``."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "project.toml").is_file():
            return candidate
    return current
