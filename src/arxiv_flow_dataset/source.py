from __future__ import annotations

from io import BytesIO
import gzip
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import zipfile


class UnsafeArchiveError(ValueError):
    pass


STRUCTURED_TEXT_SUFFIXES = {".tex", ".ltx", ".tikz", ".pgf"}
TIKZ_MARKER_RE = re.compile(rb"\\begin\s*\{tikzpicture\}", re.IGNORECASE)
SVG_MARKER_RE = re.compile(rb"<svg(?:\s|>)", re.IGNORECASE)


def _source_markers(name: str, payload: bytes) -> tuple[bool, bool]:
    suffix = PurePosixPath(name.replace("\\", "/")).suffix.lower()
    tikz = suffix in STRUCTURED_TEXT_SUFFIXES and bool(TIKZ_MARKER_RE.search(payload))
    svg = suffix == ".svg" and bool(SVG_MARKER_RE.search(payload[:1_000_000]))
    return tikz, svg


def inspect_source_archive(
    payload: bytes,
    *,
    max_files: int = 10_000,
    max_scan_bytes: int = 250_000_000,
) -> dict[str, int | bool]:
    """Inspect an arXiv source response without extracting it to disk."""
    tikz_files = 0
    svg_files = 0
    scanned_files = 0
    scanned_bytes = 0

    def inspect(name: str, content: bytes) -> None:
        nonlocal tikz_files, svg_files, scanned_files, scanned_bytes
        suffix = PurePosixPath(name.replace("\\", "/")).suffix.lower()
        if suffix not in STRUCTURED_TEXT_SUFFIXES | {".svg"}:
            return
        scanned_files += 1
        scanned_bytes += len(content)
        if scanned_bytes > max_scan_bytes:
            raise UnsafeArchiveError("archive scan exceeds the configured size limit")
        tikz, svg = _source_markers(name, content)
        tikz_files += int(tikz)
        svg_files += int(svg)

    try:
        with tarfile.open(fileobj=BytesIO(payload), mode="r:*") as archive:
            regular = [member for member in archive.getmembers() if member.isfile()]
            if len(regular) > max_files:
                raise UnsafeArchiveError("archive has too many files")
            for member in regular:
                suffix = PurePosixPath(member.name.replace("\\", "/")).suffix.lower()
                if suffix not in STRUCTURED_TEXT_SUFFIXES | {".svg"}:
                    continue
                source = archive.extractfile(member)
                if source is not None:
                    with source:
                        inspect(member.name, source.read())
    except tarfile.ReadError:
        if zipfile.is_zipfile(BytesIO(payload)):
            with zipfile.ZipFile(BytesIO(payload)) as archive:
                entries = [entry for entry in archive.infolist() if not entry.is_dir()]
                if len(entries) > max_files:
                    raise UnsafeArchiveError("archive has too many files")
                for entry in entries:
                    suffix = PurePosixPath(entry.filename.replace("\\", "/")).suffix.lower()
                    if suffix in STRUCTURED_TEXT_SUFFIXES | {".svg"}:
                        with archive.open(entry) as source:
                            inspect(entry.filename, source.read())
        else:
            try:
                content = gzip.decompress(payload)
            except gzip.BadGzipFile:
                content = payload
            inspect("main.tex", content)

    return {
        "has_structured_source": bool(tikz_files or svg_files),
        "tikz_files": tikz_files,
        "svg_files": svg_files,
        "scanned_files": scanned_files,
        "scanned_bytes": scanned_bytes,
    }


def _safe_destination(root: Path, member_name: str) -> Path:
    normalized = PurePosixPath(member_name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise UnsafeArchiveError(f"unsafe archive path: {member_name}")
    destination = (root / Path(*normalized.parts)).resolve()
    if root.resolve() not in (destination, *destination.parents):
        raise UnsafeArchiveError(f"archive path escapes target: {member_name}")
    return destination


def extract_source_archive(
    payload: bytes,
    target_dir: Path,
    *,
    max_files: int = 10_000,
    max_total_bytes: int = 1_000_000_000,
) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    try:
        with tarfile.open(fileobj=BytesIO(payload), mode="r:*") as archive:
            members = archive.getmembers()
            regular = [member for member in members if member.isfile()]
            if len(regular) > max_files:
                raise UnsafeArchiveError("archive has too many files")
            if sum(member.size for member in regular) > max_total_bytes:
                raise UnsafeArchiveError("archive expands beyond the configured size limit")
            for member in regular:
                destination = _safe_destination(target_dir, member.name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    continue
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted.append(destination)
            return extracted
    except tarfile.ReadError:
        pass

    if zipfile.is_zipfile(BytesIO(payload)):
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            if len(entries) > max_files:
                raise UnsafeArchiveError("archive has too many files")
            if sum(entry.file_size for entry in entries) > max_total_bytes:
                raise UnsafeArchiveError("archive expands beyond the configured size limit")
            for entry in entries:
                destination = _safe_destination(target_dir, entry.filename)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted.append(destination)
            return extracted

    try:
        content = gzip.decompress(payload)
    except gzip.BadGzipFile:
        content = payload
    if len(content) > max_total_bytes:
        raise UnsafeArchiveError("single-file source is too large")
    destination = target_dir / "main.tex"
    destination.write_bytes(content)
    return [destination]
