"""Keep PPTX objects inside the slide canvas without changing their design.

This maintenance command is useful after an authoring runtime emits shapes
slightly outside the slide bounds.  It rewrites only DrawingML transform
coordinates; dimensions, text, styles, relationships, and media are preserved.
Run it through the installed public entry point::

    pptbench-normalize-bounds deck.pptx [another-deck.pptx ...]

The operation is in-place.  Make a copy first when the original package must be
retained for audit purposes.  The same command is also available as
``python -m eval_harness.tools.normalize_pptx_bounds``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from .check_pptx_bounds import parse_integral_emu


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}


def _bounded_offset(offset: int, extent: int, canvas: int) -> int:
    """Translate one object coordinate into the slide without resizing it.

    Args:
        offset: Current x or y position in English Metric Units.
        extent: Current width or height in English Metric Units.
        canvas: Slide width or height in English Metric Units.

    Returns:
        The nearest coordinate that keeps the object inside the canvas. Objects
        larger than the canvas are expected to be clamped before this helper is
        called.
    """

    return min(max(offset, 0), canvas - extent)


def _bounded_extent(extent: int, canvas: int) -> int:
    """Clamp one object dimension to the corresponding slide dimension."""

    return min(max(extent, 0), canvas)


def normalize_deck(path: Path) -> int:
    """Translate out-of-bounds slide objects into the visible PPTX canvas.

    The operation edits only DrawingML transform offsets. Shape dimensions,
    text, styling, relationships, and every non-slide ZIP member remain intact.

    Args:
        path: PowerPoint deck to normalize in place.

    Returns:
        Number of object transforms moved.
    """

    with zipfile.ZipFile(path, "r") as source:
        members = [(item, source.read(item.filename)) for item in source.infolist()]
        presentation = ET.fromstring(source.read("ppt/presentation.xml"))
        size = presentation.find("p:sldSz", NS)
        if size is None:
            raise ValueError(f"{path} has no slide-size declaration")
        width, height = parse_integral_emu(size.attrib["cx"]), parse_integral_emu(
            size.attrib["cy"]
        )

    replacements: dict[str, bytes] = {}
    moved = 0
    for item, payload in members:
        if not (item.filename.startswith("ppt/slides/slide") and item.filename.endswith(".xml")):
            continue
        root = ET.fromstring(payload)
        changed = False
        for element in (
            root.findall(".//p:sp", NS)
            + root.findall(".//p:pic", NS)
            + root.findall(".//p:cxnSp", NS)
        ):
            transform = element.find("p:spPr/a:xfrm", NS)
            if transform is None:
                continue
            offset = transform.find("a:off", NS)
            extent = transform.find("a:ext", NS)
            if offset is None or extent is None:
                continue
            x, y = parse_integral_emu(offset.attrib["x"]), parse_integral_emu(
                offset.attrib["y"]
            )
            cx, cy = parse_integral_emu(extent.attrib["cx"]), parse_integral_emu(
                extent.attrib["cy"]
            )
            bounded_cx = _bounded_extent(cx, width)
            bounded_cy = _bounded_extent(cy, height)
            bounded_x = _bounded_offset(x, bounded_cx, width)
            bounded_y = _bounded_offset(y, bounded_cy, height)
            if (bounded_x, bounded_y, bounded_cx, bounded_cy) != (x, y, cx, cy):
                offset.attrib.update(x=str(bounded_x), y=str(bounded_y))
                extent.attrib.update(cx=str(bounded_cx), cy=str(bounded_cy))
                moved += 1
                changed = True
        if changed:
            replacements[item.filename] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    if not replacements:
        return 0
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}-", suffix=".pptx", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w") as destination:
            for item, payload in members:
                destination.writestr(item, replacements.get(item.filename, payload))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return moved


def main(argv: list[str] | None = None) -> int:
    """Normalize CLI-supplied decks and report how many objects moved."""

    parser = argparse.ArgumentParser(
        prog="pptbench-normalize-bounds",
        description="Translate out-of-bounds PPTX objects into the slide canvas",
        epilog=(
            "The files are edited in place; use a copy if the original artifact "
            "must remain unchanged."
        ),
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    moved = sum(normalize_deck(path) for path in args.paths)
    print(f"normalized {len(args.paths)} deck(s): moved {moved} object(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
