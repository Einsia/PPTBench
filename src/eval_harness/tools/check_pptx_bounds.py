from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}

BOUNDS_TOLERANCE_EMU = 1


def parse_integral_emu(value: str) -> int:
    """Parse an OOXML EMU value, accepting an integral decimal spelling.

    Some authoring libraries serialize an integer coordinate as ``123.0``.
    LibreOffice accepts that spelling, so the bounds audit should not crash
    before it can classify the deck. Fractional and non-finite coordinates
    remain invalid.
    """

    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"invalid EMU coordinate: {value!r}") from exc
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise ValueError(f"EMU coordinate must be a finite integer: {value!r}")
    return int(parsed)


def check_deck(path: Path) -> list[str]:
    """Return every slide object whose bounding box exceeds the slide canvas.

    Args:
        path: PowerPoint deck to inspect as an Open XML ZIP package.

    Returns:
        Human-readable issue strings; an empty list means all parsed shapes,
        pictures, and connectors remain inside their slide dimensions.
    """

    issues: list[str] = []
    with zipfile.ZipFile(path) as archive:
        presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
        size = presentation.find("p:sldSz", NS)
        if size is None:
            return [f"{path.name}: missing slide size"]
        width, height = parse_integral_emu(size.attrib["cx"]), parse_integral_emu(
            size.attrib["cy"]
        )
        slides = sorted(
            name
            for name in archive.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        for slide_name in slides:
            root = ET.fromstring(archive.read(slide_name))
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
                if (
                    x < -BOUNDS_TOLERANCE_EMU
                    or y < -BOUNDS_TOLERANCE_EMU
                    or x + cx > width + BOUNDS_TOLERANCE_EMU
                    or y + cy > height + BOUNDS_TOLERANCE_EMU
                ):
                    issues.append(
                        f"{path.name} {slide_name}: bounds {(x, y, cx, cy)} exceed {(width, height)}"
                    )
    return issues


def main() -> int:
    """Check CLI-supplied decks and return a nonzero status on any violation.

    Returns:
        Zero when every object is within bounds, otherwise one.
    """

    parser = argparse.ArgumentParser(
        description="Check PPTX object bounds without modifying the deck"
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    issues: list[str] = []
    for path in args.paths:
        issues.extend(check_deck(path))
    if issues:
        print("\n".join(issues))
        return 1
    print(f"checked {len(args.paths)} deck(s): all object bounds are inside the slide canvas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
