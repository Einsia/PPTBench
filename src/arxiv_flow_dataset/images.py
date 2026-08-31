from __future__ import annotations

import hashlib
from pathlib import Path

import fitz
from PIL import Image, ImageOps


class UnsupportedImageError(ValueError):
    pass


def _load_image(path: Path, dpi: int = 220, render_max_side: int = 4096) -> Image.Image:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        document = fitz.open(path)
        try:
            if document.page_count < 1:
                raise UnsupportedImageError(f"empty PDF: {path}")
            page = document.load_page(0)
            page_side = max(page.rect.width, page.rect.height)
            if page_side <= 0:
                raise UnsupportedImageError(f"invalid PDF page dimensions: {path}")
            scale = min(dpi / 72, render_max_side / page_side)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=True)
            return Image.frombytes("RGBA", (pixmap.width, pixmap.height), pixmap.samples)
        finally:
            document.close()
    if suffix in {".svg", ".eps"}:
        raise UnsupportedImageError(f"conversion support is unavailable for {suffix}")
    image = Image.open(path)
    image.load()
    return ImageOps.exif_transpose(image)


def normalize_image(source: Path, destination: Path, max_side: int = 4096) -> tuple[int, int]:
    image = _load_image(source)
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")

    if max(image.size) > max_side:
        ratio = max_side / max(image.size)
        image = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
            Image.Resampling.LANCZOS,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    return image.size


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(path: Path, size: int = 8) -> str:
    with Image.open(path) as image:
        gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        pixel_reader = getattr(gray, "get_flattened_data", gray.getdata)
        pixels = list(pixel_reader())
    bits = 0
    for row in range(size):
        offset = row * (size + 1)
        for column in range(size):
            bits = (bits << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return f"{bits:0{size * size // 4}x}"


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()
