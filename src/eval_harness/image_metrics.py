"""Deterministic visual metrics for rendered PowerPoint slides."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Load an image as a resized floating-point RGB array."""

    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB").resize(size, Image.Resampling.LANCZOS),
            dtype=np.float64,
        )


def block_ssim(reference: np.ndarray, candidate: np.ndarray, block: int = 8) -> float:
    """Return a compact blockwise structural-similarity estimate."""

    left = np.dot(reference[..., :3], [0.299, 0.587, 0.114])
    right = np.dot(candidate[..., :3], [0.299, 0.587, 0.114])
    height = left.shape[0] - left.shape[0] % block
    width = left.shape[1] - left.shape[1] % block
    left = left[:height, :width].reshape(height // block, block, width // block, block)
    right = right[:height, :width].reshape(height // block, block, width // block, block)
    mu_left = left.mean(axis=(1, 3))
    mu_right = right.mean(axis=(1, 3))
    var_left = left.var(axis=(1, 3))
    var_right = right.var(axis=(1, 3))
    covariance = ((left - mu_left[:, None, :, None]) * (right - mu_right[:, None, :, None])).mean(
        axis=(1, 3)
    )
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    score = ((2 * mu_left * mu_right + c1) * (2 * covariance + c2)) / (
        (mu_left**2 + mu_right**2 + c1) * (var_left + var_right + c2)
    )
    return float(np.clip(score.mean(), -1, 1))


def edge_f1(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return a tolerant F1 score over detected image edges."""

    def edges(array: np.ndarray) -> np.ndarray:
        image = Image.fromarray(array.astype(np.uint8)).convert("L").filter(ImageFilter.FIND_EDGES)
        return np.asarray(image) > 28

    left = edges(reference)
    right = edges(candidate)
    left_dilated = np.asarray(Image.fromarray(left).filter(ImageFilter.MaxFilter(3))) > 0
    right_dilated = np.asarray(Image.fromarray(right).filter(ImageFilter.MaxFilter(3))) > 0
    precision = float((right & left_dilated).sum() / max(1, right.sum()))
    recall = float((left & right_dilated).sum() / max(1, left.sum()))
    return 2 * precision * recall / max(1e-9, precision + recall)


def compare(reference_path: Path, candidate_path: Path) -> dict[str, float]:
    """Compare a rendered PowerPoint slide with its reference image."""

    size = (800, 450)
    reference = _rgb(reference_path, size)
    candidate = _rgb(candidate_path, size)
    pixel = float(1 - np.abs(reference - candidate).mean() / 255)
    ssim = block_ssim(reference, candidate)
    edge = edge_f1(reference, candidate)
    composite = 0.6 * max(0, ssim) + 0.25 * pixel + 0.15 * edge
    return {
        "pixel_similarity": round(pixel, 6),
        "ssim": round(ssim, 6),
        "edge_f1": round(edge, 6),
        "composite": round(composite, 6),
    }
