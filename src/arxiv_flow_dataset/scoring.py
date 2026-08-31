from __future__ import annotations

from .config import SelectionConfig
from .models import FigureRef


def passes_text_filters(figure: FigureRef, config: SelectionConfig) -> tuple[bool, str]:
    if len(figure.caption_text) < config.min_caption_chars:
        return False, "caption-too-short"
    if config.require_single_graphic and figure.graphics_in_environment != 1:
        return False, "multi-graphic-environment"
    if config.require_positive_keyword and not any(
        keyword.casefold() in figure.caption_text.casefold()
        for keyword in config.positive_keywords
    ):
        return False, "no-flow-keyword"
    return True, ""


def score_figure(
    figure: FigureRef,
    width: int,
    height: int,
    source_suffix: str,
    config: SelectionConfig,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    caption = figure.caption_text.casefold()

    for keyword in config.positive_keywords:
        if keyword.casefold() in caption:
            score += 3.0
            reasons.append(f"positive:{keyword}")
    for keyword in config.negative_keywords:
        if keyword.casefold() in caption:
            score -= 3.0
            reasons.append(f"negative:{keyword}")

    if len(figure.caption_text) >= 80:
        score += 1.0
        reasons.append("descriptive-caption")
    if len(figure.caption_text) >= 180:
        score += 0.5
        reasons.append("detailed-caption")

    area = width * height
    if area >= 1_000_000:
        score += 1.0
        reasons.append("high-resolution")
    if area >= 2_500_000:
        score += 0.5
        reasons.append("very-high-resolution")

    aspect = width / height
    if 1.2 <= aspect <= 3.2:
        score += 1.0
        reasons.append("diagram-like-aspect")
    if source_suffix.lower() in {".pdf", ".svg", ".eps"}:
        score += 1.0
        reasons.append("vector-source")
    if figure.graphics_in_environment == 1:
        score += 0.5
        reasons.append("single-graphic-figure")
    return score, reasons


def passes_hard_filters(
    figure: FigureRef, width: int, height: int, config: SelectionConfig
) -> tuple[bool, str]:
    text_passes, text_reason = passes_text_filters(figure, config)
    if not text_passes:
        return False, text_reason
    if width < config.min_width or height < config.min_height:
        return False, "image-too-small"
    if width * height < config.min_area:
        return False, "image-area-too-small"
    aspect = width / height
    if not config.min_aspect_ratio <= aspect <= config.max_aspect_ratio:
        return False, "extreme-aspect-ratio"
    return True, ""
