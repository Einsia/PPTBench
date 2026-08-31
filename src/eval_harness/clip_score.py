from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


class ImageEncoder(Protocol):
    model_name: str

    def encode(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        """Encode image paths as one normalized feature vector per image.

        Args:
            paths: Reference or reconstruction images to encode in order.
            batch_size: Maximum images processed in one model forward pass.

        Returns:
            A two-dimensional array with one feature row for every input path.
        """

        ...


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute row-wise cosine similarity between two embedding matrices."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim == 1:
        left = left[None, :]
    if right.ndim == 1:
        right = right[None, :]
    if left.shape != right.shape:
        raise ValueError(f"embedding shapes differ: {left.shape} != {right.shape}")
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = left_norm * right_norm
    if np.any(denominator <= 0):
        raise ValueError("CLIP returned a zero-length embedding")
    return np.sum(left * right, axis=1) / denominator


def clip_score_from_cosine(value: float) -> float:
    """Map image cosine similarity to a simple 0--100 benchmark score."""

    return round(max(0.0, min(1.0, float(value))) * 100, 4)


class TransformersClipEncoder:
    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: str = "auto") -> None:
        """Load a Transformers CLIP encoder on the requested device."""
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel
        except ImportError as exc:  # pragma: no cover - exercised by CLI environment
            raise RuntimeError(
                'CLIP dependencies are missing; install with `pip install -e ".[clip]"`'
            ) from exc
        self._torch = torch
        self.model_name = model_name
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).eval().to(device)

    def encode(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        """Encode image paths into feature vectors."""
        outputs: list[np.ndarray] = []
        for start in range(0, len(paths), max(1, batch_size)):
            images = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as loaded:
                    images.append(loaded.convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt")
            pixels = inputs["pixel_values"].to(self.device)
            with self._torch.inference_mode():
                features = self.model.get_image_features(pixel_values=pixels)
                # transformers 5.x currently returns the vision-model output
                # object here, whereas 4.x returned projected embeddings.
                if not self._torch.is_tensor(features):
                    pooled = features.pooler_output
                    projection = self.model.visual_projection
                    features = (
                        projection(pooled) if pooled.shape[-1] == projection.in_features else pooled
                    )
            outputs.append(features.detach().float().cpu().numpy())
        if not outputs:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def encode_text(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode text strings into feature vectors."""
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), max(1, batch_size)):
            inputs = self.processor(
                text=texts[start : start + batch_size],
                padding=True,
                return_tensors="pt",
            )
            model_inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                features = self.model.get_text_features(**model_inputs)
                if not self._torch.is_tensor(features):
                    pooled = features.pooler_output
                    projection = self.model.text_projection
                    features = (
                        projection(pooled) if pooled.shape[-1] == projection.in_features else pooled
                    )
            outputs.append(features.detach().float().cpu().numpy())
        if not outputs:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(outputs, axis=0)


def score_pairs(
    pairs: list[dict],
    *,
    encoder: ImageEncoder,
    batch_size: int = 16,
) -> list[dict]:
    """Compute CLIP similarities for reference/candidate image pairs."""
    if not pairs:
        return []
    references = [Path(pair["reference_path"]) for pair in pairs]
    candidates = [Path(pair["candidate_path"]) for pair in pairs]
    missing = [str(path) for path in (*references, *candidates) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing CLIP inputs:\n" + "\n".join(missing[:20]))
    left = encoder.encode(references, batch_size=batch_size)
    right = encoder.encode(candidates, batch_size=batch_size)
    similarities = cosine_similarity(left, right)
    return [
        {
            **pair,
            "clip_model": encoder.model_name,
            "clip_cosine": round(float(similarity), 6),
            "clip_score": clip_score_from_cosine(float(similarity)),
        }
        for pair, similarity in zip(pairs, similarities, strict=True)
    ]


def _load_pairs(path: Path) -> list[dict]:
    """Load JSONL image pairs and resolve paths relative to the input file."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        for key in ("reference_path", "candidate_path"):
            value = Path(row[key])
            if not value.is_absolute():
                row[key] = str((path.parent / value).resolve())
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the optional CLIP scorer."""
    parser = argparse.ArgumentParser(description="Image-to-image CLIP similarity scorer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pairs", type=Path, help="JSONL rows with reference_path/candidate_path")
    source.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point and return its process status."""
    args = build_parser().parse_args(argv)
    if args.pairs:
        pairs = _load_pairs(args.pairs)
    else:
        if args.candidate is None:
            raise SystemExit("--candidate is required with --reference")
        pairs = [
            {
                "reference_path": str(args.reference.resolve()),
                "candidate_path": str(args.candidate.resolve()),
            }
        ]
    encoder = TransformersClipEncoder(args.model, device=args.device)
    rows = score_pairs(pairs, encoder=encoder, batch_size=args.batch_size)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
