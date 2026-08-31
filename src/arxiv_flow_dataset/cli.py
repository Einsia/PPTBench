from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .html_fallback import build_arxiv_html_fallbacks
from .pipeline import collect, dataset_stats, discover_papers, finalize
from .structured import build_structured_dataset
from .structured_collect import scan_structured_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pptbench-collect",
        description="Collect caption-aligned scientific flow diagrams for PPTBench",
    )
    parser.add_argument("--config", default="config.toml", help="TOML configuration path")
    commands = parser.add_subparsers(dest="command", required=True)
    discover_parser = commands.add_parser("discover", help="query and cache paper metadata")
    discover_parser.add_argument(
        "--refresh",
        action="store_true",
        help="rerun the configured queries and replace the paper metadata manifest",
    )
    commands.add_parser("collect", help="download sources, extract figures, score and dedupe")
    finalize_parser = commands.add_parser("finalize", help="build release from review.csv")
    finalize_parser.add_argument(
        "--include-unknown-license",
        action="store_true",
        help="include accepted figures without a redistributable license for local research only",
    )
    finalize_parser.add_argument("--take", type=int, default=None)
    commands.add_parser("stats", help="print candidate statistics")
    structured_parser = commands.add_parser(
        "structured",
        help="build a source-structured TikZ/SVG flow-diagram dataset without OCR",
    )
    structured_parser.add_argument(
        "--output-dir",
        default="data/pptbench-structured",
        help="clean dataset output directory",
    )
    structured_parser.add_argument(
        "--tectonic",
        required=True,
        help="path to the Tectonic executable used to render TikZ sources",
    )
    structured_parser.add_argument("--limit", type=int, default=None)
    structured_parser.add_argument("--max-per-paper", type=int, default=2)
    structured_parser.add_argument(
        "--node",
        default=None,
        help="optional Node.js executable for browser-accurate SVG rendering",
    )
    structured_parser.add_argument(
        "--node-modules",
        default=None,
        help="optional runtime node_modules containing Playwright",
    )
    structured_scan_parser = commands.add_parser(
        "structured-scan",
        help="download arXiv sources and retain only TikZ/SVG flow candidates",
    )
    structured_scan_parser.add_argument("--target", type=int, default=None)
    structured_scan_parser.add_argument("--max-per-paper", type=int, default=None)
    structured_scan_parser.add_argument("--workers", type=int, default=2)
    fallback_parser = commands.add_parser(
        "structured-html-fallback",
        help="recover failed TikZ renders from matching SVG figures in arXiv HTML",
    )
    fallback_parser.add_argument("--output-dir", required=True)
    fallback_parser.add_argument("--node", required=True)
    fallback_parser.add_argument("--node-modules", required=True)
    fallback_parser.add_argument("--max-per-paper", type=int, default=4)
    fallback_parser.add_argument("--min-similarity", type=float, default=0.5)
    fallback_parser.add_argument("--refresh-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "discover":
        result = {"papers_discovered": len(discover_papers(config, refresh=args.refresh))}
    elif args.command == "collect":
        result = collect(config)
    elif args.command == "finalize":
        result = finalize(
            config,
            include_unknown_license=args.include_unknown_license,
            take=args.take,
        )
    elif args.command == "structured":
        result = build_structured_dataset(
            config.dataset.output_dir,
            Path(args.output_dir).resolve(),
            Path(args.tectonic).resolve(),
            limit=args.limit,
            max_per_paper=args.max_per_paper,
            node=Path(args.node).resolve() if args.node else None,
            node_modules=(
                Path(args.node_modules).resolve() if args.node_modules else None
            ),
        )
    elif args.command == "structured-scan":
        result = scan_structured_sources(
            config,
            target_figures=args.target,
            max_per_paper=args.max_per_paper,
            workers=args.workers,
        )
    elif args.command == "structured-html-fallback":
        result = build_arxiv_html_fallbacks(
            config.dataset.output_dir,
            Path(args.output_dir).resolve(),
            Path(args.node).resolve(),
            Path(args.node_modules).resolve(),
            max_per_paper=args.max_per_paper,
            min_similarity=args.min_similarity,
            refresh_existing=args.refresh_existing,
        )
    else:
        result = dataset_stats(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
