from __future__ import annotations

from dataclasses import dataclass
from html import escape
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable
from xml.etree import ElementTree

from .images import difference_hash, file_sha256, normalize_image
from .latex import _balanced_argument, _graphic_paths, latex_to_text, resolve_graphic, strip_comments
from .models import Paper


TIKZ_BEGIN_RE = re.compile(r"\\begin\s*\{tikzpicture\}", re.IGNORECASE)
FIGURE_RE = re.compile(
    r"\\begin\s*\{figure\*?\}(.*?)\\end\s*\{figure\*?\}",
    re.IGNORECASE | re.DOTALL,
)
INPUT_RE = re.compile(r"\\(?:input|include|subfile)\s*\{([^{}]+)\}", re.IGNORECASE)
INCLUDE_RE = re.compile(
    r"\\includegraphics\s*\*?\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}",
    re.IGNORECASE | re.DOTALL,
)
RASTER_RE = re.compile(
    r"\\(?:includegraphics|pgfimage)\b|\\addplot\s+graphics|<image\b",
    re.IGNORECASE,
)
PLOT_RE = re.compile(
    r"\\begin\s*\{axis\}|\\addplot\b|\\begin\s*\{semilog|\\begin\s*\{loglog|"
    r"use\s+as\s+bounding\s+box|\\path\s*\[\s*clip\b",
    re.IGNORECASE,
)
FLOW_TERMS = (
    "architecture",
    "block diagram",
    "causal chain",
    "closed-loop integration",
    "configuration interface",
    "context management hierarchy",
    "data flow",
    "design topology",
    "diagram",
    "flow graph",
    "flowchart",
    "framework",
    "entanglement generation protocol",
    "first-stage decisions",
    "interaction",
    "knowledge recovery arc",
    "lifecycle",
    "logical flow",
    "method overview",
    "model overview",
    "network overview",
    "overall architecture",
    "overview",
    "online path",
    "pipeline",
    "process",
    "processing flow",
    "resolution path",
    "routing",
    "steps of monte-carlo tree search",
    "schematic",
    "system overview",
    "stage evolution",
    "state space evolution",
    "streaming fpga path",
    "test selection",
    "transfer learning",
    "workflow",
)
HARD_PLOT_TERMS = (
    "accuracy curve",
    "angular multiplexing",
    "bar chart",
    "box plot",
    "confusion matrix",
    "histogram",
    "heatmap",
    "loss curve",
    "precision-recall",
    "risk set",
    "roc curve",
    "scatter plot",
    "measurement setup",
    "ofdm frame",
)
RENDER_VERSION = 3


@dataclass(slots=True)
class StructuredFigure:
    paper_id: str
    source_kind: str
    source_file: Path
    source_code: str
    support_code: str
    caption_latex: str
    caption_text: str
    label: str
    figure_index: int
    graphic_index: int
    graph: dict


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _balanced_environment_blocks(text: str, environment: str) -> list[tuple[int, int, str]]:
    begin_re = re.compile(rf"\\begin\s*\{{{re.escape(environment)}\}}", re.IGNORECASE)
    token_re = re.compile(
        rf"\\(?:begin|end)\s*\{{{re.escape(environment)}\}}",
        re.IGNORECASE,
    )
    blocks: list[tuple[int, int, str]] = []
    cursor = 0
    while match := begin_re.search(text, cursor):
        depth = 0
        end = -1
        for token in token_re.finditer(text, match.start()):
            depth += 1 if token.group(0).lower().lstrip().startswith(r"\begin") else -1
            if depth == 0:
                end = token.end()
                break
        if end < 0:
            break
        blocks.append((match.start(), end, text[match.start() : end]))
        cursor = end
    return blocks


def _balanced_groups(text: str) -> list[str]:
    groups: list[str] = []
    start = -1
    depth = 0
    for index, character in enumerate(text):
        escaped = index > 0 and text[index - 1] == "\\"
        if character == "{" and not escaped:
            if depth == 0:
                start = index + 1
            depth += 1
        elif character == "}" and not escaped and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                groups.append(text[start:index])
    return groups


def _commands(text: str, names: Iterable[str]) -> list[str]:
    name_pattern = "|".join(re.escape(name) for name in names)
    command_re = re.compile(rf"\\(?:{name_pattern})\b", re.IGNORECASE)
    commands: list[str] = []
    for match in command_re.finditer(text):
        brace_depth = 0
        bracket_depth = 0
        cursor = match.end()
        while cursor < len(text):
            character = text[cursor]
            escaped = cursor > 0 and text[cursor - 1] == "\\"
            if character == "{" and not escaped:
                brace_depth += 1
            elif character == "}" and not escaped:
                brace_depth = max(0, brace_depth - 1)
            elif character == "[" and not escaped:
                bracket_depth += 1
            elif character == "]" and not escaped:
                bracket_depth = max(0, bracket_depth - 1)
            elif character == ";" and brace_depth == 0 and bracket_depth == 0:
                commands.append(text[match.start() : cursor + 1])
                break
            cursor += 1
    return commands


def _clean_label(value: str) -> str:
    value = re.sub(r"\\\\", " ", value)
    value = re.sub(r"\\(?:textbf|textit|emph|mathrm|mathbf|texttt)\s*\{([^{}]*)\}", r"\1", value)
    return latex_to_text(value)[:300]


def _node_name(command: str) -> str:
    for value in re.findall(r"\(([^()]*)\)", command):
        value = value.strip().split(".", 1)[0]
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_:.\-]*", value):
            return value
    return ""


def _shape_from_command(command: str) -> str:
    lowered = command.lower()
    for shape in ("diamond", "ellipse", "circle", "cloud", "cylinder"):
        if shape in lowered:
            return shape
    if "rounded corners" in lowered or "rounded rectangle" in lowered:
        return "roundRect"
    return "rect" if "draw" in lowered else "text"


def parse_tikz_graph(source: str) -> dict:
    raster_count = len(RASTER_RE.findall(source))
    raw_nodes = list(
        re.finditer(r"%\s*Shape:\s*([^\[]+?)\s*\[id:([^\]]+)\]", source, re.IGNORECASE)
    )
    raw_connectors = list(
        re.finditer(r"%\s*(?:Straight|Curve|Elbow|Freeform)\s+Lines?\s*\[id:([^\]]+)\]", source, re.IGNORECASE)
    )
    cleaned = strip_comments(source)
    node_commands = _commands(cleaned, ("node",))
    text_labels = []
    command_nodes = []
    for index, command in enumerate(node_commands, start=1):
        groups = _balanced_groups(command)
        label = _clean_label(groups[-1]) if groups else ""
        name = _node_name(command) or f"node-{index:03d}"
        command_nodes.append(
            {
                "id": name,
                "label": label,
                "shape": _shape_from_command(command),
                "source": command[:1000],
            }
        )
        if label:
            text_labels.append(label)

    if raw_nodes:
        nodes = [
            {
                "id": match.group(2).strip(),
                "label": text_labels[index] if index < len(text_labels) else "",
                "shape": match.group(1).strip(),
                "source": match.group(0),
            }
            for index, match in enumerate(raw_nodes)
        ]
    else:
        seen: set[str] = set()
        nodes = []
        for node in command_nodes:
            if node["id"] in seen:
                continue
            seen.add(node["id"])
            nodes.append(node)

    node_ids = {node["id"] for node in nodes}
    if raw_connectors:
        connectors = [
            {
                "id": match.group(1).strip(),
                "source": "",
                "target": "",
                "directed": True,
            }
            for match in raw_connectors
        ]
    else:
        connectors = []
        for index, command in enumerate(_commands(cleaned, ("draw", "path")), start=1):
            lowered = command.lower()
            if not any(marker in lowered for marker in ("->", "<-", " edge ", " to ", "--")):
                continue
            references = []
            for value in re.findall(r"\(([A-Za-z][A-Za-z0-9_:.\-]*)\)", command):
                name = value.split(".", 1)[0]
                if name in node_ids and name not in references:
                    references.append(name)
            connectors.append(
                {
                    "id": f"edge-{index:03d}",
                    "source": references[0] if references else "",
                    "target": references[-1] if len(references) > 1 else "",
                    "directed": "->" in command or "<-" in command,
                    "source_code": command[:1000],
                }
            )

    labels = [node["label"] for node in nodes if node.get("label")]
    return {
        "format": "tikz",
        "nodes": nodes,
        "connectors": connectors,
        "groups": [],
        "labels": labels[:60],
        "node_count": len(nodes),
        "connector_count": len(connectors),
        "embedded_raster_count": raster_count,
    }


def _svg_tag(element: ElementTree.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].lower()


def parse_svg_graph(source: str) -> dict:
    root = ElementTree.fromstring(source)
    elements = list(root.iter())
    embedded = [element for element in elements if _svg_tag(element) == "image"]
    labels = [
        " ".join("".join(element.itertext()).split())[:300]
        for element in elements
        if _svg_tag(element) == "text" and "".join(element.itertext()).strip()
    ]
    shape_tags = {"rect", "circle", "ellipse", "polygon"}
    nodes = []
    for index, element in enumerate(
        (item for item in elements if _svg_tag(item) in shape_tags), start=1
    ):
        nodes.append(
            {
                "id": element.attrib.get("id", f"node-{index:03d}"),
                "label": labels[index - 1] if index <= len(labels) else "",
                "shape": _svg_tag(element),
            }
        )
    connectors = []
    for index, element in enumerate(
        (
            item
            for item in elements
            if _svg_tag(item) in {"line", "polyline", "path"}
            and (
                "marker-end" in item.attrib
                or "marker-start" in item.attrib
                or _svg_tag(item) in {"line", "polyline"}
                or (
                    _svg_tag(item) == "path"
                    and item.attrib.get("fill", "").strip().lower() == "none"
                    and item.attrib.get("stroke", "").strip().lower() not in {"", "none"}
                    and (
                        item.attrib.get("pointer-events", "").lower() == "stroke"
                        or any(
                            term in (
                                item.attrib.get("id", "")
                                + " "
                                + item.attrib.get("class", "")
                            ).lower()
                            for term in ("edge", "connector")
                        )
                    )
                )
            )
        ),
        start=1,
    ):
        connectors.append(
            {
                "id": element.attrib.get("id", f"edge-{index:03d}"),
                "source": "",
                "target": "",
                "directed": "marker-end" in element.attrib,
            }
        )
    return {
        "format": "svg",
        "nodes": nodes,
        "connectors": connectors,
        "groups": [item.attrib.get("id", "") for item in elements if _svg_tag(item) == "g"],
        "labels": labels[:60],
        "node_count": len(nodes),
        "connector_count": len(connectors),
        "embedded_raster_count": len(embedded),
    }


def _resolve_input(source_root: Path, tex_file: Path, value: str) -> Path | None:
    value = value.strip().replace("\\", "/")
    if not value or "#" in value:
        return None
    relative = Path(value)
    candidates = [tex_file.parent / relative, source_root / relative]
    if not relative.suffix:
        base_candidates = list(candidates)
        for suffix in (".tex", ".tikz", ".pgf", ".ltx"):
            candidates.extend(path.with_suffix(suffix) for path in base_candidates)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _support_code(text: str, before: int | None = None) -> str:
    prefix = text if before is None else text[:before]
    snippets: list[str] = []
    for pattern in (
        r"(?m)^\s*\\usetikzlibrary\s*\{[^\n]+$",
        r"(?m)^\s*\\definecolor\s*\{[^\n]+$",
        r"(?m)^\s*\\colorlet\s*\{[^\n]+$",
        r"(?m)^\s*\\tikzstyle\s*\{[^\n]+$",
    ):
        for match in re.finditer(pattern, prefix):
            snippet = match.group(0).strip()
            if snippet.count("{") == snippet.count("}"):
                snippets.append(snippet)
    command_re = re.compile(
        r"(?m)^\s*\\(?:newcommand|renewcommand|providecommand)\b[^\n]+$"
    )
    for match in command_re.finditer(prefix):
        command = match.group(0).strip()
        if not re.match(
            r"\\(?:newcommand|renewcommand|providecommand)\s*\{\\[A-Za-z]+\}",
            command,
        ):
            continue
        if command.count("{") == command.count("}") and command.count("{") >= 2:
            command = re.sub(
                r"^\\(?:newcommand|renewcommand)",
                r"\\providecommand",
                command,
            )
            snippets.append(command)
    for match in re.finditer(r"\\tikzset\s*\{", prefix, re.IGNORECASE):
        start = match.start()
        cursor = match.end()
        depth = 1
        while cursor < len(prefix) and depth:
            depth += (prefix[cursor] == "{") - (prefix[cursor] == "}")
            cursor += 1
        if depth == 0:
            snippets.append(prefix[start:cursor].strip())
    cleaned: list[str] = []
    for item in snippets:
        item = item.strip()
        if item.lower().startswith("\\tikzstyle"):
            item = item.rstrip(";").rstrip()
        if item:
            cleaned.append(item)
    return "\n".join(dict.fromkeys(cleaned))


def discover_structured_figures(paper_root: Path, paper_id: str) -> list[StructuredFigure]:
    tex_files = sorted(
        path
        for path in paper_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tex", ".ltx", ".tikz", ".pgf"}
    )
    contents: dict[Path, str] = {}
    for tex_file in tex_files:
        try:
            contents[tex_file.resolve()] = _read_text(tex_file)
        except OSError:
            continue

    support_sources = list(contents.values())
    global_support = "\n".join(
        dict.fromkeys(
            support
            for source in support_sources
            if (support := _support_code(source))
        )
    )

    figures: list[StructuredFigure] = []
    seen: set[tuple[str, str]] = set()
    figure_index = 0

    def add_tikz(
        source_file: Path,
        source_text: str,
        block_start: int,
        block: str,
        caption_latex: str,
        label: str,
        parent_support: str,
        graphic_index: int,
    ) -> None:
        key = (str(source_file.resolve()), str(hash(block)))
        if key in seen:
            return
        seen.add(key)
        graph = parse_tikz_graph(block)
        figures.append(
            StructuredFigure(
                paper_id=paper_id,
                source_kind="tikz",
                source_file=source_file.resolve(),
                source_code=block,
                support_code="\n".join(
                    item
                    for item in (
                        global_support,
                        parent_support,
                        _support_code(source_text, block_start),
                    )
                    if item
                ),
                caption_latex=caption_latex,
                caption_text=latex_to_text(caption_latex),
                label=label,
                figure_index=figure_index,
                graphic_index=graphic_index,
                graph=graph,
            )
        )

    for tex_file, text in contents.items():
        for figure_match in FIGURE_RE.finditer(text):
            figure_index += 1
            environment = figure_match.group(1)
            caption_latex = _balanced_argument(environment, "caption")
            label = _balanced_argument(environment, "label")
            parent_support = _support_code(text, figure_match.start())
            graphic_index = 0
            for start, _, block in _balanced_environment_blocks(environment, "tikzpicture"):
                graphic_index += 1
                add_tikz(
                    tex_file,
                    text,
                    figure_match.start() + start,
                    block,
                    caption_latex,
                    label,
                    parent_support,
                    graphic_index,
                )
            for input_value in INPUT_RE.findall(environment):
                input_file = _resolve_input(paper_root, tex_file, input_value)
                if input_file is None or input_file not in contents:
                    continue
                input_text = contents[input_file]
                for start, _, block in _balanced_environment_blocks(input_text, "tikzpicture"):
                    graphic_index += 1
                    add_tikz(
                        input_file,
                        input_text,
                        start,
                        block,
                        caption_latex,
                        label,
                        parent_support,
                        graphic_index,
                    )
            graphics_paths = _graphic_paths(strip_comments(text))
            for raw_path in INCLUDE_RE.findall(environment):
                resolved = resolve_graphic(paper_root, tex_file, raw_path, graphics_paths)
                if resolved is None:
                    continue
                if resolved.suffix.lower() == ".svg":
                    svg_sources = [resolved]
                else:
                    svg_sources = [
                        candidate
                        for candidate in (
                            resolved.with_suffix(".svg"),
                            *sorted(resolved.parent.glob(f"{resolved.stem}.*.svg")),
                        )
                        if candidate.is_file()
                    ]
                for svg_source in dict.fromkeys(svg_sources):
                    graphic_index += 1
                    source = _read_text(svg_source)
                    key = (str(svg_source), str(hash(source)))
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        graph = parse_svg_graph(source)
                    except ElementTree.ParseError:
                        continue
                    figures.append(
                        StructuredFigure(
                            paper_id=paper_id,
                            source_kind="svg",
                            source_file=svg_source,
                            source_code=source,
                            support_code="",
                            caption_latex=caption_latex,
                            caption_text=latex_to_text(caption_latex),
                            label=label,
                            figure_index=figure_index,
                            graphic_index=graphic_index,
                            graph=graph,
                        )
                    )

    for tex_file, text in contents.items():
        for start, _, block in _balanced_environment_blocks(text, "tikzpicture"):
            key = (str(tex_file), str(hash(block)))
            if key in seen:
                continue
            figure_index += 1
            add_tikz(
                tex_file,
                text,
                start,
                block,
                "",
                "",
                "",
                1,
            )

    for svg_file in sorted(paper_root.rglob("*.svg")):
        try:
            source = _read_text(svg_file)
            key = (str(svg_file.resolve()), str(hash(source)))
            if key in seen:
                continue
            graph = parse_svg_graph(source)
        except (OSError, ElementTree.ParseError):
            continue
        seen.add(key)
        figure_index += 1
        figures.append(
            StructuredFigure(
                paper_id=paper_id,
                source_kind="svg",
                source_file=svg_file.resolve(),
                source_code=source,
                support_code="",
                caption_latex="",
                caption_text="",
                label="",
                figure_index=figure_index,
                graphic_index=1,
                graph=graph,
            )
        )
    return figures


def structured_score(figure: StructuredFigure) -> tuple[float, list[str]]:
    graph = figure.graph
    if graph["embedded_raster_count"]:
        return -100.0, ["embedded-raster"]
    if graph["node_count"] < 3 or graph["connector_count"] < 2:
        return -100.0, ["insufficient-graph-structure"]
    if graph["connector_count"] > max(100, graph["node_count"] * 8):
        return -100.0, ["excessively-dense-graph"]
    source_lower = figure.source_code.lower()
    if PLOT_RE.search(source_lower):
        return -100.0, ["plot-source"]
    searchable = " ".join(
        (
            figure.caption_text,
            figure.source_file.stem.replace("_", " "),
            " ".join(graph.get("labels", [])),
        )
    ).lower()
    if any(term in searchable for term in HARD_PLOT_TERMS):
        return -100.0, ["plot-caption"]
    matched = [term for term in FLOW_TERMS if term in searchable]
    if not matched:
        return -100.0, ["no-flow-keyword"]
    score = min(20.0, 4.0 + 1.5 * len(matched))
    score += min(8.0, graph["node_count"] * 0.35)
    score += min(8.0, graph["connector_count"] * 0.35)
    if figure.caption_text:
        score += 2.0
    if figure.source_kind == "tikz":
        score += 2.0
    return round(score, 3), [*(f"flow:{term}" for term in matched), "structured-source"]


def _tikz_wrapper(figure: StructuredFigure) -> str:
    support = figure.support_code
    return f"""\\PassOptionsToPackage{{dvipsnames,svgnames,x11names}}{{xcolor}}
\\documentclass[tikz,border=4pt]{{standalone}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath,amssymb,mathtools,bm,xcolor,graphicx,url}}
\\usepackage{{tikz}}
\\usetikzlibrary{{arrows,arrows.meta,automata,backgrounds,babel,calc,chains,decorations.markings,decorations.pathmorphing,decorations.pathreplacing,fit,graphs,matrix,patterns,positioning,quotes,shadows,shadows.blur,shapes,shapes.geometric,shapes.misc}}
\\providecommand{{\\expectation}}{{\\mathbb{{E}}}}
\\providecommand{{\\var}}{{\\mathrm{{Var}}}}
\\providecommand{{\\normal}}{{\\mathcal{{N}}}}
\\providecommand{{\\categorical}}{{\\mathrm{{Cat}}}}
\\providecommand{{\\faBalanceScale}}{{}}
\\providecommand{{\\faChartArea}}{{}}
\\providecommand{{\\faCheckCircle}}{{}}
\\providecommand{{\\faClock}}{{}}
\\providecommand{{\\faDatabase}}{{}}
\\providecommand{{\\faEye}}{{}}
\\providecommand{{\\faFileMedical}}{{}}
\\providecommand{{\\faPauseCircle}}{{}}
\\providecommand{{\\faSearchPlus}}{{}}
{support}
\\begin{{document}}
{figure.source_code}
\\end{{document}}
"""


def _render_tikz(
    figure: StructuredFigure,
    sample_dir: Path,
    tectonic: Path,
) -> tuple[Path, str]:
    wrapper = sample_dir / "compile.tex"
    wrapper.write_text(_tikz_wrapper(figure), encoding="utf-8")
    build_dir = sample_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(tectonic),
            "--untrusted",
            "--keep-logs",
            "--outdir",
            str(build_dir),
            str(wrapper),
        ],
        cwd=sample_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    log = (result.stdout + "\n" + result.stderr).strip()
    pdf = build_dir / "compile.pdf"
    if result.returncode or not pdf.exists():
        raise RuntimeError(log[-4000:] or f"tectonic exit code {result.returncode}")
    return pdf, log


def _render_svg_browser(
    source_path: Path,
    render_path: Path,
    node: Path,
    node_modules: Path,
) -> str:
    script = Path(__file__).resolve().parents[2] / "scripts" / "render_svg.mjs"
    environment = os.environ.copy()
    environment["RUNTIME_NODE_MODULES"] = str(node_modules.resolve())
    environment["NODE_PATH"] = str((node_modules / ".pnpm" / "node_modules").resolve())
    result = subprocess.run(
        [
            str(node.resolve()),
            str(script),
            "--input",
            str(source_path.resolve()),
            "--output",
            str(render_path.resolve()),
        ],
        cwd=script.parent.parent,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
        check=False,
    )
    log = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode or not render_path.is_file():
        raise RuntimeError(log[-4000:] or f"SVG browser renderer exit code {result.returncode}")
    return log or "Rendered SVG with sandboxed Chromium"


def _write_gallery(output_dir: Path, rows: list[dict]) -> None:
    cards = []
    for row in rows:
        cards.append(
            "<article>"
            f"<img src='{escape(row['render_path'])}'>"
            f"<h3>{escape(row['structured_id'])}</h3>"
            f"<p>{escape(row['description'])}</p>"
            f"<p>{row['node_count']} nodes · {row['connector_count']} connectors · "
            f"{escape(row['source_kind'])}</p>"
            f"<a href='{escape(row['source_path'])}'>source</a> · "
            f"<a href='{escape(row['graph_path'])}'>graph</a>"
            "</article>"
        )
    html = f"""<!doctype html><meta charset='utf-8'><title>Structured arXiv flow figures</title>
<style>body{{font:14px/1.45 system-ui;margin:24px;background:#f5f7fb}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}}article{{background:white;border:1px solid #d9deea;border-radius:12px;padding:12px}}img{{width:100%;height:220px;object-fit:contain}}h3{{font-size:15px;word-break:break-all}}p{{color:#4d5669}}</style>
<h1>Structured arXiv flow figures</h1><p>{len(rows)} samples · no embedded raster sources</p><main>{''.join(cards)}</main>"""
    (output_dir / "gallery.html").write_text(html, encoding="utf-8")


def _row_semantic_signature(output_dir: Path, row: dict) -> tuple | None:
    caption = re.sub(r"\W+", " ", row.get("description", "").casefold()).strip()
    graph_path = output_dir / row.get("graph_path", "")
    if not caption or not graph_path.is_file():
        return None
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    labels = tuple(
        sorted(
            {
                normalized
                for label in graph.get("labels", [])
                if (normalized := re.sub(r"\W+", " ", str(label).casefold()).strip())
            }
        )
    )
    if len(labels) < 2:
        return None
    return (
        caption,
        labels,
        int(row.get("node_count", 0)),
        int(row.get("connector_count", 0)),
    )


def build_structured_dataset(
    source_dataset: Path,
    output_dir: Path,
    tectonic: Path,
    *,
    limit: int | None = None,
    max_per_paper: int = 2,
    dedupe_hamming_distance: int = 4,
    node: Path | None = None,
    node_modules: Path | None = None,
) -> dict:
    source_dataset = source_dataset.resolve()
    output_dir = output_dir.resolve()
    workspace = Path.cwd().resolve()
    if output_dir in {source_dataset, workspace, Path(output_dir.anchor)}:
        raise ValueError(f"unsafe structured dataset output directory: {output_dir}")
    papers_path = source_dataset / "work" / "papers.jsonl"
    papers = [
        Paper(**json.loads(line))
        for line in papers_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_safe_id = {paper.arxiv_id.replace("/", "_").replace(".", "-"): paper for paper in papers}
    existing_licenses: dict[str, str] = {}
    old_manifest = source_dataset / "candidates" / "manifest.jsonl"
    if old_manifest.exists():
        for line in old_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing_licenses[row["paper"]["arxiv_id"]] = row["paper"].get("license_url", "")

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    partial_manifest = output_dir / "manifest.partial.jsonl"
    manifest = output_dir / "manifest.jsonl"

    def read_rows(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    rows_by_id: dict[str, dict] = {}
    for row in [*read_rows(manifest), *read_rows(partial_manifest)]:
        sample = output_dir / row.get("render_path", "")
        source = output_dir / row.get("source_path", "")
        graph = output_dir / row.get("graph_path", "")
        valid = sample.is_file() and source.is_file() and graph.is_file()
        valid = valid and row.get("connector_count", 0) <= max(
            100, row.get("node_count", 0) * 8
        )
        if valid:
            source_code = source.read_text(encoding="utf-8", errors="replace")
            valid = not PLOT_RE.search(source_code)
            source.write_text(source_code, encoding="utf-8", newline="\n")
            row["source_sha256"] = file_sha256(source)
        if valid:
            rows_by_id[row["structured_id"]] = row
        elif sample.is_file():
            sample_dir = sample.parent.resolve()
            if output_dir in sample_dir.parents:
                shutil.rmtree(sample_dir, ignore_errors=True)
    rows = list(rows_by_id.values())
    partial_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    errors = list(
        {
            row.get("structured_id", f"legacy-{index}"): row
            for index, row in enumerate(read_rows(output_dir / "errors.jsonl"))
        }.values()
    )
    errors_by_id = {row.get("structured_id"): row for row in errors}
    paper_counts: dict[str, int] = {}
    for row in rows:
        paper_id = row["paper"]["arxiv_id"]
        paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1
    source_hashes = {row["source_sha256"] for row in rows}
    render_hashes = [row["perceptual_hash"] for row in rows]

    def append_jsonl(path: Path, row: dict) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def hamming(left: str, right: str) -> int:
        return (int(left, 16) ^ int(right, 16)).bit_count()

    discovered = 0
    rejected = 0
    duplicates = 0
    source_root = source_dataset / "work" / "sources"

    for paper_dir in sorted(source_root.iterdir()):
        if not paper_dir.is_dir() or paper_dir.name not in by_safe_id:
            continue
        paper = by_safe_id[paper_dir.name]
        if paper_counts.get(paper.arxiv_id, 0) >= max_per_paper:
            continue
        paper.license_url = existing_licenses.get(paper.arxiv_id, paper.license_url)
        for figure in discover_structured_figures(paper_dir, paper.arxiv_id):
            if paper_counts.get(paper.arxiv_id, 0) >= max_per_paper:
                break
            discovered += 1
            score, reasons = structured_score(figure)
            if score < 0:
                rejected += 1
                continue
            digest = file_sha256(figure.source_file)[:10]
            structured_id = (
                f"{paper_dir.name}_sf{figure.figure_index:03d}_g{figure.graphic_index:02d}_"
                f"{digest}"
            )
            sample_dir = samples_dir / structured_id
            if structured_id in rows_by_id:
                continue
            source_digest = hashlib.sha256(figure.source_code.encode("utf-8")).hexdigest()
            if source_digest in source_hashes:
                duplicates += 1
                continue
            previous_error = errors_by_id.get(structured_id)
            if (
                previous_error is not None
                and previous_error.get("source_sha256") == source_digest
                and previous_error.get("renderer_version") == RENDER_VERSION
            ):
                continue
            sample_dir.mkdir(parents=True, exist_ok=True)
            suffix = ".tex" if figure.source_kind == "tikz" else ".svg"
            source_path = sample_dir / f"source{suffix}"
            source_path.write_text(
                figure.source_code,
                encoding="utf-8",
                newline="\n",
            )
            graph_path = sample_dir / "graph.json"
            graph_path.write_text(
                json.dumps(figure.graph, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            render_path = sample_dir / "render.png"
            try:
                if figure.source_kind == "tikz":
                    pdf, compile_log = _render_tikz(figure, sample_dir, tectonic)
                    normalize_image(pdf, render_path)
                    (sample_dir / "compile.log").write_text(compile_log, encoding="utf-8")
                    (sample_dir / "compile.tex").unlink(missing_ok=True)
                    shutil.rmtree(sample_dir / "build", ignore_errors=True)
                else:
                    document = ElementTree.fromstring(figure.source_code)
                    if _svg_tag(document) != "svg":
                        raise RuntimeError("invalid SVG root")
                    if node is not None and node_modules is not None:
                        compile_log = _render_svg_browser(
                            source_path,
                            render_path,
                            node,
                            node_modules,
                        )
                        (sample_dir / "compile.log").write_text(
                            compile_log,
                            encoding="utf-8",
                        )
                    else:
                        import fitz

                        svg_document = fitz.open(
                            stream=figure.source_code.encode("utf-8"),
                            filetype="svg",
                        )
                        pdf_bytes = svg_document.convert_to_pdf()
                        svg_document.close()
                        pdf_path = sample_dir / "render.pdf"
                        pdf_path.write_bytes(pdf_bytes)
                        normalize_image(pdf_path, render_path)
                        (sample_dir / "compile.log").write_text(
                            "Rendered SVG with PyMuPDF; pass --node and --node-modules "
                            "to preserve foreignObject text with Chromium\n",
                            encoding="utf-8",
                        )
            except Exception as exc:  # noqa: BLE001
                errors = [
                    row for row in errors if row.get("structured_id") != structured_id
                ]
                error_row = {
                    "structured_id": structured_id,
                    "paper_id": paper.arxiv_id,
                    "source_file": str(figure.source_file),
                    "source_sha256": source_digest,
                    "renderer_version": RENDER_VERSION,
                    "error": str(exc),
                }
                errors.append(error_row)
                errors_by_id[structured_id] = error_row
                append_jsonl(output_dir / "errors.jsonl", error_row)
                shutil.rmtree(sample_dir, ignore_errors=True)
                print(
                    f"[structured] render error {structured_id}: {str(exc).splitlines()[-1]}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            width, height = 0, 0
            from PIL import Image

            with Image.open(render_path) as image:
                width, height = image.size
            render_hash = difference_hash(render_path)
            if any(
                hamming(render_hash, previous) <= dedupe_hamming_distance
                for previous in render_hashes
            ):
                duplicates += 1
                shutil.rmtree(sample_dir, ignore_errors=True)
                continue
            def relative(path: Path) -> str:
                return path.relative_to(output_dir).as_posix()

            row = {
                "structured_id": structured_id,
                "candidate_id": structured_id,
                "description": figure.caption_text or figure.source_file.stem.replace("_", " "),
                "paper": {
                    key: getattr(paper, key)
                    for key in Paper.__dataclass_fields__
                },
                "source_kind": figure.source_kind,
                "source_path": relative(source_path),
                "source_origin": str(figure.source_file.relative_to(source_dataset)),
                "graph_path": relative(graph_path),
                "image_path": relative(render_path),
                "render_path": relative(render_path),
                "caption_latex": figure.caption_latex,
                "label": figure.label,
                "node_count": figure.graph["node_count"],
                "connector_count": figure.graph["connector_count"],
                "embedded_raster_count": 0,
                "width": width,
                "height": height,
                "score": score,
                "score_reasons": reasons,
                "source_sha256": source_digest,
                "render_sha256": file_sha256(render_path),
                "perceptual_hash": render_hash,
            }
            rows.append(row)
            errors = [row for row in errors if row.get("structured_id") != structured_id]
            errors_by_id.pop(structured_id, None)
            rows_by_id[structured_id] = row
            source_hashes.add(source_digest)
            render_hashes.append(render_hash)
            paper_counts[paper.arxiv_id] = paper_counts.get(paper.arxiv_id, 0) + 1
            append_jsonl(partial_manifest, row)
            print(
                f"[structured] ok {structured_id} nodes={row['node_count']} "
                f"connectors={row['connector_count']}",
                file=sys.stderr,
                flush=True,
            )
            if limit is not None and len(rows) >= limit:
                break
        if limit is not None and len(rows) >= limit:
            break

    rows.sort(key=lambda item: (item["score"], item["node_count"]), reverse=True)
    semantic_rows: dict[tuple, dict] = {}
    deduplicated_rows: list[dict] = []
    for row in rows:
        signature = _row_semantic_signature(output_dir, row)
        previous = semantic_rows.get(signature) if signature is not None else None
        if previous is not None and hamming(
            row["perceptual_hash"], previous["perceptual_hash"]
        ) <= max(12, dedupe_hamming_distance):
            duplicates += 1
            sample_dir = (output_dir / row["render_path"]).parent.resolve()
            if output_dir in sample_dir.parents:
                shutil.rmtree(sample_dir, ignore_errors=True)
            continue
        if signature is not None:
            semantic_rows[signature] = row
        deduplicated_rows.append(row)
    rows = deduplicated_rows
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "errors.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in errors),
        encoding="utf-8",
    )
    summary = {
        "source_dataset": str(source_dataset.resolve()),
        "structured_figures_discovered": discovered,
        "rejected_non_flow_or_raster": rejected,
        "rendered_samples": len(rows),
        "render_failures": len(errors),
        "duplicate_sources_or_renders": duplicates,
        "tikz_samples": sum(row["source_kind"] == "tikz" for row in rows),
        "svg_samples": sum(row["source_kind"] == "svg" for row in rows),
        "embedded_raster_samples": 0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_gallery(output_dir, rows)
    return summary
