from __future__ import annotations

from pathlib import Path
import re

from .models import FigureRef


GRAPHIC_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".eps", ".svg")
FIGURE_RE = re.compile(
    r"\\begin\s*\{figure\*?\}(.*?)\\end\s*\{figure\*?\}",
    flags=re.IGNORECASE | re.DOTALL,
)
INCLUDE_RE = re.compile(
    r"\\includegraphics\s*\*?\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}",
    flags=re.IGNORECASE | re.DOTALL,
)


def strip_comments(tex: str) -> str:
    lines: list[str] = []
    for line in tex.splitlines():
        index = 0
        while True:
            index = line.find("%", index)
            if index < 0:
                break
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                line = line[:index]
                break
            index += 1
        lines.append(line)
    return "\n".join(lines)


def _balanced_argument(text: str, command: str) -> str:
    match = re.search(rf"\\{command}\*?\s*", text, flags=re.IGNORECASE)
    if not match:
        return ""
    cursor = match.end()
    if cursor < len(text) and text[cursor] == "[":
        depth = 1
        cursor += 1
        while cursor < len(text) and depth:
            depth += (text[cursor] == "[") - (text[cursor] == "]")
            cursor += 1
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
        return ""
    start = cursor + 1
    depth = 1
    cursor += 1
    while cursor < len(text):
        if text[cursor] == "{" and (cursor == 0 or text[cursor - 1] != "\\"):
            depth += 1
        elif text[cursor] == "}" and (cursor == 0 or text[cursor - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start:cursor]
        cursor += 1
    return ""


def latex_to_text(value: str) -> str:
    value = re.sub(
        r"\\(?:cite\w*|(?:auto|[cC]|eq|page|name)?ref|label|footnote)\s*"
        r"(?:\[[^\]]*\])?\s*\{[^{}]*\}",
        "",
        value,
    )
    value = re.sub(r"\\(?:textbf|textit|emph|textrm|texttt|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\(?:url|href)\s*\{([^{}]*)\}(?:\{([^{}]*)\})?", lambda m: m.group(2) or m.group(1), value)
    replacements = {
        r"\&": "&",
        r"\%": "%",
        r"\_": "_",
        r"\#": "#",
        r"~": " ",
        r"\\": " ",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = re.sub(r"\$+([^$]*)\$+", r"\1", value)
    value = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", " ", value)
    value = value.replace("{", "").replace("}", "")
    return " ".join(value.split()).strip()


def _graphic_paths(tex: str) -> list[str]:
    paths: list[str] = []
    for raw_group in re.findall(r"\\graphicspath\s*\{(.*?)\}", tex, flags=re.DOTALL):
        paths.extend(re.findall(r"\{([^{}]*)\}", raw_group))
    return paths


def resolve_graphic(source_root: Path, tex_file: Path, raw_path: str, extra: list[str]) -> Path | None:
    raw_path = raw_path.strip().replace("\\", "/")
    if raw_path.startswith(r"\detokenize{") and raw_path.endswith("}"):
        raw_path = raw_path[len(r"\detokenize{") : -1]
    if not raw_path or "\\" in raw_path or "#" in raw_path:
        return None

    relative = Path(raw_path)
    bases = [tex_file.parent, source_root]
    bases.extend(tex_file.parent / path for path in extra)
    candidates: list[Path] = []
    for base in bases:
        candidate = base / relative
        candidates.append(candidate)
        if not candidate.suffix:
            candidates.extend(candidate.with_suffix(extension) for extension in GRAPHIC_EXTENSIONS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    basename = relative.name.lower()
    stem = relative.stem.lower()
    matches = [
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and (path.name.lower() == basename or (not relative.suffix and path.stem.lower() == stem))
        and path.suffix.lower() in GRAPHIC_EXTENSIONS
    ]
    return matches[0].resolve() if len(matches) == 1 else None


def parse_figures(source_root: Path) -> list[FigureRef]:
    figures: list[FigureRef] = []
    figure_index = 0
    for tex_file in sorted(source_root.rglob("*.tex")):
        try:
            tex = tex_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        tex = strip_comments(tex)
        graphics_paths = _graphic_paths(tex)
        for environment in FIGURE_RE.findall(tex):
            figure_index += 1
            caption_latex = _balanced_argument(environment, "caption")
            caption_text = latex_to_text(caption_latex)
            label = _balanced_argument(environment, "label")
            includes = INCLUDE_RE.findall(environment)
            for graphic_index, raw_path in enumerate(includes, start=1):
                resolved = resolve_graphic(source_root, tex_file, raw_path, graphics_paths)
                if resolved is None:
                    continue
                figures.append(
                    FigureRef(
                        tex_file=str(tex_file.resolve()),
                        source_path=str(resolved),
                        caption_latex=caption_latex,
                        caption_text=caption_text,
                        label=label,
                        figure_index=figure_index,
                        graphic_index=graphic_index,
                        graphics_in_environment=len(includes),
                    )
                )
    return figures
