from __future__ import annotations

from html import unescape
from datetime import date
import re
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

from .http import CachedHttpClient
from .models import Paper


ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


def _text(node: ET.Element, name: str, default: str = "") -> str:
    child = node.find(name)
    if child is None or child.text is None:
        return default
    return " ".join(child.text.split())


def canonical_arxiv_id(value: str) -> str:
    value = value.rsplit("/abs/", 1)[-1]
    return re.sub(r"v\d+$", "", value.strip())


def with_submitted_date_filter(query: str, after: str = "", before: str = "") -> str:
    if not after and not before:
        return query

    def format_boundary(value: str, end_of_day: bool) -> str:
        parsed = date.fromisoformat(value)
        return parsed.strftime("%Y%m%d") + ("2359" if end_of_day else "0000")

    lower = format_boundary(after, False) if after else "000101010000"
    upper = format_boundary(before, True) if before else "999912312359"
    return f"({query}) AND submittedDate:[{lower} TO {upper}]"


def parse_atom(payload: bytes, query: str = "") -> list[Paper]:
    root = ET.fromstring(payload)
    papers: list[Paper] = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = _text(entry, f"{ATOM}id")
        versioned_id = raw_id.rsplit("/abs/", 1)[-1].strip()
        arxiv_id = canonical_arxiv_id(raw_id)
        if not arxiv_id:
            continue

        links = {
            link.attrib.get("title", link.attrib.get("rel", "")): link.attrib.get("href", "")
            for link in entry.findall(f"{ATOM}link")
        }
        categories = [
            category.attrib.get("term", "")
            for category in entry.findall(f"{ATOM}category")
            if category.attrib.get("term")
        ]
        primary = entry.find(f"{ARXIV}primary_category")
        paper = Paper(
            arxiv_id=arxiv_id,
            title=_text(entry, f"{ATOM}title"),
            abstract=_text(entry, f"{ATOM}summary"),
            authors=[
                _text(author, f"{ATOM}name") for author in entry.findall(f"{ATOM}author")
            ],
            categories=categories,
            primary_category=primary.attrib.get("term", "") if primary is not None else "",
            published=_text(entry, f"{ATOM}published"),
            updated=_text(entry, f"{ATOM}updated"),
            abs_url=links.get("alternate", f"https://arxiv.org/abs/{arxiv_id}"),
            pdf_url=links.get("pdf", f"https://arxiv.org/pdf/{arxiv_id}"),
            source_url=f"https://arxiv.org/src/{versioned_id}",
            doi=_text(entry, f"{ARXIV}doi"),
            journal_ref=_text(entry, f"{ARXIV}journal_ref"),
            query=query,
            versioned_id=versioned_id,
        )
        papers.append(paper)
    return papers


def search_papers(
    client: CachedHttpClient,
    query: str,
    limit: int,
    page_size: int,
    query_number: int,
) -> list[Paper]:
    results: list[Paper] = []
    for start in range(0, limit, page_size):
        params = urlencode(
            {
                "search_query": query,
                "start": start,
                "max_results": min(page_size, limit - start),
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        url = f"https://export.arxiv.org/api/query?{params}"
        payload = client.get(url, f"query-{query_number:02d}-{start:05d}", ".xml")
        page = parse_atom(payload, query)
        results.extend(page)
        if len(page) < min(page_size, limit - start):
            break
    return results


def fetch_license(client: CachedHttpClient, paper: Paper) -> str:
    """Read the per-version license from OAI metadata, with an abs-page fallback."""
    safe_id = paper.arxiv_id.replace("/", "_")
    params = urlencode(
        {
            "verb": "GetRecord",
            "identifier": f"oai:arXiv.org:{paper.arxiv_id}",
            "metadataPrefix": "arXivRaw",
        }
    )
    try:
        payload = client.get(
            f"https://export.arxiv.org/oai2?{params}", f"license-{safe_id}", ".xml"
        )
        root = ET.fromstring(payload)
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1].lower() == "license" and element.text:
                return element.text.strip()
    except (ET.ParseError, OSError):
        pass

    try:
        html = client.get(paper.abs_url, f"abs-{safe_id}", ".html").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        return ""
    match = re.search(
        r'<div[^>]+class="[^"]*abs-license[^"]*"[^>]*>.*?href="([^"]+)"',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return unescape(match.group(1)) if match else ""
