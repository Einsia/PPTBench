# Source data license audit

Audit date: `2026-08-17`

This is a reproducible provenance and license check, not legal advice. The audit uses the
license attached to the exact arXiv version recorded by each frozen task. It does not infer
reuse rights from free access, a DOI, or a publisher's open-access label.

## Release decision

**BUNDLED-ASSET RELEASE: BLOCKED.** Use the downloader-only release; do not commit paper pixels.

The public PPTBench package is downloader-only: it ships task specifications and hashes,
not source PDFs, reference renderings, or raster resource payloads.

- Tasks audited: **500**
- Unique source papers: **500**
- Release-ready: **211**
- Manual review required: **17**
- Assets to exclude or replace with download instructions: **272**

## Detected licenses

| License family | Tasks |
|---|---:|
| `arxiv-nonexclusive` | 226 |
| `cc-by` | 197 |
| `cc-by-nc-nd` | 46 |
| `cc-by-nc-sa` | 17 |
| `cc-by-sa` | 13 |
| `cc0` | 1 |

## Policy

- `release-ready`: CC BY, CC BY-SA, CC0, or a public-domain mark. Required
  attribution and ShareAlike conditions still apply per task.
- `review-required`: unknown, unrecognized, or non-commercial licenses. These are not
  included in an unrestricted public release without separate review/permission.
- `exclude-assets`: arXiv non-exclusive or NoDerivatives licenses. Keep only provenance
  metadata and provide a deterministic downloader, unless separate permission is recorded.

The full per-task evidence, title, authors, exact arXiv version, license URL, conditions,
and decision are in `benchmark/source_license_audit.jsonl`. The machine-readable summary
is in `benchmark/source_license_audit_summary.json`.

## Authoritative references

- [arXiv license information](https://info.arxiv.org/help/license/index.html)
- [arXiv permissions and reuse FAQ](https://info.arxiv.org/help/license/reuse.html)
- [arXiv API terms and rate limits](https://info.arxiv.org/help/api/tou.html)

arXiv states that its default non-exclusive license grants distribution rights to arXiv
and does not itself grant third parties permission to mirror e-prints or figures.
