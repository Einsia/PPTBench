# Dataset licensing and provenance

This file separates rights in PPTBench-authored material from rights in third-party scientific
figures. It is a project compliance record, not legal advice.

## PPTBench-authored material

Selection annotations, task-specific structural statistics, judge findings and benchmark scores
created by PPTBench are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Descriptive arXiv metadata such as titles, authors, abstracts, identifiers and categories is made
available by arXiv under CC0. Code is covered separately by the repository's MIT `LICENSE`.

## Downloader-only distribution

The public repository does **not** contain `source.pdf`, `reference.png`, or raster resource
payloads from papers. It contains only provenance, hashes, rendering/crop specifications, task
selection annotations and evaluation results. Users may run `pptbench-materialize` to retrieve the
exact arXiv source version and construct task files locally. PPTBench does not relicense the
downloaded content; copyright and reuse terms remain with the relevant authors or publishers.

Free access on arXiv is not, by itself, permission for PPTBench to redistribute an e-print or its
figures. arXiv's [reuse FAQ](https://info.arxiv.org/help/license/reuse.html) states that reuse depends
on the version's license; the default arXiv non-exclusive license grants distribution rights to
arXiv rather than to downstream mirrors. arXiv's
[API terms](https://info.arxiv.org/help/api/tou.html) likewise prohibit storing and serving e-prints
without permission from the copyright holder or an applicable license grant.

Running the downloader requires an explicit `--accept-arxiv-terms` acknowledgement, uses one
connection with at least three seconds between requests, and records the applicable license row in
each local materialization report. Materialized output is Git-ignored and must not be republished
unless the user has an applicable license grant or separate permission.

## Reproducible audit

Run the version-specific audit from the repository root:

```bash
pptbench-license-audit
```

The command obeys arXiv's single-connection, one-request-per-three-seconds limit and writes:

- `benchmark/source_license_audit.jsonl`: per-task title, authors, exact version, license evidence,
  obligations and release decision;
- `benchmark/source_license_audit_summary.json`: machine-readable counts and policy;
- `SOURCE_LICENSE_AUDIT.md`: human-readable release verdict.

The audit is a sidecar rather than a modification of each `metadata.json`, so frozen task hashes do
not change. Do not infer a Creative Commons grant from an empty field.

If a future release bundles assets, they are accepted automatically only under CC BY, CC BY-SA,
CC0, or a recognized public-domain mark.
Attribution and ShareAlike conditions still apply. Non-commercial or unrecognized licenses require
manual review. Assets under the arXiv non-exclusive license or a NoDerivatives license must be
removed and replaced by provenance metadata plus deterministic download/materialization steps,
unless separate permission from the rights holder is documented.
