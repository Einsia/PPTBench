# Maintainer commands

The public package keeps the collection, reconstruction, and judging paths in
one namespace.  Install the optional dependencies needed for release curation
with:

```bash
python -m pip install -e ".[maintainer]"
```

The maintainer group adds OpenCV for exact native-image crop localization in
the materialization-manifest builder.  `lxml` is part of the base install
because the PPTX parser uses it for DrawingML and package validation.

## Build the downloader manifest

`scripts/data/build_materialization_manifest.py` is run against a private,
fully materialized task tree.  It writes provenance and resource addresses,
not paper PDFs or image bytes:

```bash
python scripts/data/build_materialization_manifest.py \
  --tasks-root /private/pptbench/tasks \
  --output benchmark/materialization_manifest.jsonl
```

The builder expects the 500 frozen `task_*/` directories to contain their
source PDF, reference render, metadata, and any approved resources.  Do not
point it at an unreviewed candidate directory.

## Normalize a deck's bounds

`pptbench-normalize-bounds` adjusts only out-of-canvas DrawingML offsets and
dimensions.  It edits the supplied files in place:

```bash
pptbench-normalize-bounds /path/to/reconstruction.pptx
```

Use `python -m eval_harness.tools.normalize_pptx_bounds` when the package has
not been installed.  Keep a copy of the original deck if it is needed for an
audit.

## Rendering dependency

PPTBench renders every candidate with LibreOffice 25.8.  LibreOffice is an
external system dependency rather than a pip package; install that release
line on the evaluation host and pass its `soffice` executable with
`--soffice-command` when it is not on `PATH`.  The same executable must be
used for all candidates in a result set.

The `scripts/vlm_judge/` directory contains optional analysis helpers.  The
production judging commands are the installed `pptbench-vlm-*` entry points;
the helpers do not define a second scoring policy.

## Release checks

```bash
python -m unittest discover -s tests -v
python scripts/release/verify_release.py
```

Before publishing a task manifest, materialize it into an empty output directory with an empty
source cache, then rerun with `--offline --resume` to recheck all saved hashes. The manifest
builder compares every RGBA channel, including RGB under transparent pixels; matching alpha
alone cannot establish an exact crop. Source members with identical frozen byte hashes are
interchangeable even when an upstream archive contains several copies.
