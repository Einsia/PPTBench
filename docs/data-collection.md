# Data collection and task materialization

`pptbench-collect` turns arXiv figure candidates into reproducible PPTBench task specifications.

The collection pipeline queries arXiv, downloads source archives, resolves LaTeX figure
references, locates the figure asset, and records provenance. It renders each resolved single-page
figure PDF, inspects native raster image blocks, and lets human reviewers approve irreducible
assets that an agent must embed unchanged. Vector paths, text, nodes, and connectors remain part
of the diagram rather than being extracted as resources. Paper-level deduplication removes
repeated figures and alternate exports of the same figure. Human screening then removes figures
that are not process or architecture diagrams, are too simple to be discriminative, are too
illegible to judge, or have ambiguous structure. The screened set is the frozen 500-task benchmark.

Run the collection stages explicitly. The configuration option is global and must appear before
the subcommand:

```bash
pptbench-collect --config configs/collection.example.toml discover
pptbench-collect --config configs/collection.example.toml collect
pptbench-collect --config configs/collection.example.toml stats
```

After human review, finalize the accepted candidate manifest (add
`--include-unknown-license` only for a local research run):

```bash
pptbench-collect --config configs/collection.example.toml finalize --take 500
```

For a local candidate manifest, use `pptbench-materialize-candidates`. It copies the candidate PDFs and
reference renders into a writable task tree; `--manifest` is the manifest-file option:

```bash
pptbench-materialize-candidates \
  --manifest /path/to/accepted-candidates.jsonl \
  --dataset-root /path/to/collection \
  --output-root /path/to/tasks
```

For the public downloader-only frozen set, use `pptbench-materialize` instead. It downloads the
recorded source versions and materializes pixels locally after the arXiv terms have been accepted:

```bash
pptbench-materialize \
  --tasks-root benchmark/tasks \
  --manifest benchmark/materialization_manifest.jsonl \
  --license-audit benchmark/source_license_audit.jsonl \
  --task-file configs/taskset.txt \
  --output-root data/pptbench-tasks \
  --accept-arxiv-terms
```

The public release keeps metadata, resource indexes, source addresses, and license evidence. It
does not commit paper PDFs, reference pixels, or extracted resource bytes.
