# Reconstruction harness

`pptbench-eval` gives each agent an isolated image-only workspace and requires a one-slide
`reconstruction.pptx`. The harness validates native components and approved resources, rejects
unapproved raster near-copies, and renders the candidate with LibreOffice. Raw agent responses
and failed artifacts remain in the run directory.

## Local setup

Use Linux or WSL2 with Python 3.10+, a virtual environment, Bubblewrap (`bwrap`), LibreOffice
(`soffice`), and a native Linux Codex binary. A Windows executable cannot run inside the Linux
isolation environment. Install the repository with `python -m pip install -e .`.

Authenticate Codex normally before running. The harness reads `auth.json` and connection-provider
settings from `CODEX_HOME` (default `~/.codex`). It does not inherit unrelated project instructions,
MCP servers, or local conversation history. Verify authentication with `codex login status`.

Official runs use LibreOffice 25.8. A distribution's other version can check installation and
artifact production locally; record its version and do not mix it into comparable official runs.
Use `--soffice-command /path/to/soffice` to select a specific installation.

## One task

After materializing the frozen task set, run:

```bash
pptbench-eval --config project.toml harness-run \
  --harness single-run --agent codex \
  --model gpt-6-astra --reasoning-effort low \
  --sample task_0004 --timeout 0
```

`--timeout 0` means no agent deadline. The command exits successfully only when its cases complete;
inspect the run summary and artifact-validation result as well. A successful local reconstruction
smoke does not establish a visual judge score.

To download just one task first:

```bash
pptbench-materialize --task-id task_0004 --accept-arxiv-terms --resume
```

Use the same `--sample task_0004` command above; only selected task assets need to exist.
The model sees the normalized reference and approved resources, not the source PDF, paper
identity, captions, or evaluator code.

## Batch runs and evaluation

`pptbench-rollout-batch --help` documents collection and retry orchestration. Keep model artifact
failures as explicit outcomes in the denominator. Execution and rendering recovery must preserve
original model responses; do not call the model again simply to repeat downstream rendering.

The [evaluation guide](evaluation.md) describes judge rounds and deterministic aggregation.

## Troubleshooting

- Missing `bwrap`, `codex`, or `soffice`: install the Linux executable or pass its explicit CLI path.
- Missing task inputs: run `pptbench-materialize` for the selected task IDs, with the output directory
  matching `tasks_root` in your project configuration.
- A download stops: rerun with `--resume`; completed task hashes are verified before reuse.
- A reference or resource hash differs: keep the failing source and error for inspection. The frozen
  hashes must not be changed to accept different upstream bytes.
- Authentication or model access fails: check the same model with your local Codex configuration;
  credentials and provider configuration remain local.
