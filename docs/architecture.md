# Pipeline architecture

PPTBench has one shared task contract and three maintained pipeline components:

```text
collection and materialization
  discover -> collect -> inspect -> human screen -> materialize 500 tasks
                                         |
                                         v
reconstruction
  task brief + reference + approved resources -> isolated agent -> reconstruction.pptx
                                         |
                                         v
judging
  normalized reference/candidate -> artifact checks -> semantic gate -> render/text gate -> detail findings x3
                                         |
                                         v
  deterministic consensus scoring -> CSV/JSON release data
```

## Task contract

```text
benchmark/tasks/task_0000/
  metadata.json
  resources/index.json
```

After materialization, the local task additionally contains `source.pdf`, `reference.png`, and
any approved `resources/resource_*.png`. The reference is the model-visible target. The source
PDF is retained locally for provenance and inspection; the public repository contains only the
instructions needed to recover it.

## Isolation

The reconstruction and Judge harnesses create separate writable workspaces for each task. The
agent receives only the task brief, reference, approved resources, and authoring runtime. Other
tasks, source manifests, evaluator implementation, and host paths are not exposed through the
task interface. Generated workspaces and logs remain outside the repository.

## Native artifacts

The harness validates that the output is a readable one-slide PPTX and renders it with the pinned
LibreOffice 25.8 installation. Native editability is enforced by the artifact contract and
component checks. An image-similarity check rejects an unapproved raster that reproduces the
reference; approved irreducible resources are exempt. The benchmark score is based on the rendered
comparison and Judge findings after these artifact checks.
