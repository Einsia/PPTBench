# Evaluation and deterministic scoring

Every candidate is judged three times independently by GPT-5.6 Luna High. The Judge sees a
normalized reference/candidate comparison and the two original renderings. Normalization removes
only outer-canvas whitespace and preserves the aspect ratio; corresponding diagram regions are
compared at a common scale. Whitespace inside the figure remains part of the visual judgment.

## Stages

1. **Artifact gate.** A missing, unreadable, or unrenderable PPTX receives a zero score. The
   validator also rejects an unapproved raster image whose normalized visual similarity to the
   reference indicates that the slide is being used as a screenshot. Approved irreducible raster
   resources are exempt from this check.
2. **Semantic gate.** The Judge checks node existence, connector attachment, direction,
   directedness, and process meaning.
3. **Render/text gate.** The Judge checks widespread rendering collapse and severe text overflow,
   overlap, or broken wrapping.
4. **Detail findings.** If both gates pass, the Judge records residual visual differences in
   layout and composition, text and typography, and local graphics and nodes. Findings contain a
   canonical issue type, reference/candidate description, affected count, and affected fraction;
   the Judge does not assign points.

The Judge may write scripts in its isolated workspace to crop, segment, enlarge, or measure dense
regions. This is especially useful for repeated grids, small icons, arrows, and text alignment.

## Consensus and score

Any hard gate in at least two of the three rounds makes the case score zero. Otherwise, findings
from non-gated rounds are united by `(dimension, category, issue_type)` and the largest affected
fraction is retained. Counts are diagnostic only. The deterministic scorer applies the fixed
severity points and affected-fraction staircase independently within:

| Dimension | Maximum |
|---|---:|
| Layout and composition | 30 |
| Text and typography | 40 |
| Local graphics and nodes | 30 |

Each dimension is clipped at zero, and the total is the sum of the three dimension scores. Shadow
differences and outer-canvas whitespace are diagnostic or excluded unless they change a scored
element's appearance. No nonlinear post-processing or leaderboard-dependent calibration is used.

The executable policy is implemented in `src/vlm_judge_harness/issue_ranker.py` and
`src/vlm_judge_harness/consensus_ranker.py`; the prompts are under `configs/vlm_judge/`.
