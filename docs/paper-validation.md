# Check reproducibility

[Repository home](../README.md) · [Terminology](glossary.md) · [Analysis conventions](paper-notes.md)

This guide maps the paper's figures, tables and methods to the supplied data and reproduction commands.
The main entry points use included tables and weights; no images, data account or GPU are required.
Figures containing acquisition/reconstructed images have a different scope, listed below.

## 1. Verify the files and evaluate the trained models

Follow [core environment setup](environment.md#core-environment), then run from the release root:

```bash
python scripts/prepare_data.py --profile all --output runs/data-preparation.json
python scripts/verify_release.py --tables
python scripts/reproduce.py --suite terminal --output runs/reproducibility-test
```

Data preparation reconstructs the original numerical files and checks their SHA-256 hashes.
The verification command checks both the compact inventory and all prepared table schemas.
The final command runs all three pretrained models on the held-out synthetic test seed, 260745.
Expected AP is 0.917 for Ours, 0.901 for the depth-free variant and 0.858 for MinIP-based 2D. [Processing conditions](benchmark-reconstruction.md) list the reconstruction and stage-specific settings.
Read `runs/reproducibility-test/verification.json`; predictions and PR curves are in its `terminal/` subdirectory.
The report includes the measured numerical differences, not just a pass/fail flag.

## 2. Recalculate the performance and stage-comparison tables

```bash
python scripts/reproduce_tables.py --output runs/reproducibility-tables
```

Outputs are `table3.csv`, `table4.csv`, the paper-layout `table3.tex` and `table4.tex`, the underlying PR curves, four stage event tables, and `verification.json`. Every displayed LaTeX value is checked against the paper's tables. Table 3 differences in parentheses are method minus Ours, calculated from displayed three-decimal values; the CSVs retain full precision.
Use `--tables 3` or `--tables 4` to run one table.
Allow several minutes and a few GB of memory on a CPU; Table 4 runs tree/attention inference on the complete 260,123-row scoring set.

| Table 3 metric | Ours | Depth-free variant | MinIP-based 2D |
| --- | ---: | ---: | ---: |
| Average precision (AP) | 0.917 | 0.901 | 0.858 |
| Maximum F1 | 0.878 | 0.846 | 0.811 |
| Precision at recall ≥ 0.8 | 0.960 | 0.895 | 0.813 |
| False positives at that recall | 67 | 188 | 368 |
| Precision at recall ≥ 0.9 | 0.766 | 0.637 | 0.424 |
| False positives at that recall | 550 | 1,024 | 2,447 |
| Candidate-pool recall | 0.985 | 0.990 | 0.991 |

Candidate coverage uses the generated shortlist before scoring and suppression, denoted H in the paper.
Each shortlist contains 566,261 candidates.
It is not the larger permissive pool or the final predicted-event set.
The exact recalls are 0.9845, 0.9895 and 0.9905, displayed as 0.985, 0.990 and 0.991 for Ours, the depth-free variant and MinIP-based 2D.

| Table 4 stage | AP | Precision at recall ≥ 0.8 | FP | Precision at recall ≥ 0.9 | FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trajectory features | 0.879 | 0.857 | 266 | 0.532 | 1,583 |
| + image features | 0.888 | 0.881 | 216 | 0.603 | 1,186 |
| + relation attention | 0.896 | 0.917 | 144 | 0.668 | 896 |
| + coordinate duplicate suppression | 0.917 | 0.960 | 67 | 0.766 | 550 |

What is recomputed: Table 3 starts from the saved final event predictions and candidate geometry, then repeats one-to-one matching.
Table 4 starts from the saved screening/trajectory scores, runs the bundled full-feature and relation models, and recalculates event selection, residual correction and matching.
The full-feature scores are also checked against the original saved array.
Reference CSVs under `data/paper_tables/` are used only for assertions.

How to interpret the stage comparison: Every row uses temporal-mode aggregation, at most five modes per terminating track, a track-rank penalty and the same fitted localization regressors.
Only the last row enables coordinate-based suppression.
Thus the first row is trajectory-only scoring, not removal of all other pipeline operations; even its event correction uses the shared regressors with unavailable image columns left missing.
These are the original comparison conditions, documented in [the algorithm guide](algorithm.md).

## 3. Recalculate the figures and experimental statistics

Install the [figure dependencies](figures.md#1-set-up-the-two-plotting-environments), then run:

```bash
python scripts/reproduce_figures.py --output runs/reproducibility-figures --verify-rendered
```

Read `index.html`, `numerical_verification.json` and `rendering_verification.json` in that output directory.
The script recomputes all nine numerical figures and compares all 18 PDF panels with the paper figures. Strict pixel comparison uses Poppler 22.02.0 and the specified plotting environments. With another Poppler version, omit `--verify-rendered` to run the numerical checks and record rendering differences without treating them as an error. See [count and exposure settings](physical-conventions.md).

| Paper result | Released input and implementation | What can be checked |
| --- | --- | --- |
| Section 2: conservation and gravitational kernel | [Physical diagnostics](physics.md), `src/ra3d/config/physics/` | Diameter/momentum predictions, efficiency interpolation and pair counts |
| Table 1: acquisition settings | [Units and calibration](data.md#units-and-identifiers), acquisition YAML files | Stored metadata and [processing conditions](benchmark-reconstruction.md) |
| Table 2 and Appendix A: algorithm/settings/features | [Algorithm](algorithm.md), [92-feature catalog](features.md), ordered schemas in `provenance/` | Exact feature definitions, model settings, candidate allocation and training splits |
| Figures 4–5: experimental population/velocity | `data/paper_figures/fig04/` and `fig05/` | Resample scene summaries and plot per-track measurements from all 14 sequences |
| Figure 7: synthetic particle accuracy | `fig07/`, complete test particle truth | Rematch detections and paired raw/smoothed positions; plot frozen density/length summaries |
| Table 3 and Figure 8: collision accuracy | Saved predictions, candidates and true events for seed 260745 | Event matching, AP, F1, target-recall precision and false positives |
| Table 4: cumulative stage comparison | Saved trajectory scores and trained models | Stage-specific inference, selection, correction and evaluation |
| Figures 11–12: experimental review results | [1,120-row top-80 review CSV](../data/paper_figures/review/visual_review_labels.csv) | Counts, score distributions and confirmed-event coverage at a review budget; all scenes use ranks 1–80 |
| Figure 13: physical consistency | `fig13/window_measurements.parquet` and review CSV | Recompute conservation predictions from measured windows |
| Figures 14–15: counts/rates by scene and diameter pair | `gravity/` density tables, physical kernel and review CSV | Recompute counts, rates, density resampling and count SD using the plotted exposure; compare with saved plots |
| Figures 1–3 and 6 | Method/apparatus/architecture illustrations | Explained by the manuscript; not numerical reproduction targets |
| Figures 9–10 | Image-based case studies | Original images are excluded. These panels and independent visual relabeling cannot be regenerated from the numerical release alone. |

The full figure paths, filters, random seeds and CSV field dictionary are in [the figure guide](figures.md).
For Figures 4–5 and 13–15, the numerical summaries/measurements cover all 14 experimental sequences; full experimental tracks and collision feature tables are outside this distribution.
Recomputing all 14 sequences from raw recordings is outside the image-free release.

## Evaluation definitions

For a score threshold, let M be the maximum number of one-to-one matches between P accepted predictions and T true events.
Then `precision = M/P`, `recall = M/T` and `F1 = 2*precision*recall/(precision+recall)`.
A match requires an integer frame difference at most 5 and an image-plane distance at most 350 µm; no axial or particle-ID condition is used for test evaluation.
Matching never crosses sequence boundaries.
When competing predictions can match more than one truth, earlier assignments may be changed to maximize the match count.

Equal score values enter the curve as one group.
At target recall r, the operating point is the highest threshold achieving `recall >= r`; precision and FP counts come from the same threshold.
This release's operating-point columns consistently use the first threshold reaching the target, as in Table 4.

For curve points ordered by decreasing score, define `envelope[i] = max(precision[j] for j >= i)`.
The reported AP is `sum((recall[i] - recall[i-1]) * envelope[i])`, with initial recall zero.
Unreached recall contributes zero.
This is interpolated event AP; it is not a trapezoidal area, candidate-label classification accuracy, or a direct call to scikit-learn's non-interpolated `average_precision_score`.
The evaluator is [evaluation.py](../src/ra3d/evaluation.py).

Experimental coverage has a different denominator: the 185 visually confirmed events among selected review candidates.
At 30 reviews per sequence, `177/185 = 95.7%` coverage and `177/420 = 42.1%` precision.
The 420 reviews include 18 ambiguous judgments; removing them would change the denominator.
These data do not establish recall over all physical experimental collisions.

## 4. Inspect the trajectory calculation and method

The preparation command calculates the full test 3D trajectory-feature table from the included trajectories and fixed candidate geometry. [Preparation details](data-preparation.md#what-is-calculated) explain the inputs and exact-file verification. The [feature catalog](features.md), [algorithm](algorithm.md) and [analysis conventions](paper-notes.md) describe the scientific calculations.

Training, supplementary evaluation, full experimental replay and image-to-track workflows require inputs outside this reduced distribution. [Verified scope](reproduction.md).
