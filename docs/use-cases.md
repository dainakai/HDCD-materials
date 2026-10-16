# Choose a workflow

Begin with [environment setup](environment.md) and [data preparation](data-preparation.md). All commands run from the repository root and use a new output directory.

## A. Evaluate the pretrained models on the test seed

```bash
python scripts/reproduce.py --suite terminal --output runs/pretrained-test
```

This runs the three supplied models on complete test seed 260745. It starts from prepared candidate features and ends with scored events, corrected times/positions and one-to-one matching against 2,000 true events. Expected AP: Ours 0.917, depth-free variant 0.901, MinIP-based 2D 0.858. Read `runs/pretrained-test/verification.json`.

The event counts are 26,604, 22,976 and 24,078, respectively, including low-score predictions. They are not counts of confirmed collisions. This workflow does not fit models or rerun particle detection.

## B. Reproduce the paper tables and figures

```bash
python scripts/reproduce_tables.py --output runs/paper-tables
```

Install the [figure environments](figures.md#1-set-up-the-two-plotting-environments), then run:

```bash
python scripts/reproduce_figures.py --output runs/paper-figures --verify-rendered
```

Outputs cover Tables 3–4 and Figures 4, 5, 7, 8, 11, 12, 13, 14 and 15. The [reproducibility guide](paper-validation.md) identifies which measured inputs and model stages each result uses. Figure 9 is outside this workflow.

## C. Recalculate 3D trajectory features

Preparation calculates the entire 566,261-row 3D shortlist from the included test trajectories and fixed candidate geometry. To inspect a small independent calculation afterward:

```bash
python scripts/recompute_static_features.py --max-candidates 500 --output runs/feature-sample
```

This recalculates 53 features and preserves the 24 candidate-generation features. Use zero as the limit for the complete shortlist. [Input and exactness details](data-preparation.md#what-is-calculated).

## D. Reanalyze experimental review and physical measurements

Read `data/paper_figures/review/visual_review_labels.csv` and the measured inputs described in the [figure guide](figures.md). They cover the top 80 candidates in each of 14 sequences. The 1,120 judgments contain 185 confirmed collisions, 896 non-collisions and 39 ambiguous cases.

You can change score thresholds, review budgets or physical calculations and compare results with the included references. The labels describe the reviewed cohort; they do not supply exhaustive experimental recall or images for independent visual relabeling.

## Scope of this distribution

The complete held-out test remains intact. Five-seed training data, nested development evaluation inputs, supplementary seed 260913, full experimental replay data and upstream image-processing weights are not included. Thus full retraining, full experimental replay and hologram-to-track reproduction are outside the supported workflows. Source routines remain available for inspection, but the README does not assume those missing inputs are present.
