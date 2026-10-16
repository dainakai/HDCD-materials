# Holographic droplet collision detection — evaluation data, models and code

Materials accompanying “Semi-automated detection of droplet collision-coalescence using digital in-line holography” by Dai Nakai.

Evaluate the three pretrained collision models on the complete held-out test sequence, reproduce Tables 3–4 and Figures 4, 5, 7, 8 and 11–15, and examine the experimental review labels. These workflows use numerical data and run on a CPU. Hologram images are not required.

The download is approximately 503 MB. It includes test seed 260745, with 6,000 frames and 2,000 true collision events, plus numerical measurements for the nine figures. The five training sequences, supplementary benchmark, full experimental trajectory/feature sequence, and HoloD3 measurement weights are outside this distribution. Full model retraining and image-to-track reproduction therefore require inputs beyond this repository.

## Choose a task

| Task | Starting point | Result |
| --- | --- | --- |
| Evaluate pretrained models | [Quick start](#quick-start) | New event predictions and precision–recall metrics for all three methods |
| Check reproducibility | [Tables and figures](docs/paper-validation.md) | Tables 3–4 and nine figures, with reference comparisons |
| Recalculate 3D trajectory features | [Feature preparation](docs/data-preparation.md#what-is-calculated) | All 566,261 fixed test candidates; 77 trajectory features |
| Reanalyze experimental judgments | [Review CSV dictionary](docs/figures.md#5-visual-review-labels) | 1,120 labels and measured values for the physical comparisons |

The [use-case guide](docs/use-cases.md) describes the input boundary of each workflow.

## Quick start

Download and extract the [v1.0.0 ZIP](https://github.com/dainakai/collision-paper-materials/archive/refs/tags/v1.0.0.zip), or clone the release:

```bash
git clone --branch v1.0.0 --depth 1 https://github.com/dainakai/collision-paper-materials.git
cd collision-paper-materials
```

Use Linux and Python 3.12. Run all commands from this directory. Installation needs the package registry; numerical inputs and collision weights are included. [Environment setup](docs/environment.md).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install uv==0.10.6
.venv/bin/python -m uv pip install --python .venv/bin/python \
  -r requirements-core-reference.txt -e .
source .venv/bin/activate
python scripts/check_environment.py --profile core
python scripts/prepare_data.py --profile all --output runs/data-preparation.json
python scripts/verify_release.py --tables
python scripts/reproduce.py --suite terminal --output runs/pretrained-test
```

Preparation calculates the 3D trajectory features and test particle states locally, and assembles the selected input tables. Every completed table is checked against its recorded SHA-256. It reuses verified files after interruption and preserves existing files whose contents differ. Allow 6 GB of additional disk space for prepared files and temporary calculations, separately from the Python environments. Runtime on the verification host was 14.4 minutes. [Preparation details and selective profiles](docs/data-preparation.md).

The evaluation command loads the supplied models and scores the entire test sequence. The directory name `validation/` refers to this independent test sequence; it was excluded from fitting.

| Test metric | Ours | Depth-free variant | MinIP-based 2D |
| --- | ---: | ---: | ---: |
| Average precision (AP) | 0.917 | 0.901 | 0.858 |
| Maximum F1 | 0.878 | 0.846 | 0.811 |
| Precision at recall ≥ 0.8 | 0.960 | 0.895 | 0.813 |
| False positives at that recall | 67 | 188 | 368 |
| Precision at recall ≥ 0.9 | 0.766 | 0.637 | 0.424 |
| False positives at that recall | 550 | 1,024 | 2,447 |

MinIP means minimum-intensity projection of a reconstructed volume. [Method comparison](docs/baseline2d.md#method-comparison).

Read `runs/pretrained-test/verification.json`. The `terminal/` subdirectory contains event predictions and precision–recall curves. AP integrates the interpolated precision envelope. Events match one-to-one within ±5 frames and 350 µm in the image plane. Scores rank candidates; they are not calibrated probabilities. [Evaluation definitions](docs/paper-validation.md#evaluation-definitions).

## Reproduce the tables and figures

```bash
python scripts/reproduce_tables.py --output runs/paper-tables
```

This recalculates Tables 3–4 and writes CSVs, LaTeX tables and a verification report. Table 3 uses reference predictions and candidate geometry; Table 4 reruns the full-feature and relation models from the prepared test tables.

Install the [figure environments](docs/figures.md#1-set-up-the-two-plotting-environments), then run:

```bash
python scripts/reproduce_figures.py --output runs/paper-figures --verify-rendered
```

The command produces nine figures with 18 PDF panels, numerical outputs, English captions and the paper's layout. Exact pixel checks use the specified plotting environments and Poppler 22.02.0. With another Poppler version, omit `--verify-rendered` to retain numerical checks and report rendering differences.

A ready-to-view [figure and table preview](paper/preview.pdf) is included. [Figure guide](docs/figures.md) · [Calculation settings](docs/physical-conventions.md).

## What is calculated and what is supplied

- The original particle simulator regenerates the complete test particle-state table from the fixed seed and sampling distributions.
- The original trajectory routines calculate 53 additional features from supplied 3D test trajectories and 24 fixed candidate-generation features. Small recorded bitwise differences recover the exact original table representation.
- Model scores, relation attention, duplicate suppression, event time/position corrections and evaluation are calculated from the prepared tables and trained weights.
- Image-derived measurements and the two comparison methods' final feature tables are supplied. They are not inferred from the random seed.
- Numerical figure measurements cover all 14 experimental sequences. The [review CSV](data/paper_figures/review/visual_review_labels.csv) contains the top 80 candidates per sequence: 185 collisions, 896 non-collisions and 39 ambiguous cases. These judgments are excluded from training.

[Data dictionary](docs/data.md) · [Model definitions](docs/models.md) · [Feature catalog](docs/features.md) · [Verification scope](docs/reproduction.md).

## Citation and licenses

Please cite the associated paper and the dataset version used. [CITATION.cff](CITATION.cff) contains the citation metadata. Numerical research data use [CC BY 4.0](LICENSE-DATA.md); software and included collision-model weights use MIT, with third-party components retaining their own licenses. [Third-party notices](THIRD_PARTY_NOTICES.md).
