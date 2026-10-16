# Reproduce Figures 4, 5, 7, 8 and 11–15

[Repository home](../README.md) · [Reproducibility guide](paper-validation.md) · [Terminology](glossary.md)


This workflow recalculates nine figures from the bundled numerical inputs and produces 18 PDF panels with the paper's design. It needs no holograms, external datasets or model training. Figure 9 is outside this workflow.

To inspect the figures before installing software, open the [figure and table preview](../paper/preview.pdf). The commands below regenerate the numerical results, plots and captions. Original figure dimensions and rendering hashes are included for comparison.

## 1. Set up the two plotting environments

Set up the [core environment](environment.md#core-environment) first, then run these commands from the archive root.
The original figures used two Matplotlib versions, so exact reproduction uses two environments.
The second environment is only for plotting; it does not need PyTorch, a GPU, or separate data.

```bash
.venv/bin/python -m uv pip install --python .venv/bin/python \
  -r requirements-figures-reference.txt -e .

.venv/bin/python -m venv .venv-figures311
.venv/bin/python -m uv pip install --python .venv-figures311/bin/python \
  -r requirements-figures311-reference.txt

source .venv/bin/activate
python scripts/check_environment.py --profile core
```

Install Poppler for PDF-to-PNG rendering; on Ubuntu/Debian, use `sudo apt-get install poppler-utils`.
`pdftoppm -v` should work before running the figure command.
The strict reference hashes use Poppler 22.02.0 on Linux.
Both required Liberation font revisions are bundled and selected explicitly, independently of the system's installed fonts.

| Environment | Matplotlib | Panels |
| --- | --- | --- |
| `.venv` | 3.10.9 | 4, 5, 7(a–d), 11, 13, 14, 15 |
| `.venv-figures311` | 3.11.1 | 7(e,f), 8, 12 |

## 2. Generate all nine figures

```bash
python scripts/reproduce_figures.py --output runs/paper-figures --verify-rendered
```

All input tables, models and reference values stay unchanged.
The script requires a new or empty output directory.
It runs on CPU and needs no images or network access after environment setup.
Figure 7 evaluates all 6,000 frames, including 1,473,698 low-confidence particle detections and 1,225,299 track points; allow several minutes and several GB of free memory.
Other figures use smaller tables.

The default `saved-figure` convention reproduces Figures 14–15 using the paper plots' exposure and uncertainty settings. The expected result is `PASS`, with all 18 panel page sizes and rendered-pixel hashes matching. [Calculation settings](physical-conventions.md) define the rates, resampling seed and optional alternatives.

If the environments or Poppler executable are elsewhere:

```bash
python scripts/reproduce_figures.py --output runs/paper-figures-custom \
  --plot-python /path/to/figures311/bin/python --pdftoppm /usr/bin/pdftoppm
```

Generate only selected figures with:

```bash
python scripts/reproduce_figures.py --figures 8 11 12 \
  --output runs/selected-figures --verify-rendered
```

## 3. Read the outputs

| Output | Purpose |
| --- | --- |
| `index.html` | Gallery of the nine figures with PDF links |
| `figure04.pdf`, `figure05.pdf`, …, `figure15.pdf` | Standalone vector figure previews; the multipanel figures follow the paper's ordering |
| `figure04.png`, … | Convenient previews of those PDFs |
| `figureXX/<original-paper-filename>.pdf` | The 18 computed assets, using the original filenames and dimensions; Figures 14–15 depend on the chosen convention |
| `figureXX/*.csv`, `*.parquet` | Recomputed curves, counts, distributions and errors |
| `paper-assets/figures/...` | The same assets arranged at the paths used by the paper's LaTeX |
| `paper-assets/figures.tex` | The paper's English captions, figure layout and numbering |
| `numerical_verification.json` | Calculation convention, numerical checks and whether reference values match |
| `rendering_verification.json` | Per-asset geometry and rendered-pixel comparison |
| `verification.json` | Combined numerical and rendering result |
| `figure14/paper_reference_comparison.json`, `figure15/paper_reference_comparison.json` | Differences in counts, rates, uncertainty and exposure from the paper's saved values |

Open `runs/paper-figures/index.html` in a browser.
To typeset the captions and layout, compile `figures.tex` from inside the generated `paper-assets/` directory with LuaLaTeX and the packages listed in its preamble. The [preview instructions](../paper/README.md#regenerate) combine all nine figures with Tables 3–4.
The Python workflow does not require TeX; its combined PDFs are previews rather than full typeset paper pages.

## 4. What is recalculated for each figure

Paths in the input column are relative to `data/paper_figures/`, unless stated otherwise.
The 14-scene experimental inputs are numerical summaries and measured windows. Complete experimental tracks and collision feature tables are outside this distribution.

| Figure | Input | Computation and expected result |
| --- | --- | --- |
| 4 | `fig04/diameter_by_scene.csv`, `frame_metrics.parquet` | Mean number-density curve across 14 scenes; 50,000 bootstrap resamples, seed 260715; per-scene particle count and liquid water content (LWC) means and sample standard deviations (SDs) (`ddof=1`) |
| 5 | `fig05/velocity_tracks.parquet` | Diameter/velocity hexbin distributions of per-track summaries from all 14 scenes; diameter 25–300 µm, at least five valid points; Beard terminal velocity minus 0.26 m/s |
| 7(a) | `fig07/low_confidence_detections.parquet` and `data/labels/validation/seed260745/particles.parquet` | Confidence-ranked greedy matching within 8 pixels in each frame; 1,200,000 in-volume truth points; particle AP 0.975 |
| 7(b–d) | `fig07/paired_tracks_20mm.parquet` and the same truth table | Framewise one-to-one assignment minimizing distance on raw XY (the Hungarian algorithm), then errors from smoothed XYZ/diameter; 1,160,051 matched points |
| 7(e,f) | `fig07/number_density_by_diameter_5um.csv`, `truth_trajectory_summary.csv`, `measured_trajectory_summary.csv` | Frozen original density/trajectory summaries; 25 ≤ diameter < 300 µm; cumulative counts use 8,275 truth and 20,297 measured trajectories |
| 8 | `data/predictions/{3d,2d,minip2d}/validation/seed260745.parquet` and collision truth | Recompute every precision–recall point with maximum one-to-one projected-XY matching; AP 0.917 / 0.901 / 0.858; 67 / 188 / 368 false positives at target recall 0.8 (Ours / Depth-free variant / MinIP-based 2D) |
| 11 | `review/visual_review_labels.csv` | Single histogram over score 0.98–1.00 from `collision_score`; 185 collisions and 896 non-collisions, excluding 39 ambiguous cases |
| 12 | Same review CSV | Ranked top-score cohort: 80 candidates × 14 scenes; at top 30 per scene, 177/185 = 95.7% confirmed-event coverage and 177/420 = 42.1% precision |
| 13 | Review CSV and `fig13/window_measurements.parquet` | Recompute volume-conservation diameter and momentum-conservation vertical velocity from measured pre-event values; 185 confirmed and 877 rejected diameter cases; 180 confirmed and 803 rejected valid velocity cases |
| 14 | Review CSV, `gravity/fine_density_by_scene_5um.csv`, `time_block_density_by_scene_5um.csv`, bundled physical tables | 185 confirmed events versus 210.1 ± 14.6 predicted events across 14 scenes; density resampling and counting SD with the plotted exposure |
| 15 | Same as Figure 14 | Observed and predicted counts in hard 25 µm diameter bins; 66 unordered bin pairs; the same total counts as Figure 14 |

Figure 8(c) uses absolute false-positive counts.
For review denominators, training association and event-policy definitions, read [analysis conventions](paper-notes.md).

### Figure 7 processing inputs

Figure 7(b–d) uses `paired_tracks_20mm.parquet`, with 74,900 µm front reconstruction distance, 3 phase iterations, 1,536-pixel padding and fixed linked-track membership. These paired raw/smoothed measurements are included specifically for the error panels. Figure 7(e,f) uses the supplied benchmark's saved density and trajectory summaries. [Processing conditions](benchmark-reconstruction.md) identify the input for each figure and the collision benchmark.

### Gravitational prediction and randomness

Figures 14–15 use a 5 µm density/kernel grid and a 25 µm display grid. The default calculation uses 10,904 frames per scene, 4,000 Hz, and 10.71644672 cm³ per frame. Its density uncertainty uses 11 temporal blocks, 4,000 resamples, and seed 260724. Count SD is `sqrt(expected count + density bootstrap variance)`.

The optional `poisson-only` mode keeps this density/exposure and uses `sqrt(expected count)`. The optional `manuscript` mode uses all 10,917 frames and the nominal volume, with Poisson-only SD. Both change Figures 14–15; use the default for exact reproduction. [Detailed settings](physical-conventions.md).

The kernel is evaluated at fine-bin centers before aggregation, including half-weight for unordered pairs within the same coarse bin. Every mode retains the hard 25 µm diameter bins.

## 5. Visual-review labels

The paper-cohort CSV is [visual_review_labels.csv](../data/paper_figures/review/visual_review_labels.csv): 1,120 reviewed candidates, 80 per scene, from 14 experimental sequences.
There are 185 `collision`, 896 `non_collision`, and 39 `ambiguous` labels.
The original decimal score strings are retained; media paths, reviewer identifiers and review timestamps are omitted.
These labels are observations from selected review candidates, not exhaustive collision truth. The CSV supports reanalysis of those judgments; independent visual reassessment needs the omitted experimental images.
They are excluded from collision-model training.

| Columns | Meaning |
| --- | --- |
| `scene_num`, `source_date` | Raw acquisition scene number and date code 260715 |
| `event_id` | Review event identifier within a scene; not a simulator collision-event ID |
| `vanished_track_id`, `survivor_track_id` | Measured track identities within the scene |
| `candidate_frame_raw` | Candidate frame before residual correction; join key with the two track IDs and scene |
| `candidate_frame`, `frame` | Saved corrected/review event frame; frames are zero-based at 4,000 Hz |
| `status`, `binary_label` | Three-way judgment; binary value is 1/0 for collision/non-collision and missing for ambiguous |
| `collision_score`, `score` | Original review-display score and exported ranking score; Figure 11 uses `collision_score` |
| `review_score_rank`, `review_stratum` | Rank 1–80 within each scene; every released review has stratum `top_score` |
| `review_inclusion_probability`, `review_inverse_probability_weight` | Both equal 1 for this complete top-80 cohort; neither is a collision probability |
| `pre_*`, `post_*`, `pixel_pitch_um` | Measured diameters, number of samples and window boundaries used by the physical analyses |
| `training_eligible` | Legacy field name indicating an unambiguous binary review; it does not authorize training on the experimental reviews |

Each scene contributes exactly 80 `top_score` reviews. [Counts by scene](../data/paper_figures/review/review_counts_by_scene.csv) and [selection provenance](../provenance/review_cohort.json) identify the cohort.
Ambiguous cases are excluded from the binary histograms and physical comparisons, but remain part of the review-budget denominator in Figure 12 when they occur in the top-score cohort.
Coverage in Figure 12 is relative to the 185 confirmed reviewed events; it is not absolute recall over all physical collisions.
Filter `scene_num` in this CSV to inspect one sequence.

Paper scene indices 1–14 correspond, in order, to raw scene numbers:
`12, 13, 15, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30`.
The mapping is implemented in `scripts/figure_support/scene_index.py`.

## 6. Verification and other use cases

The [figure verification record](../provenance/reduced_distribution_verification.json) contains all nine calculations and the 18 panel comparisons with the paper figures.
PDF file bytes can differ because of creation timestamps and metadata.
Other Poppler/platform versions may render font edges differently; omit `--verify-rendered` to run the same numerical checks and record the rendering comparison without requiring exact pixel hashes.

The plotting workflow starts from frozen predictions and measured numerical summaries.
To rerun the trained collision weights on the test seed, use `python scripts/reproduce.py --suite terminal --output runs/terminal`.
For the supported evaluation, 3D trajectory-feature and review workflows, follow the [use-case guide](use-cases.md).
Run outputs are written under `runs/` and excluded from the release manifest. The compact generated plot PDFs in `paper/` are included for immediate inspection.
