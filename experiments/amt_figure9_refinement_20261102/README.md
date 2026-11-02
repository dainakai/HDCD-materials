# Fig.9 refinement, 2026-11-02

This is an author-requested revision of illustrative cases and presentation.
It does not change training, candidate scores, thresholds or performance results.
The fixed test sequence is seed 260745 from the 102.3 mm, phase-retrieval-3 campaign.
The original manuscript is not edited.

## Original figure provenance

The existing `../CollisionPaper` clone was fetched and checked at
`2a5d3f4e4b4723aab782c2f42b473d5b5e3bbd92`, matching `origin/main`.
Its original Fig.9 is `本番フォーマット/figures/fig09.pdf`.
The plot script was recovered from
`/media/dai-server/DATAM2/siderepos/260804/CollisionPaper/scripts/plot_image_feature_case_study.py`.
It is byte-identical to the archived
`paper-range-only-20260930/sources/paper_style/image_feature_case_study.py`.
The unmodified source is frozen at `source/original_plot.py` (SHA-256
`b4b1c09f1e34ea694ee599eebc984318b3581c4a0a51902cc500cca928640a4f`).
`source/prepare_figures.py` is the current manuscript's submission relabeler.

The original design uses 128×128 crops, red circles and blue squares with short
track tails, two rows of three images, and horizontal score bars at right.
The same Matplotlib 3.10.9 and Liberation Sans plot font are used here. The
submission relabeler's documented Arial-compatible Liberation Sans fallback is
used for panel captions (the original submission PDF embeds ArialMT there).
All visible text fonts are embedded; the relabeler's default Helvetica resource
is unused. Original image
panel positions are identical within 0.01 pt. The tight PDF bounding box grows
3.16 pt at right because the new larger score label extends farther right.
Image panels, axes and label centres are not repositioned to accommodate this.

## Selection and display

`search.py` checks the full saved scoring union and truth association. It shortlists
30 true events from 60 eligible candidates, plus one shared-particle artifact
surviving the visibility, actual-observation and truth-identity checks. All 30 true
cases and the artifact were visually examined. T22 was selected for two separated
pre-event silhouettes and a clean post-event daughter. A01 has both track IDs
overlapping one particle and a clear saved image-score decrease. This is selection
after seeing results for illustration, not an estimate of average effectiveness.

| Case | Frame / tracks | Trajectory → image score | Truth |
| --- | --- | --- | --- |
| T22 | 3782 / 12263, 12264 | 0.9982488931 → 0.9967463759 | Event 1135; parents 128.434 + 126.759 µm, daughter 160.768 µm |
| A01 | 3586 / 10387, 10980 | 0.8744009959 → 0.4803826797 | Same particle 4807 over nine preceding frames; no event within common evaluation gate |

The final candidate policy retains T22 and excludes A01. Scores in the figure
precede relation-attention. The retained/rejected labels refer to final selection,
not a 0.5 threshold applied to the displayed bars.

All displayed continuing-track positions and depth are actual smoothed observations.
Both tracks have actual observations in the pre-frame. At time zero the artifact
has both IDs observed; only the continuing track is observed in either post-frame.
The common displayed offsets are −5, 0 and +4 frames (−1.25, 0, +1.00 ms).
There are no interpolated markers and no manual or truth-based refocusing.
All six 128-pixel crops contain only recorded pixels. Contrast uses the original
linear 1st–99th percentile normalization independently for each crop. This display
normalization is not fed back into feature extraction or inference.

## Reproduce and verify

Use the environment with the original Matplotlib version; reconstruction is
explicitly CPU-only with four Torch threads, preserving the ongoing GPU campaign.

```bash
PY=/media/dai-server/M2Disk2TB/repos/260712/.venv/bin/python
MPLCONFIGDIR=/tmp/amt-figure9-mpl "$PY" experiments/amt_figure9_refinement_20261102/search.py
MPLCONFIGDIR=/tmp/amt-figure9-mpl "$PY" experiments/amt_figure9_refinement_20261102/search.py render
MPLCONFIGDIR=/tmp/amt-figure9-mpl "$PY" experiments/amt_figure9_refinement_20261102/finish.py
MPLCONFIGDIR=/tmp/amt-figure9-mpl "$PY" experiments/amt_figure9_refinement_20261102/verify.py
```

Private outputs and reconstruction caches are under
`../audit/revision-figure9-refinement-20261102/`. The final selection, panel intensity
limits, frame IDs, source hashes and checks are in `selection.json` and
`verification.json`. Earlier exploration is retained in `initial-preview/`.
Only the final PDF, preview PNG, comparison PNG and provenance are copied into
the author package by `amt_paper_replacements_20261102/build.py`.
The same final assets and provenance are also versioned in `results/` here;
contact sheets, full candidate tables and raw reconstruction caches stay private.
