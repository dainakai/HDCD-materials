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
about 3.10 pt at right because the new larger score label extends farther right.
Image panels, axes and label centres are not repositioned to accommodate this.

## Selection and display

The first refinement used `search.py` with fully centered 128-pixel crops and
selected T22/A01. At the author's request for larger particles, `larger.py`
expanded the display selection to allow moving the whole crop inside the image
at its edges. The recorded image itself is not altered or rescaled. The earlier
centered-crop condition excluded the large particles near the image boundary.
Nine larger true-event candidates and one larger artifact passed the new checks;
all ten were visually inspected. L03 was chosen for separated large parent
silhouettes and a clear daughter, and B01 for two actual track observations on
one large particle. This is post-hoc illustrative selection, not an estimate of
average effectiveness.

| Case | Frame / tracks | Trajectory → image score | Size / truth |
| --- | --- | --- | --- |
| L03 | 1767 / 4342, 4349 | 0.9978357501 → 0.9967471864 | Event 348; true parents 194.379 + 257.688 µm, daughter 290.263 µm |
| B01 | 1787 / 4486, 4436 | 0.5792988148 → 0.3152239211 | Smaller pre-frame measured diameter 251.726 µm; truth particle 2054 |

The exact true diameters are recorded in `results/selection.json` (the rounded
values printed in the figure are 194 + 258 → 290 µm). The earlier true event was
128 + 127 → 161 µm. The same 128-pixel field of view and plotting scale are used.
The final candidate policy retains L03 and excludes B01. Scores in the figure
precede relation-attention. The retained/rejected labels refer to final selection,
not a 0.5 threshold applied to the displayed bars.

All displayed continuing-track positions and depths are actual smoothed observations.
Both tracks have actual observations in the pre-frame and, for B01, at zero.
Only the continuing track is observed in either post-frame. The common displayed
offsets remain −5, 0 and +4 frames (−1.25, 0, +1.00 ms). No markers are interpolated,
and focus is not selected manually or from truth. Crop centers are clamped to
[64,960] in each coordinate so all pixels are recorded; the measured particle
disks have at least one pixel clearance from every crop boundary. The display
normalization remains the original per-crop 1st–99th percentile linear transform.
It is not fed back into feature extraction or inference.

B01 has a single ground-truth particle supporting both IDs over nine distinct
pre-frames (five and four observations), with no conflicting identity and no true
event within the common evaluation gate. A shared connected component is indicated
in eight of the nine image-feature pre-frames, not all nine.

## Reproduce and verify

Use the environment with the original Matplotlib version; reconstruction is
explicitly CPU-only with four Torch threads, preserving the ongoing GPU campaign.

```bash
PY=/media/dai-server/M2Disk2TB/repos/260712/.venv/bin/python
MPLCONFIGDIR=/tmp/amt-figure9-mpl "$PY" experiments/amt_figure9_refinement_20261102/larger.py
MPLCONFIGDIR=/tmp/amt-figure9-mpl "$PY" experiments/amt_figure9_refinement_20261102/larger.py render
MPLCONFIGDIR=/tmp/amt-figure9-mpl "$PY" experiments/amt_figure9_refinement_20261102/finish.py
MPLCONFIGDIR=/tmp/amt-figure9-mpl "$PY" experiments/amt_figure9_refinement_20261102/verify.py
```

Final outputs are under `../audit/revision-figure9-refinement-20261102/`.
The larger-particle search, contact sheets and caches are under
`../audit/revision-figure9-larger-20261102/`. The previous delivered smaller-particle
figure is preserved in `previous-smaller-cases/` under the final-output directory. The final selection, panel intensity
limits, frame IDs, source hashes and checks are in `selection.json` and
`verification.json`. Earlier exploration is retained in `initial-preview/`.
Only the final PDF, preview PNG, comparison PNG and provenance are copied into
the author package by `amt_paper_replacements_20261102/build.py`.
The same final assets and provenance are also versioned in `results/` here;
contact sheets, full candidate tables and raw reconstruction caches stay private.
