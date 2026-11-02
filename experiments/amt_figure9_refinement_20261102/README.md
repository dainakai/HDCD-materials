# Fig.9 correction, 2026-11-02

This author-requested revision changes illustrative examples and display only.
Model fits, saved candidate scores and 34 frozen numerical outputs remain
unchanged. The manuscript is not edited.

## Correction and selected cases

The previous larger-particle version (`3cf2861`, L03/B01) incorrectly moved
128-pixel crop windows inside the recorded image to avoid black margins. That
made targets appear near the crop edge. Its artifact trajectory score was also
only 0.579. This version supersedes it: crops are centered on actual continuing
track observations, and out-of-frame regions stay black as in the original.

| Case | Frame / tracks | Trajectory → image score | Ground truth |
| --- | --- | --- | --- |
| C04 | 5523 / 18638, 18667 | 0.9584858402 → 0.9574296554 | Event 1815; 285.733 + 285.336 → 359.751 µm |
| H01 | 2813 / 8518, 8507 | 0.9086474013 → 0.4914342017 | Both IDs on particle 3605, diameter 276.472 µm |

Four true-event and two high-score artifact previews were visually inspected,
including C01, the regenerated realization of original planned event 1165.
C04 was chosen for its large, visible two-to-one transition. H01 satisfies the
author's trajectory score >=0.9 requirement and drops by 0.417213 after image
features. These are post-hoc illustrative choices, not an average-effect
estimate. The final candidate policy retains C04 and excludes H01; displayed
scores precede relation-attention and do not directly define that final policy.

The frames are −4, 0, +1 relative to candidate time (−1.00, 0, +0.25 ms).
The post-frame precedes the targets leaving the recording area. All markers are
actual observations; no missing positions are invented. All target disks lie
inside the recorded image and crop. The two post-frame crops alone have small
black bottom margins. Only recorded pixels determine the linear 1st–99th
percentile display normalization. Focus is the continuing track's estimated
depth, with no manual or truth-based adjustment.

`audit_centered.py` independently checks nine displayed observations against
truth: nearest projected center, center within the correct projected disk,
and axial error below 100 µm. C04 has `identity_unknown` in the strict training
labels because its terminating track has no one-to-one truth match. Do not
call it strict `identity_consistent`: its displayed observations independently
correspond to parents 7607/7608 and daughter 7683. H01's one-to-one mapping
supports only continuing track 8507; it does not map both IDs to one particle.
Independent geometry establishes that both displayed IDs lie on particle 3605.
Its image features indicate a shared component in all nine pre-event frames.

Particle 3605 is a parent of a real collision two frames later. That event is
481 µm from this candidate, outside the common XY gate of 350 µm. The artifact
is the duplicated track pair on one particle; it is not evidence that no
physical collision occurs anywhere in the view. This detail is disclosed in
the HTML and audit JSON. No labels, fitting or evaluation rules are changed.

## Original figure and data provenance

The `../CollisionPaper` clone was fetched and checked at
`2a5d3f4e4b4723aab782c2f42b473d5b5e3bbd92`, matching `origin/main`.
Original Fig.9: `本番フォーマット/figures/fig09.pdf`.
Original plot recovered from
`/media/dai-server/DATAM2/siderepos/260804/CollisionPaper/scripts/plot_image_feature_case_study.py`.
Its byte-identical archived copy is
`paper-range-only-20260930/sources/paper_style/image_feature_case_study.py`.
The unmodified source is frozen at `source/original_plot.py`, SHA-256
`b4b1c09f1e34ea694ee599eebc984318b3581c4a0a51902cc500cca928640a4f`.
`source/prepare_figures.py` is the manuscript's submission relabeler.

The original true candidate is 12515_12410_3859 (event 1165), score
0.976999 → 0.964198; original artifact 17742_17721_5331, score
0.889649 → 0.335793. Same training/test seeds and identical empirical input
CSV distributions do not imply identical samples after regeneration at a
102.3 mm generation depth. Original event 1165 has parents 5136/5137 and child
5213 at (1.035, 9.091, 46.774) mm; the current realization has parents
5167/5168 and child 5233 at (0.231, 7.536, 15.824) mm, at the same frame 3859.
The current C01 preview was inspected but C04 gives a clearer and larger
transition. `results/original-comparison.json` records hashes and exact values.

Two rows of three 128-pixel images, red circles/blue squares and track tails,
horizontal bars, Matplotlib 3.10.9 and Liberation Sans follow the original.
The submission relabeler's Arial-compatible Liberation Sans fallback is used
for captions; the original PDF embeds ArialMT there. Original image-panel
positions match within 0.01 pt. Tight PDF width varies slightly with score
labels, without changing panel coordinates. All rendered fonts are embedded.
The final PDF was rendered and visually inspected after assembly.

## Reproduce and verify

Use the original Matplotlib environment. Display reconstruction runs on CPU
with four Torch threads, preserving the ongoing GPU campaign.

```bash
PY=/media/dai-server/M2Disk2TB/repos/260712/.venv/bin/python
MPLCONFIGDIR=/tmp/amt-figure9-mpl "$PY" experiments/amt_figure9_refinement_20261102/centered.py
MPLCONFIGDIR=/tmp/amt-figure9-mpl "$PY" experiments/amt_figure9_refinement_20261102/audit_centered.py
MPLCONFIGDIR=/tmp/amt-figure9-mpl "$PY" experiments/amt_figure9_refinement_20261102/finish.py
MPLCONFIGDIR=/tmp/amt-figure9-mpl "$PY" experiments/amt_figure9_refinement_20261102/verify.py
```

Candidate previews/caches: `../audit/revision-figure9-centered-20261102/`.
Final outputs: `../audit/revision-figure9-refinement-20261102/`.
The latter preserves `previous-smaller-cases/` and
`previous-offcenter-large-cases/`. Historical `search.py` and `larger.py` remain
for provenance; they do not select the current examples. Only selected assets
and audit records are copied to the author package. Full reconstruction caches,
model weights and full research data remain private. Current final assets and
provenance are versioned in `results/`.

## Separate frame-selection gallery

`frame_gallery.py` exports all six consecutive frames from −1.00 to +0.25 ms
for the selected (a) and (b) cases, without any trajectory or score overlay.
Labels A1–A6 and B1–B6 appear outside the image pixels. Native 128-pixel PNGs,
two horizontal contact sheets, and a separate responsive `fig09-frames.html`
page are written under `../audit/revision-figure9-frames-20261102/` and copied
to the existing scoped report preview. Only six missing intermediate frames
are reconstructed; the other six match the current Fig.9 display cache exactly.
The current paper figure, selection and replacement ZIP are hash-checked to
remain unchanged. Run with the same Python environment as above:

```bash
MPLCONFIGDIR=/tmp/amt-figure9-mpl "$PY" experiments/amt_figure9_refinement_20261102/frame_gallery.py
```

The gallery uses the existing Quick Tunnel and is for the author's subsequent
frame choice. `publication.json` in its private output directory records the
HTTPS byte checks, process receipt and stop command. Raw reconstruction caches
remain outside the public directory.
