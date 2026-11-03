# Fig.9 correction, 2026-11-02

This author-requested revision changes illustrative examples and display only.
Model fits, saved candidate scores and 34 frozen numerical outputs remain
unchanged. The manuscript is not edited.

## Current author decision: (a) confirmed, (b) alternatives, 2026-11-03

The author confirmed N1 with −0.25, 0 and +0.25 ms: frames 3957, 3958 and
3959, tracks 12754/12764, true event 1204. `confirm_true.py` exports these
three cached images through the original plotting functions as a standalone
Fig.9(a) PDF/PNG. Original image-panel dimensions, fonts, markers, tails and
scale bar are preserved. All four displayed positions have one-to-one truth
matches and all crops are fully recorded. Scores remain 0.9989552783 →
0.9979272758. `results/confirmed-a/selection.json` is the current author
decision; the older case-selection commands below are historical.

Full Fig.9 and the existing author replacement ZIP remain historical versions
until (b) is selected. Public pages explicitly flag that status and link to
the confirmed (a) component. Do not treat the old hard-coded C04/H01 choice
in `finish.py` as the current author decision.

## Corrected (b) review: same physical particle, 2026-11-03

The author rejected B01/B02 because both follow **distinct** true particles.
Their recommendations and proposed changes to the scientific phenomenon/caption
are withdrawn. `publish_artifact_options.py` is disabled to prevent republishing
them. A shared reconstructed component alone does not establish common physical
identity. The manuscript definition is explicit at `template.tex` lines 164–169
and 914–925: two observed IDs intermittently track one true particle.

`shared_particle.py select` searches the entire frozen scoring union (263,185
rows), including all 13,031 candidates with trajectory score >=0.9 and a score
decrease. It independently associates actual XY/Z observations with physical
truth. `intermittent` additionally checks the 412 candidates with a decrease
>0.2 using a broader support fraction, because a particle can be created inside
the history window and detections can alternate between IDs. These screens are
explicitly not a proof that every possible duplicate-tracking pattern is found.
Final displayed observations are individually audited and visually inspected.

| Candidate | Frame / red, blue IDs | Same truth ID / diameter | Trajectory → image score | Display times | Final list |
| --- | --- | --- | --- | --- | --- |
| D4, preferred | 5337 / 17945, 18031 | 7364 / 321.39 µm | 0.9436670450 → 0.6247505742 | −0.50, 0, +0.50 ms | absent |
| D2, qualified comparison | 5508 / 18567, 18658 | 7645 / 318.55 µm | 0.9372951809 → 0.4880977487 | −0.25, 0, +0.25 ms | retained |

D4's five actual positions in the proposed three panels all lie inside the
same physical truth disk; axial errors are below 100 µm. Its full 128px crops
have at least 35px to the hologram boundary. Its blue ID begins after the true
particle's creation. The earlier real coalescence is 25 frames (6.25ms) before
the candidate. Red positions at −0.75/−0.25ms are outside the target truth disk;
blue is missing at +0.25ms. These are excluded from the proposed paper panels
and explicitly shown as diagnostic frames. The +0.25ms diagnostic uses linearly
interpolated image center/focus only; no blue marker or tail is invented.

D2's proposed three crops have at least 204px to the recording boundary and
both IDs follow one true particle before red terminates. The IDs originally
followed distinct parents that merged eight frames (2ms) before this candidate;
this history is disclosed. It remains in the final policy, so the original
`(rejected)` score-panel annotation cannot be used with D2. Its −1.00ms blue
point lies 1.38px outside the physical truth disk and is diagnostic only.

Other examined cases remain private: D1's pair follows actual distinct parents
before a coalescence six frames before its candidate; D3 is a smaller 147µm
example with a lower decrease (0.920→0.651). H01 has a strong score decrease
but the +0.25ms centered crop crosses the hologram boundary. None is silently
substituted for the requested phenomenon or promoted as the chosen (b).

`publish_shared_particle.py` verifies saved scores, final presence, independent
truth geometry, all 34 frozen outputs, original manuscript and confirmed (a)
checksums before publication. It produces separate paper-style track images
and yellow truth-outline proof images, plus explicit diagnostics. All presented
rasters were inspected. The public `fig09-artifact-options.html` URL is reused
with a withdrawal notice and the corrected review; old B01/B02 direct downloads
contain withdrawal notices. No final full Fig.9 or author replacement ZIP is
changed before author selection.

Private outputs: `../audit/revision-figure9-shared-particle-20261103/`.
Versioned review record and three-panel previews: `results/shared-particle/`.
The original (a) record stays in `results/confirmed-a/selection.json`.

```bash
PY=/media/dai-server/M2Disk2TB/repos/260712/.venv/bin/python
MPLCONFIGDIR=/tmp/amt-figure9-mpl OPENBLAS_NUM_THREADS=1 "$PY" experiments/amt_figure9_refinement_20261102/shared_particle.py select
MPLCONFIGDIR=/tmp/amt-figure9-mpl OPENBLAS_NUM_THREADS=1 "$PY" experiments/amt_figure9_refinement_20261102/shared_particle.py intermittent
MPLCONFIGDIR=/tmp/amt-figure9-mpl OPENBLAS_NUM_THREADS=1 "$PY" experiments/amt_figure9_refinement_20261102/shared_particle.py render
MPLCONFIGDIR=/tmp/amt-figure9-mpl OPENBLAS_NUM_THREADS=1 "$PY" experiments/amt_figure9_refinement_20261102/shared_particle.py diagnostics
MPLCONFIGDIR=/tmp/amt-figure9-mpl OPENBLAS_NUM_THREADS=1 "$PY" experiments/amt_figure9_refinement_20261102/publish_shared_particle.py
```

## Earlier candidates: complete crops inside the hologram, 2026-11-03

The author requested (a) candidates without clipping near the hologram boundary.
`interior_true.py` now requires all six complete 128×128 crops to remain inside
the 1024×1024 hologram, with at least 8 pixels between the crop edge and the
hologram edge. Crops remain centered on actual continuing-track measurements;
no inward shift, clamp, fill or change of scale is used. Both truth disks and
measured disks have at least 3 pixels of clearance within each crop.

The search found 232 cases with six complete interior crops, 223 with verified
identities and visible disks, and 79 after initial particle separation and
common-focus checks. Of these, 31 have both parents >=100 µm and all ten XY
errors <=3 pixels (also <=40% of the truth radius, with axial error <=250 µm).
The larger N1 option has a maximum XY error of 4.31 pixels, disclosed explicitly.

| Candidate | Frame / terminating, continuing tracks | True diameters (µm) | Mean / max XY error (px) | Minimum crop-to-hologram edge distance (px) |
| --- | --- | --- | --- | ---: |
| N1 | 3958 / 12754, 12764 | 154 + 242 → 261 | 1.45 / 4.31 | 12 |
| N2 | 5894 / 19387, 19524 | 146 + 154 → 190 | 1.20 / 2.61 | 43 |
| N3 | 163 / 65, 325 | 145 + 163 → 195 | 1.12 / 2.00 | 51 |
| N4 | 1897 / 4252, 4359 | 146 + 151 → 187 | 1.42 / 2.84 | 63 |
| N5 | 4178 / 12913, 13140 | 130 + 142 → 172 | 1.11 / 2.36 | 208 |
| N6 | 5472 / 17670, 17995 | 116 + 123 → 151 | 0.67 / 1.86 | 307 |

All 36 panels were visually inspected. N5 has relatively unobstructed target
images farther inside the hologram; N1 preserves larger particles; N2/N3
balance size and localization. The errors summarize ten displayed observations
per case and are not whole-test metrics. All cases are strict identity-consistent
at candidate time and retained by the final policy; candidate time equals the
true event time. N1's blue point at −0.75 ms has no existing one-to-one mapping,
but independently matches the nearest expected truth parent with 1.99px XY and
14.3µm axial error. N1's maximum error is a different point, at +0.25 ms. N2–N6
have one-to-one mappings for all ten displayed points.

Private outputs: `../audit/revision-figure9-interior-20261103/`.
The public `fig09-interior-a.html` page contains six time frames per candidate,
original red/blue markers and recent tails, optional plain PNGs, exact scores,
boundary distances and a ZIP with 72 individual PNGs and six contact sheets.
Previous candidate galleries link to this page. The current paper PDF still
uses the rejected C04 and is unchanged pending the author's selection.

```bash
PY=/media/dai-server/M2Disk2TB/repos/260712/.venv/bin/python
MPLCONFIGDIR=/tmp/amt-figure9-mpl OPENBLAS_NUM_THREADS=1 "$PY" experiments/amt_figure9_refinement_20261102/interior_true.py select
MPLCONFIGDIR=/tmp/amt-figure9-mpl OPENBLAS_NUM_THREADS=1 "$PY" experiments/amt_figure9_refinement_20261102/interior_true.py render
MPLCONFIGDIR=/tmp/amt-figure9-mpl OPENBLAS_NUM_THREADS=1 "$PY" experiments/amt_figure9_refinement_20261102/publish_interior_true.py
```

Rendering uses the same three-iteration phase recovery on CPU with four
threads. Exact candidate keys are pinned in `PREVIEW_CASES`; raw crops can be
reused only after checking the saved center coordinates. Every reconstruction
asserts that all pixels were recorded. Publication protects the current paper
PDF, selection record and replacement ZIP with before/after SHA-256 checks.
The existing tunnel is reused, with HTTPS checks and process details recorded
in the private `publication.json` receipt. Scientific results and experimental
processing remain unchanged.

## Current status: the old (a) is rejected

The author correctly identified poor particle localization in C04. It is no
longer recommended as a clean tracking illustration. The current paper PDF
still contains that example, pending the author's choice of a replacement.
The earlier selection rationale below is preserved as history, not as approval
of C04. The artifact H01 remains unchanged.

All four red observations in frames 5519–5522 already have raw XY errors of
12.27–12.67 pixels. Smoothing leaves errors of 12.47–14.10 pixels. The source
measurement records have `contrast_area_f70_bg90_full_gate2px_bbox_fallback`
and coordinates equal to their small detection-box centers near the particle
edge. In the frozen production code, this method means the contrast-area
centroid was more than 2 pixels from the box center, so the box center was
returned. The crop and marker overlay accurately display those saved values;
their display coordinates are not the cause. Requiring only inclusion within
the truth particle's projected disk was an inadequate selection criterion.

`clean_true.py` checks identity, visibility and all ten measured positions
across the six displayed frames. `render_clean_true.py` reconstructs four
alternatives with the same 128-pixel crop, measured focus, original contrast
and red/blue overlay. `publish_clean_true.py` records the old failure and
publishes `fig09-clean-a.html`, with a correction link from the old gallery.
All 24 reconstructed panels were visually inspected.

| Alternative | Event frame | True diameters (µm) | Mean / maximum XY error (px) |
| --- | ---: | --- | --- |
| C1, larger particles | 4335 | 232 + 269 → 317 | 2.42 / 4.43 |
| C2 | 3593 | 186 + 221 → 259 | 1.41 / 4.22 |
| C3, position priority | 5249 | 150 + 176 → 206 | 1.05 / 2.47 |
| C4 | 4740 | 153 + 226 → 247 | 1.61 / 2.85 |

These means and maxima cover ten actual markers per example, not the whole
test series. One pixel is 10 µm. C1's red positions have errors of 1.82–2.70
pixels; its maximum is the blue position at 0 ms. None of these examples is
claimed to be error-free. Truth is used for selection and auditing only; no
marker is moved and no image is refocused using truth. The search criteria
and unsuccessful stricter thresholds are recorded in `search-protocol.json`.

Use the Python environment below, in this order:

```bash
MPLCONFIGDIR=/tmp/amt-figure9-mpl OPENBLAS_NUM_THREADS=1 "$PY" experiments/amt_figure9_refinement_20261102/clean_true.py
MPLCONFIGDIR=/tmp/amt-figure9-mpl OPENBLAS_NUM_THREADS=1 "$PY" experiments/amt_figure9_refinement_20261102/render_clean_true.py
MPLCONFIGDIR=/tmp/amt-figure9-mpl OPENBLAS_NUM_THREADS=1 "$PY" experiments/amt_figure9_refinement_20261102/publish_clean_true.py
```

Output and private diagnostics:
`../audit/revision-figure9-clean-true-20261102/`. The candidate ZIP includes
24 overlaid frames and four contact sheets. The paper PDF, selected-case
record and author replacement ZIP are hash-checked to remain unchanged.
Final paper assembly awaits the author's frame choice. The existing tunnel,
port 8769, server PID 2058041 and tunnel PID 523581 are reused; its logs and
stop command are recorded in the private `publication.json` receipt.

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
MPLCONFIGDIR=/tmp/amt-figure9-mpl "$PY" experiments/amt_figure9_refinement_20261102/frame_gallery_tracks.py
```

The second command adds the author's requested red-circle/blue-square markers
and recent track lines, and makes that version the gallery's primary view.
It reuses the original plot's overlay functions and the saved 12 image crops;
no reconstruction is repeated. Lines cover the preceding 1.00 ms through the
displayed frame. Only actual observations are marked: A5/A6/B6 have no red
observation. The overlay is rendered separately at the original panel's
physical scale and composited over exact 2x nearest-neighbor images; pixels
outside the overlay are verified unchanged. A1–A6 and B1–B6 retain their
times and crop positions. `tracks-manifest.json` records the 21 actual markers
and tail coordinates. The original plain images remain available per panel.

The gallery uses the existing Quick Tunnel and is for the author's subsequent
frame choice. `publication.json` in its private output directory records the
HTTPS byte checks, process receipt and stop command. Raw reconstruction caches
remain outside the public directory.
