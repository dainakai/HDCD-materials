# Experimental manuscript replacements after completed review

This analysis uses the completed top-80 review of all 14 sequences from
`audit/revision-experimental-20261102`. It does not retrain the detector, modify
the scientific processing engine, change human judgments, or edit the manuscript.

Outputs are in `audit/revision-experimental-results-20261103/package` (a sibling
of this repository). Private consistent SQLite and joined candidate snapshots
are outside the published package. Model SHA-256 is
`d648ebf407d18ff66b54670711dbebfe8997b1fa6a502a982b39803a19fda114`.

## Primary results

- 1,120 reviews: 186 collision, 922 noncollision, 12 ambiguous; last judgment
  saved at 2026-11-03 16:35:16 JST. Freeze completed at 16:45:07 JST.
- Top 30 per sequence: 175/186 conditional coverage (94.086%), 175/420
  precision (41.667%). The minimum budget reaching 95% is 34 per sequence
  (476 reviews, 177/186 = 95.161%, precision 37.185%). The precision-50%
  boundary is between N=24 (336 reviews, 51.488%) and N=25 (350, 49.429%);
  both recover 173/186 = 93.011%. There is no exact integer-budget 50% point.
  The earlier rounded-N=40 line is removed from Fig.12.
- Diameter diagnostics: 184 confirmed / 908 rejected. Velocity diagnostics:
  179 confirmed / 822 rejected. No residual-based exclusions.
- Confirmed diameter RMSE: 19.8994 µm; velocity RMSE: 0.0793867 m/s. The
  largest velocity residual accounts for 64.3% of squared error and has three
  post-event observations. Five-point sensitivity: 173 events, 0.0461580 m/s;
  it does not replace the primary three-point rule.
- Full-recording nominal exposure: 0.00040987072659456 m³ s. Confirmed count
  186; gravity prediction 222.410280 ± 14.913426 (conditional Poisson SD).
  Observed/predicted = 83.6292%. Rate = 453801.6207 m⁻³ s⁻¹.
- 185 events can be assigned to parent-diameter pairs, occupying 31 cells.
  S0015 rank 34 lacks one parent diameter. Its feature-table min/max both use
  the remaining diameter; using them unchecked would fabricate a same-bin pair.
  This analysis validates both actual parents, excludes that one event only
  from the pair map, and retains it in the total confirmed count.
- S0022 rank 27 lacks a post diameter but has both parents, so it remains in
  the pair map. These distinctions explain 186 total, 185 paired, 184 diameter
  and 179 velocity diagnostic events.

The public and revised reviews concern independently selected cohorts. Their
aggregate difference is not a paired event comparison or a causal ablation.
Coverage is conditional on the top-80 confirmed cohort, never absolute recall.

## Deliverables and integration

Main replacements are `figures/fig10.pdf` through `fig15.pdf`, with PNGs and
English captions. `text/experimental_section.tex` contains the full replacement
experimental subsection, including the figure environments and original labels.
Separate abstract/conclusion snippets and explicit anchors are supplied.
An optional review-budget table has no preassigned manuscript table number.

Reprocessed Figures 4–5 are supplemental replacement candidates. If used, the
provenance sentence must distinguish current experimental measurements from
the **unchanged** historical empirical inputs used to generate synthetic data.
Fig.5 retains the original display limits and now shares one logarithmic count
normalization across all three panels. Clipped display counts remain recorded.
Fig.3 is a procedure illustration and is outside this statistical update.

The author confirmed options 03 and 08 for Fig.10. Case (a) is S0015 rank 1,
ID `15:6509:6639:6750`; case (b) is S0022 rank 2, ID `22:3758:4041:4087`.
Both retain five views at −2, −1, 0, +1 and +2 ms relative to raw candidate
time, with the existing 128×128 source crops. `figure10_selected.py` uses
those cached images and restores the original 12.8-by-12.1-inch layout: 3D
trajectories on the left and five images at their original positions on the right.
Only the image field is enlarged inside those fixed panels: 96 pixels / 0.96 mm
for 03 (1.333x), and 120 pixels / 1.20 mm for 08 (1.067x). A 96px field for 08
would clip a parent at -2 ms. Both cases retain a fixed scale across all five
frames, and all observed particle disks remain inside their respective fields.
Scale bars remain 500 micrometres with their actual 50-source-pixel length.
The HTML view uses the same original composition rather than rearranging frames.

Fig.12 restores the original dimensions, colors, axes, typography and legend.
Only two vertical guides at N=24 and N=34 with short precision/count labels
replace the former N=40 guide. The full threshold explanation stays in the
caption and report; added shaded bands, horizontal guides and text blocks
have been removed.

The earlier two-case Fig.10 was kept in `versions/before-selection-2310bfc`.
It is now historical, as are the old `case_images.py` default cases. The ten-case
option gallery remains accessible with a notice of the confirmed choice.
`data/figure10_provenance.json` and `fig10-selected/selection.json` record the
current author decision, fixed frames, source hashes and the Fig.9 visual
reference. Original manuscript and parallel Fig.9 work remain untouched.

## Reproduction

Run from the repository root with the original local campaign available:

```bash
.venv/bin/python experiments/amt_experimental_results_20261103/freeze.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 .venv/bin/python experiments/amt_experimental_results_20261103/analyze.py
OPENBLAS_NUM_THREADS=4 .venv/bin/python experiments/amt_experimental_results_20261103/diagnostics.py
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 /media/dai-server/M2Disk2TB/repos/260712/.venv/bin/python experiments/amt_experimental_results_20261103/case_images.py
OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR=/tmp/amt-experimental-mpl .venv/bin/python experiments/amt_experimental_results_20261103/plot.py
```

Render final PDFs using system `/usr/bin/pdftoppm` into `package/preview/`
at 1600 pixels on the longer side; render `fig04.png` at 300 dpi into `figures/`.
Inspect the resulting images before finalizing. Other figure PNGs are saved at
300 dpi by their plotters.

```bash
.venv/bin/python experiments/amt_experimental_results_20261103/build_report.py
.venv/bin/python experiments/amt_experimental_results_20261103/verify.py
.venv/bin/python experiments/amt_experimental_results_20261103/publish.py
```

`freeze.py` reuses and validates an existing snapshot; it never overwrites one.
New human edits require a deliberately new analysis snapshot, not silent mixing.
`verify.py` independently recounts all 80 budget points, the parent-pair map,
valid diagnostic counts and RMSEs; it verifies live labels still match the
snapshot, validates PDF/PNG artifacts and report links, and checks the manuscript
has not changed. Density predictions are additionally checked against the public
physics API and a direct symmetric fine-bin integration during analysis.

The terminal has no TeX compiler. The PDF figures are rendered and verified;
full manuscript assembly/compilation is intentionally left to the author.

## Preview operation

The existing server on 127.0.0.1:8769 and existing Cloudflare Quick Tunnel are
reused; no new listener is needed. The report is at:

https://messenger-recognition-forgotten-valium.trycloudflare.com/experimental-results-20261103/index.html

Only the explicit package allowlist is merged into the existing preview manifest;
unrelated assets, including concurrent Fig.9 work, are preserved. Private SQLite
files, access keys, raw holograms and model weights are never copied. Receipt,
port, PIDs, log paths and archive hash are in
`audit/revision-experimental-results-20261103/publication.json`.

Stop the shared preview only when all pages may stop, using the existing
`audit/revision-review-20261101/stop_preview.py`. Stopping or recreating the Quick
Tunnel invalidates or changes its public URL. HTTPS resource hashes and private
route rejections are recorded in `https-verification.json` alongside the package.

## Follow-up: Figures 10, 12 and 13

- `gallery_images.py` reuses the manifest-verified frozen engine to reconstruct
  five crops for each of ten human-confirmed examples. The raw candidate frame
  is the display origin; offsets are −8, −4, 0, +4, +8 at 4000 fps. Crops are
  centered on the bounding extents of the currently observed parent particles
  to avoid clipping them. Focus depth follows the existing review convention.
  This is a display operation and changes no tracks, labels or analysis values.
- `gallery.py` produces `fig10-candidates.html`, ten single-case PDFs/PNGs/GIFs,
  fifty clean image crops, captions and `fig10-candidates.pdf` (ten pages).
  The pre-selection Fig.10 remains in the version archive; its examples were 04 and 07.
- `figure12_thresholds.py` produces the requested precision/coverage boundaries
  using equal integer budgets per sequence, without interpolation.
- `figure13_exclusions.py` audits all 1120 candidates and all 3360 windows
  against hashed source tracks, including raw/smoothed observation support.
  Exclusions: diameter 2 confirmed + 14 noncollision + 12 ambiguous; velocity
  7 + 100 + 12. All 119 unique excluded candidates have individual reasons and
  observed frame ranges in `fig13-exclusions.html` and CSVs. There are no invalid
  selected diameters/y-values; insufficient observed track support explains the
  technical exclusions. One-point windows are distinguished from API count zero.

Additional reproduction (does not require rerunning the scientific campaign):

```bash
OPENBLAS_NUM_THREADS=4 .venv/bin/python experiments/amt_experimental_results_20261103/figure13_exclusions.py
OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR=/tmp/amt-experimental-mpl .venv/bin/python experiments/amt_experimental_results_20261103/figure12_thresholds.py
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /media/dai-server/M2Disk2TB/repos/260712/.venv/bin/python experiments/amt_experimental_results_20261103/gallery_images.py
OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR=/tmp/amt-experimental-mpl .venv/bin/python experiments/amt_experimental_results_20261103/gallery.py
```

Rebuild the eight-figure combined PDF and render final new PDFs with system
`pdftoppm`. Then rebuild the report, verify, and publish as above. The initial
package was preserved at `versions/initial-e744ca9` before this follow-up.

To refresh only the confirmed Fig.10 and its large view after the gallery exists:

```bash
OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR=/tmp/amt-experimental-mpl .venv/bin/python experiments/amt_experimental_results_20261103/figure10_selected.py
OPENBLAS_NUM_THREADS=4 .venv/bin/python experiments/amt_experimental_results_20261103/build_report.py
OPENBLAS_NUM_THREADS=4 .venv/bin/python experiments/amt_experimental_results_20261103/results_summary.py
OPENBLAS_NUM_THREADS=4 .venv/bin/python experiments/amt_experimental_results_20261103/verify.py
.venv/bin/python experiments/amt_experimental_results_20261103/publish.py
```

The selected view is `experimental-results-20261103/fig10-selected.html` under
the existing Quick Tunnel. `plot.py` also uses this confirmed selection for
future full figure builds; it cannot silently revert to the old case pair.

## Consolidated results and layout correction

`results_summary.py` builds `summary.html` with Figures 4–5 and 7–15, Tables
4–5, the public-versus-revised comparison, review-budget thresholds, and links
to the full exclusion audit. `all-figures.pdf` contains those eleven current
figures. Figures 7–9 and their metric tables are exact copies from the final
N1/D4 synthetic package, not recomputed outputs. The existing synthetic ZIP
is linked separately. Figures 4, 5, 11, 13, 14 and 15 retain their prior bytes.

The superseded Fig.10 layout and expanded Fig.12 design are archived in
`versions/before-layout-correction-e3c91b0`. The selected candidate IDs, raw
source crops, five displayed times, model, labels and all performance values
remain fixed. Only image fields, the two requested Fig.12 guides and their
short labels change within the restored original designs.
