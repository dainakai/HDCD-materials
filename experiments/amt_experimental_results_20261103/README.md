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
  precision (41.667%). The minimum budget reaching 95% is 34 per sequence;
  top 40 gives 179/186 (96.237%) and 179/560 (31.964%).
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

The two Fig.10 images depict the physical examples in the previous Fig.10, but
use new track hypotheses and measurements. Case (a): S0021 rank 5, raw candidate
3612, display zero 3608; (b): S0017 rank 18, raw candidate 8565, display zero
8563. Candidate IDs come from the new cohort, not the old PDF filenames.
The display origins only select illustrative snapshots and do not change the
analysis windows. Ten frames are reconstructed with the frozen engine on the
local RTX 4060 Ti, with three iterations and 2048 padding; the engine manifest
is fully checked before reconstruction. GPU numeric variability was previously
authorized. See `data/figure10_provenance.json`.

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
