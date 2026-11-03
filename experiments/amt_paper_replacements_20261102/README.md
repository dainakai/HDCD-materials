# Synthetic manuscript replacements, finalized 2026-11-03

This package uses the completed and hash-verified evaluation in
`experiments/amt_full_20261101/results`. It does not fit or rerun any model and
does not edit the manuscript or the active experimental campaign.

From the repository root:

```bash
.venv/bin/python experiments/amt_paper_replacements_20261102/build.py
.venv/bin/python experiments/amt_paper_replacements_20261102/verify.py
python3 experiments/amt_figure9_refinement_20261102/publish_final.py
.venv/bin/python experiments/amt_review_20261101/check_html.py
```

The first command writes the author package and ZIP under
`../audit/revision-paper-final-20261103` and prepares the public files.
Run `verify.py` after the build; `publish_final.py` then updates final selection
notices and the scoped HTTP allowlist. Do not expose
the repository directory itself.

The output provides Figures 7–9, Tables **4 and 5** in the current manuscript,
and English LaTeX text with exact source anchors. Historical evaluation filenames
`table3` and `table4` must not be used as manuscript table numbers.

Figures 7 and 8 are exact copies. Figure 9 uses the author-requested refinement
in `experiments/amt_figure9_refinement_20261102`: newly selected illustrative
cases, 128-pixel crops, the original per-crop 1st–99th percentile contrast,
and the recovered original plotting function and submission relabeling script.
The layout, fonts, markers and horizontal score bars follow the manuscript.
The crops were reconstructed only for display from the frozen new synthetic
sequence; candidate scores, model fits and performance metrics are unchanged.
The original package is archived privately as `paper-replacements-before-fig09-refinement.zip`.
The previous 128-pixel smaller-particle package is preserved as
`paper-replacements-before-larger-particles.zip`. The superseded off-center
version is archived as `paper-replacements-before-centered-high-score.zip`.
The final author-selected cases are N1 (153.79 + 241.59 → 260.79 µm,
0.998955 → 0.997927) and D4 (one true particle, 321.39 µm,
0.943667 → 0.624751). N1 uses −0.25/0/+0.25 ms; D4 uses
−0.50/0/+0.50 ms. All six crops are entirely within the hologram.
The original 128px image arrays are embedded natively in the PDF; visible
panel borders and layout match the original exactly. D4's two observed IDs
17945/18031 share truth particle 7364. Nine displayed observations in the
combined figure have been checked independently against physical truth.
The rejected B01/B02 distinct-particle suggestions are not used.

`../audit/revision-figure9-final-20261103/verification.json` records PDF
geometry, embedded fonts, exact raster encoding, scores and truth checks.
Visual inspection is mandatory before `build.py` accepts the final figure.
The historical package under `revision-paper-replacements-20261102` is preserved.

`texts.py` holds proposed language. `build.py` checks the 34 finalized source
hashes, generates the package, asserts that the manuscript is unchanged, and
records output hashes. `verify.py` checks numeric/caption consistency, the
refinement's verified PDF hash, TeX structure and the ZIP. `check_interactions.cjs` checks copy and
print handlers without claiming browser rendering coverage.

This host uses `/usr/bin/pdftoppm`, because the bundled Poppler binary requires a
newer glibc than the host provides. The repository Python environment supplies
matplotlib and pypdf; the Codex dependency bundle supplies pdfplumber/reportlab.

The existing Quick Tunnel and server are reused. Their port, PIDs, URL, logs and
stop instructions remain in `../audit/revision-review-20261101/preview-processes.json`
and `experiments/amt_review_20261101/PREVIEW.md`. The package publication receipt
is written separately, without copying the experimental review access key.
