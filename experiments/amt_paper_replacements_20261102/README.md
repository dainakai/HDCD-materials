# Synthetic manuscript replacements, 2026-11-02

This package uses the completed and hash-verified evaluation in
`experiments/amt_full_20261101/results`. It does not fit or rerun any model and
does not edit the manuscript or the active experimental campaign.

From the repository root:

```bash
.venv/bin/python experiments/amt_paper_replacements_20261102/build.py
.venv/bin/python experiments/amt_review_20261101/build.py
.venv/bin/python experiments/amt_review_20261101/check_html.py
```

The first command writes the author package and ZIP under
`../audit/revision-paper-replacements-20261102` and prepares the public files.
The second rebuilds the existing report and its HTTP allowlist. Do not expose
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
The previous package is archived privately as `paper-replacements-before-fig09-refinement.zip`.

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
