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

Figures 7 and 8 are exact copies. Figure 9 preserves the six image panels and
all their raster data, scale bars and overlays by clipping and translating the
original PDF graphics at scale 1.0. The same saved scores are plotted together
as panel (c). Its original audit title is removed. No case is reselected and no
scientific image is regenerated.

`texts.py` holds proposed language. `build.py` checks the 34 finalized source
hashes, generates the package, asserts that the manuscript is unchanged, and
records output hashes. `verify.py` checks numeric/caption consistency, PDF image
identity, TeX structure and the ZIP. `check_interactions.cjs` checks copy and
print handlers without claiming browser rendering coverage.

This host uses `/usr/bin/pdftoppm`, because the bundled Poppler binary requires a
newer glibc than the host provides. The repository Python environment supplies
matplotlib and pypdf; the Codex dependency bundle supplies pdfplumber/reportlab.

The existing Quick Tunnel and server are reused. Their port, PIDs, URL, logs and
stop instructions remain in `../audit/revision-review-20261101/preview-processes.json`
and `experiments/amt_review_20261101/PREVIEW.md`. The package publication receipt
is written separately, without copying the experimental review access key.
