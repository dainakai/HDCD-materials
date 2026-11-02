# AMT revision report and experimental-review preparation

This follow-up converts the preflight report into a responsive HTML document and checks whether the manuscript can be completed by replacing figures and values.
The answer remains **not by replacing figures and numbers alone**: the seven controlled comparisons and the regenerated 102.3 mm, six-sequence, three-method pipeline are complete. The final numerical results, fixed models, and Figure 7–9 exports were verified on November 2. Manuscript wording, experimental reprocessing/review, and the release's default data/model switch remain pending.
It does not run or relabel a new experimental evaluation.

The main report includes an explicit comparison against tag `v1.0.1` (`0b0419fa844419bff1b4b6a93c0505b4c31a775a`). In particular, the published Depth-free MinIP already began at 74.9 mm, measurement and MinIP already used three phase iterations, and rank-before-suppression and the AP algorithm are retained. The R5 description distinguishes existing classifier identity selection from the changed identity observations and attention supervision.

## Build and inspect

Run from the materials repository with its existing virtual environment:

```bash
.venv/bin/python experiments/amt_review_20261101/prepare.py
.venv/bin/python experiments/amt_review_20261101/build.py
.venv/bin/python experiments/amt_review_20261101/check_html.py
```

`prepare.py` verifies the 63 preflight result files against their manifest, derives controlled differences from the saved metrics, counts the actual new paired-image files, and inventories the 14 historical experimental scenes.
It reads source archives without modifying them.
Outputs are written under `../audit/revision-review-20261101/`:

- `evidence/`: aggregate inventories and numerical contrasts.
- `experimental-preparation/scenes/`: 14 source, calibration, frame-origin and target-condition records.
- `experimental-preparation/historical_review_crosswalk.csv`: 1,120 historical reviews, with new candidate and label fields left empty.
- `experimental-preparation/new_review_annotations.csv`: empty new-review schema.
- `experimental-preparation/review_contract.json`: cohort identity, label definitions, historical correspondence and analysis rules.
- `public/`: only the HTML pages and explicitly linked aggregate evidence and plots.

The preparation records the requested experimental front as 74.9 mm and preserves the historical calibrated front separately.
Scenes 17/18 previously used 74.7 mm, and 20/21/22 used 74.8 mm.
All historical experimental MinIP summaries use 12 phase iterations; the target is 3.
The source 16 TB disk was connected on November 2 and mounted read-only. All 28 camera directories have contiguous indices 1–10,917, and the six calibration bundles are present. The original preparation records remain historical; new executable acquisition contracts, file manifests, smoke checks, and processing state are in `../audit/revision-experimental-20261102/`. See [experimental processing](../amt_experimental_20261102/README.md). Final model fingerprints remain those of the completed numerical campaign.

Repeated preparation refreshes the inventories but refuses to overwrite changed annotation or correspondence CSVs. Record actual new reviews in a separate, version-bound run directory rather than editing these templates.
Historical labels must not be copied to new ranks.
Keep the historical empirical distribution used to generate synthetic data fixed unless a separate version change explicitly updates it.

## Preview

`serve.py` binds only to `127.0.0.1:8769` and serves files explicitly listed in `public-manifest.json`.
It also forwards only `/experimental-review/` to the dedicated loopback review service on port 8770. That service requires its private entry key/session and binds annotations to frozen cohorts. Only the selected reconstructed crops and local trajectory snippets are shown. Directory listings, full raw images, weights, database files, historical labels, and other repository files are not served.
The server and Cloudflare Quick Tunnel are running for this review; their current URL, PIDs and logs are in the audit directory's `preview-processes.json` and `PREVIEW.md`.
Reuse that tunnel while it is running.

For a later restart, run the server and tunnel as separately maintained processes, updating the receipt:

```bash
python3 -u experiments/amt_review_20261101/serve.py --port 8769
/home/dai-server/.local/bin/cloudflared tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:8769
```

Never serve the repository root or the parent audit directory.
Quick Tunnel URLs expire when the process stops and change on recreation.

## Verification and limits

- The November 2 completed-result report contains 43 public files; static files were fetched from the HTTPS URL and matched local bytes, while the live progress JSON was checked separately.
- Local HTML links, anchors and resources are rechecked after each report update; the latest count is in `html-verification.json`.
- Four comparison selectors produced the expected saved AP values; mobile-menu closing and print handlers were exercised with a minimal DOM substitute.
- Unlisted files and traversal paths were rejected with HTTP 404/400.
- Actual browser visual QA was unavailable because local browser use is disabled in the app settings. It is not represented as a completed screenshot or smartphone test.
- The manuscript and original scientific results are unchanged. This follow-up adds the report and preparation artifacts only.

See the audit directory's `html-verification.json`, `interaction-verification.json`, and `public-url-verification.json` for the recorded checks.
