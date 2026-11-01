# AMT revision report and experimental-review preparation

This follow-up converts the preflight report into a responsive HTML document and checks whether the manuscript can be completed by replacing figures and values.
The answer is **not yet**: the seven controlled models are complete, while the new 102.3 mm paired holograms and final three-method pipeline are not.
It does not run or relabel a new experimental evaluation.

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
The source raw-image disk is currently absent from the recorded location, and the retained 100 images per camera do not provide a complete 10,917-frame sequence.
These records are input contracts, **not executable final acquisition configurations**: complete image manifests, validated calibration coefficients, and final model fingerprints are still needed.

Repeated preparation refreshes the inventories but refuses to overwrite changed annotation or correspondence CSVs. Record actual new reviews in a separate, version-bound run directory rather than editing these templates.
Historical labels must not be copied to new ranks.
Keep the historical empirical distribution used to generate synthetic data fixed unless a separate version change explicitly updates it.

## Preview

`serve.py` binds only to `127.0.0.1:8769` and serves files explicitly listed in `public-manifest.json`.
It does not list directories or expose raw images, weights, review records, or other repository files.
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

- All 29 published files were fetched from the HTTPS URL and matched local bytes.
- All 88 HTML links, anchors and resources resolved.
- Four comparison selectors produced the expected saved AP values; mobile-menu closing and print handlers were exercised with a minimal DOM substitute.
- Unlisted files and traversal paths were rejected with HTTP 404/400.
- Actual browser visual QA was unavailable because local browser use is disabled in the app settings. It is not represented as a completed screenshot or smartphone test.
- The manuscript and original scientific results are unchanged. This follow-up adds the report and preparation artifacts only.

See the audit directory's `html-verification.json`, `interaction-verification.json`, and `public-url-verification.json` for the recorded checks.
