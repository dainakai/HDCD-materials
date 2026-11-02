# Local monitoring and experimental review

The scheduled 2026-11-02 task adds a dedicated monitoring page and extends the existing local review app. The reference `collisiongifserver` repository was inspected: it uses Workers/R2/D1 and includes old-data upload/deletion behavior. This implementation reuses the campaign's local Flask/SQLite review service, with the same three review choices, candidate browsing and revert-to-unreviewed interaction. No external review database or legacy labels are imported.

## Running pages

Both services bind only to localhost. The existing report server proxies their exact route prefixes through the existing Cloudflare Quick Tunnel.

| Service | Local port | Public path |
| --- | --- | --- |
| Monitoring | 8771 | `/experimental-monitor/` |
| Protected review | 8770 | `/experimental-review/` |
| Existing report/proxy | 8769 | `/` |

Base URL: `https://messenger-recognition-forgotten-valium.trycloudflare.com`.

The review's bootstrap URL is private and stored in `audit/revision-review-20261101/preview-processes.json`. It activates the existing secure session cookie, then redirects to a clean URL. It must not be embedded in public monitoring HTML or committed to Git. The monitoring page can link to the clean review route after the user has opened the private link once.

Current PIDs, URLs, ports and logs are recorded in that process receipt. At launch: monitor 2511735, review 2511741, report proxy 2511742; existing tunnel 523581 and aggregate report publisher 2310151 remained running. Always verify the command line before acting on a saved PID. `audit/revision-review-20261101/stop_preview.py` was extended to stop the new monitoring service as well. It checks process identity and does not stop scientific workers. Recreating the Quick Tunnel changes its URL.

Standalone startup commands, after confirming the ports are unused:

```bash
.venv/bin/python -u experiments/amt_experimental_20261102/monitor_server.py --port 8771
.venv/bin/python -u experiments/amt_experimental_20261102/review_server.py --port 8770
.venv/bin/python -u experiments/amt_review_20261101/serve.py --port 8769
```

Use detached processes with logs and update the process receipt when restarting. The tunnel command remains the existing localhost:8769 command; no new Cloudflare account resources are needed.

## Monitoring semantics

`monitor_server.py` collects a new snapshot approximately every 30 seconds in a background thread. HTTP requests return the cached snapshot immediately. `monitor_snapshot.py` is a standard-library, read-only helper also copied into the private KIT workspace. It queries GPU utilization/memory and reads stage receipts, bounded log/CSV tails and output filenames. It does not import or execute scientific code.

- Background: number of saved images from both cameras.
- MinIP: numbered completion markers written by the scientific stage.
- Measurement: final particle CSV's last emitted frame. The completed detection `frame_stats.csv` is explicitly labeled as preprocessing and is not mistaken for completed depth/diameter measurement.
- Inference: recognized progress records when available; otherwise no invented percentage.
- Review media: saved GIF completion markers.
- Stage completion: only PASS stage receipts.
- Review-ready: only a complete, full-scene, 80-candidate manifest with a completed assets stage on the local machine. Remote assets are not counted until verified import finishes.

Completed-stage counts are not wall-time percentages. No finish-time estimate is fabricated. Controller heartbeat age and process liveness are checked independently. A failed remote connection retains the previous observation with its original timestamp and an explicit connection warning. The client flags snapshots older than 90 seconds. Monitor JSON contains no raw paths, commands, PIDs, review keys or research image payloads.

The monitor's cached state and log are in `audit/revision-experimental-20261102/web/`. GPU1's independent job remains untouched; the scientific dispatcher retains its existing admission policy.

## Review behavior and persistence

Every full scene retains exactly its original ranked top 80 candidates. Filtering and gallery browsing change only the display, never the cohort or rank. Scene selection is preserved in the URL; loading a scene starts at its first unreviewed candidate. Completed scenes appear automatically.

The app supports collision, non-collision and ambiguous judgments; saved notes; three trajectory projections; enlarged GIF viewing; candidate and label filters; a lazy-loaded gallery; and returning a judgment to unreviewed. Reverting removes the current annotation row while retaining its prior history and adding an `unreviewed` history event. Images and candidate manifests remain intact.

Unsaved notes are retained in the current browser tab, including across candidate navigation. During saving, controls and notes are locked to prevent mismatched requests. A failed save preserves the draft. With automatic advance enabled, the next candidate's image is brought into view. Keyboard shortcuts do not classify while editing an input. The app uses the existing SQLite database and fixed cohort fingerprints; CSRF and access checks remain enabled.

Before restarting the review service, SQLite's backup API created `web/reviews-before-web-update.sqlite3`, including any WAL-backed data. No production judgment was entered during verification. Only test fixtures received successful save/revert requests.

## Verification

- Monitor tests distinguish final particle progress from already-complete detection, reject remote review readiness before local import, retain timestamps on lost connections, and limit HTTP routes to monitor resources.
- Review tests verify authorization/CSRF, persistence across app restarts, model/prediction/cohort binding, label validation, revert history and image preservation.
- JavaScript tests cover save payloads, automatic advance, failed-save notes, draft navigation, filters and revert behavior.
- The real HTTPS pages were opened in the in-app browser. Mobile-width layout, monitor filtering, scene/candidate navigation, XZ projection, draft retention, empty filters, 80-item gallery and GIF loading were checked without submitting real labels.
- HTTPS verification checks the page assets, live state, authenticated scene APIs, representative first/last GIF hashes for both ready scenes, CSV export, blocked private paths and rejected invalid labels. Real review/history row counts remain unchanged.

Evidence is recorded in `web/https-verification.json` and `web/browser-verification.json`. The four scientific campaign files, execution adapters and fixed numerical model are not modified by this Web task.
