# Data dictionary

## Numerical test inputs and models

`data_sources/manifest.json` maps compressed byte streams to their decoded files. Each part is smaller than 50 MiB and has a SHA-256. `scripts/prepare_data.py` joins and decompresses the parts, then checks the decoded SHA-256. The final files remain ordinary local files and are excluded from Git. `DATASET_INDEX.csv` lists table dimensions and columns; `python scripts/verify_release.py` checks the distributed file manifest and any decoded inputs already present.

| Path | Contents |
|---|---|
| `data/test/ours.parquet` | Complete 92-feature test scoring union for Ours |
| `data/test/depth_free.parquet` | Complete 64-feature test scoring union for Depth-free |
| `data/test/minip2d.parquet` | Complete 64-feature test scoring union for MinIP-based 2D |
| `data/test/trajectory.parquet` | Static shortlist and scores for the trajectory row of Table 5 |
| `data/test/collision_events.parquet` | All 2,000 test truth events and parent/daughter identities |
| `data/models/{ours,depth_free,minip2d}/` | Trained classifiers, attention weights, normalization and recipes |
| `reference/evaluation/` | Event predictions, precision-recall curves and full-precision reference metrics |
| `reference/measurement/` | Particle-measurement and tracking source; decoded network weights go under `models/` |
| `reference/generation/` | Julia renderer, dependency locks, generation arguments and reference hashes |
| `data/figure7/` | Particle PR curve, per-observation errors, number densities and track-length summaries from test seed 260745 |

The candidate key is `(vanished_track_id, survivor_track_id, candidate_frame)`. Predicted events also preserve `candidate_frame_raw`, corrected coordinates and `_source_index`. The test feature tables do not contain training labels. Feature definitions and units are in [features.md](features.md).

Model trees use skops, attention weights use safetensors, and normalization arrays use NumPy NPZ. The particle-measurement checkpoints retain their original PyTorch format and are checked against the bundled hashes before loading.

## Experimental reviews and measurements

`data/experimental/reviews.csv` contains exactly 1,120 rows: the top 80 candidates in each of 14 recordings. `scene_num` is the acquisition-system sequence number; the paper numbers these recordings consecutively from 1 to 14. The mapping is in `scripts/figure_support/scene_index.py`.

| Column | Meaning |
|---|---|
| `candidate_id` | Sequence, terminating track, continuing track and raw candidate frame |
| `review_score_rank` | One-based score rank within a recording |
| `collision_score` | Final detection score used to order the review |
| `status` | `collision`, `non_collision` or `ambiguous` |
| `binary_label` | 1 for collision, 0 for non-collision, missing for ambiguous |
| `review_stratum` | `top_score` for this fixed review cohort |
| `review_inclusion_probability` | 1 within the selected top-80 cohort; not a sampling probability for all detections |
| `candidate_frame_raw` | Frame used for physical windows and example image times |
| `pre_diameter_vanished_um`, `pre_diameter_survivor_um` | Separately measured parent diameters; missing values are retained |
| `post_diameter_um` | Measured continuing-track diameter after the candidate |

`event_measurements.parquet` provides the diameter-window boundaries, per-window velocity estimates, observation counts and eligibility flags for all reviewed candidates. `figure13_window_evidence.csv` and `figure13_eligibility_all1120.csv` record individual exclusion reasons and observed frame support. A one-point window cannot provide an interval velocity; the API reports a velocity count of zero for that case.

Diameter diagnostics include 184 confirmed and 908 non-collision events. Velocity diagnostics require at least three observations in each window and include 179 confirmed and 822 non-collision events. Ambiguous cases remain in the review budget and are excluded from binary physical diagnostics. No residual-based exclusion is applied.

`diameter_density_by_scene.csv`, `frame_metrics.parquet` and `velocity_tracks.parquet` contain compact population summaries for Figures 4–5 and the gravitational calculation. `velocity_tracks.parquet` has one summary row per eligible track, not a full recording. The complete raw holograms and full experimental trajectory sequences are not distributed.

## Collision examples

`data/experimental/collisions/manifest.json` identifies ten confirmed collision image sequences and their GIF hashes. `trajectories.parquet` contains windows around all 186 confirmed collisions; `candidate_id` distinguishes repeated track IDs across examples and recordings. The two Fig.10 examples additionally provide the five original floating reconstruction crops and display arrays in `data/experimental/figure10/`.

`data/synthetic_examples/` supplies the six Fig.9 crops, their observed trajectories, scores and truth identities. N1 is event 1204, with parent IDs 5288 and 5289 and daughter 5377. D4 contains observed tracks 17945 and 18031 of the same truth particle 7364. N1 remains in the final predictions; D4 is removed by final selection.

All image examples are reconstructed crops. They do not provide the full experimental raw holograms. The numerical holograms can be regenerated using [simulation.md](simulation.md).
