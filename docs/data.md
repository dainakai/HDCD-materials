# Data dictionary

The distribution covers the complete held-out test seed 260745 and the numerical inputs for nine paper figures. [Preparation](data-preparation.md) creates the large tables at the paths below. `DATASET_INDEX.csv` lists rows, columns and file sizes; `reference/expected_outputs.json` records checksums and schemas.

| Path | Role |
| --- | --- |
| `data/inputs/test_candidates.parquet` | Fixed 566,261-row 3D shortlist with candidate geometry and candidate-generation features |
| `data/tracking/validation/seed260745/smoothedtrajs.parquet` | Complete smoothed 3D test trajectories used to calculate features |
| `data/derived/test_static77.parquet` | Locally calculated trajectory-feature table before exact reference serialization |
| `data/features/validation/seed260745/static77.parquet` | Original 77-feature shortlist |
| `data/features/validation/seed260745/static_scored.parquet` | Same shortlist with original trajectory-stage scores and ranks |
| `data/features/validation/seed260745/paper92.parquet` | 260,123 final 3D scoring rows with the paper's image-feature missingness |
| `data/baseline2d/validation/seed260745/full.parquet` | 241,444 final depth-free scoring rows |
| `data/minip2d/validation/seed260745/full.parquet` | 241,240 final MinIP-based 2D scoring rows |
| `data/evaluation/candidate_geometry_{3d,2d,minip2d}.parquet` | All 566,261 shortlist positions per method for candidate-coverage evaluation |
| `data/labels/validation/seed260745/collision_events.parquet` | 2,000 true test collision events |
| `data/labels/validation/seed260745/particles.parquet` | Complete test particle states, generated locally |
| `data/predictions/{3d,2d,minip2d}/validation/seed260745.parquet` | Reference event predictions for comparison and Figure 8 |
| `data/models/{paper,baseline2d,minip2d}/` | Three fitted collision-model bundles |
| `data/paper_figures/` | Numerical measurements, review labels and figure references |

## Units and identifiers

Frames are zero-based at 4,000 frames/s. Candidate identity is the tuple `vanished_track_id`, `survivor_track_id`, `candidate_frame`, within one sequence. Track IDs identify measured trajectories, not simulated particle identities. Do not join different sequences by track ID alone.

Trajectory `x` and `y` are image pixels, `z` is the stored slice coordinate and `diameter_px` is a pixel diameter. Pixel pitch is 10 µm and slice spacing is 100 µm. Candidate/event columns ending in `_um` use micrometres. The stored collision-candidate coordinate is `z_um = z * 100`. Time/position regressors produce final event estimates. [Processing settings](benchmark-reconstruction.md).

`_source_index` links an event to its row in the final scoring table. `candidate_frame_raw` identifies its original candidate time; `candidate_frame` is the corrected event time. Preserve input row order because ranks and tie handling use it.

Missing feature values remain NaN. Image measurements are missing when the image policy did not select a candidate. Derived indicators follow the original formulas; do not replace missing measurements with zero indiscriminately. [Feature definitions](features.md).

The historical directory name `validation` denotes the independent test sequence. Training labels retained in some reference tables are provenance fields; pretrained inference does not use them. [Models](models.md).

## Experimental numerical inputs

The [figure guide](figures.md#4-what-is-recalculated-for-each-figure) defines the population summaries, paired particle measurements, review fields and conservation/rate inputs. These cover 14 sequences. Complete experimental tracks and collision feature tables are outside this distribution.
