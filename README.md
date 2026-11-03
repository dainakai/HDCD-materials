# HDCD-materials

Evaluation code, trained models and data for **Semi-automated detection of droplet collision-coalescence using digital in-line holography**. HDCD stands for *Holographic droplet collision detection*.

## Set up the environment

Use Linux and Python **3.12.12**. Run the following commands from the repository root:

```bash
git clone --branch revision/amt-102p3mm-phase3 https://github.com/dainakai/HDCD-materials.git
cd HDCD-materials
python3.12 -m venv .venv
.venv/bin/python -m pip install uv==0.10.6
.venv/bin/python -m uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch==2.12.0
.venv/bin/python -m uv pip install --python .venv/bin/python \
  -r requirements-core-reference.txt -e .
source .venv/bin/activate
python scripts/check_environment.py
python scripts/prepare_data.py --profile test
python scripts/verify_release.py
```

This environment runs the supplied collision models and the numerical analyses on a CPU. Reserve 6 GB for the environment and unpacked inputs. Hologram generation and particle measurement additionally require an NVIDIA GPU, the [GPU environment and Julia setup](docs/environment.md#hologram-generation-and-particle-measurement), and about 150 GB of working space. Generated holograms stay in the run directory.

## Evaluate the trained models

```bash
python scripts/reproduce.py --output runs/test-evaluation
```

The command runs all three models on the complete test sequence (seed 260745, 6,000 frames, 2,000 true collision events), calculates event precision-recall curves, and compares predictions and metrics with the reference evaluation. The five training seeds are 260740–260744.

| Test metric | Ours | Depth-free | MinIP-based 2D |
|---|---:|---:|---:|
| Average precision | 0.922169 | 0.913348 | 0.877041 |
| Maximum F1 | 0.888366 | 0.869355 | 0.830834 |
| Precision at recall ≥ 0.8 | 0.969110 | 0.945068 | 0.847009 |
| False positives at recall ≥ 0.8 | 51 | 93 | 289 |
| Precision at recall ≥ 0.9 | 0.787402 | 0.694981 | 0.495595 |
| False positives at recall ≥ 0.9 | 486 | 790 | 1,832 |

Read `runs/test-evaluation/verification.json` for the comparison. Scores order candidates; they are not calibrated collision probabilities. AP integrates the all-points interpolated precision envelope. Event matching uses maximum one-to-one association within ±5 frames and 350 µm in the image plane, without a depth gate. [Evaluation and training definitions](docs/algorithm.md).

## Reproduce the numerical experiment from holograms

After completing the GPU and Julia setup:

```bash
python scripts/reproduce_benchmark.py --output runs/benchmark --julia julia
```

This generates all six particle sequences and paired holograms, reconstructs and measures particles, links and smooths trajectories, selects candidates, calculates features, fits all three collision models, and evaluates the test sequence. It uses the bundled particle-measurement weights and fits the collision models from the five training sequences.

The generation volume spans 0–102.3 mm. Reconstruction uses 1,024 planes from 74.9 to 177.2 mm relative to the primary camera, with three phase-retrieval iterations throughout. MinIP includes the terminal plane. The generator uses explicit PCG64 streams and fixed distribution tables; every generated particle table and camera image is checked against a recorded SHA-256 before downstream processing. [Generation, resumption and reproducibility checks](docs/simulation.md).

A shorter integration check generates a complete particle sequence and processes its first 128 image pairs:

```bash
python scripts/reproduce_benchmark.py --smoke --output runs/smoke --julia julia
```

## Reproduce tables and figures

```bash
python scripts/reproduce_tables.py --output runs/tables
python scripts/reproduce_experimental.py --output runs/experimental
```

The first command recalculates **Tables 4 and 5**, including the four scoring stages and their common downstream operations. The second recalculates review budgets, diameter and velocity diagnostics, and gravitational collision predictions from the experimental measurements and labels.

Install the [plotting dependencies](docs/environment.md#figures), then run:

```bash
python scripts/reproduce_figures.py --output runs/figures
```

[Figures 4–5 and 7–15](paper/figures.pdf), individual PDFs/PNGs and English captions are also supplied in [paper/](paper/README.md). Fig.7 uses the particle measurements and tracks from the same synthetic test sequence as the collision evaluation. Fig.9 shows a true collision and two tracks of the same physical particle. Fig.10 shows two confirmed experimental collisions. [Figure inputs and commands](docs/figures.md).

## Experimental data

The review contains 80 candidates from each of 14 recordings: **186 collisions, 922 non-collisions and 12 ambiguous cases**. At 24 reviews per recording, precision is 51.49% and coverage is 93.01%; at 34 reviews, precision is 37.18% and coverage is 95.16%. Coverage is relative to the 186 confirmed events in the reviewed top-80 cohort.

Ten confirmed collision examples include reconstructed image sequences. Event trajectories, review labels, per-window physical measurements and compact population summaries support the analyses. Full experimental recordings and their raw holograms are not included. There are 185 events with both parent diameters, 184 eligible diameter diagnostics and 179 eligible velocity diagnostics. The total count remains 186; missing measurements affect only the corresponding analyses. [Data dictionary and example files](docs/data.md).

The gravitational model predicts **222.4103 ± 14.9134** collisions across the 14 recordings; the uncertainty is the conditional Poisson standard deviation. The observed count is a lower bound from the reviewed candidates. [Physical conventions](docs/physics.md).

## Citation and licenses

Please cite the associated paper and the commit or dataset version used. [CITATION.cff](CITATION.cff) contains citation metadata. Numerical research data use [CC BY 4.0](LICENSE-DATA.md); project software and collision-model weights use [MIT](LICENSE). Dependencies and bundled components retain their own licenses; see [Third-party notices](THIRD_PARTY_NOTICES.md).
