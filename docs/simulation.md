# Numerical hologram benchmark

## Fixed physical conditions

Each seed generates 6,000 frames at 4,000 frames/s, 200 particle centers inside the observation volume per frame, and 2,000 scheduled coalescence events. Seeds 260740–260744 train the collision models; 260745 is the test sequence. Each camera records 1,024 × 1,024 pixels at 10 µm/pixel. The wavelength is 0.6328 µm and the camera separation is 21,450 µm.

Particle centers move continuously through a volume spanning 0–102,300 µm in depth. The reconstruction planes are 74,900 + 100k µm from the primary camera, for k = 0,…,1023. Trajectories extend through a 256-pixel lateral propagation margin to provide natural inflow and outflow. Particle motion is not quantized to reconstruction planes.

The two fixed CSVs in `src/ra3d/config/simulation/` specify the diameter distribution and horizontal-velocity quantiles used to generate the benchmark. These empirical distributions were established before the experimental measurements presented in Figures 4–5. The new experimental summaries do not replace these generation inputs. The half-mixture scheme assigns 1,000 events to each of its two allocation components. The vertical velocity offset is −0.26 m/s, with a per-particle truncated-normal relative perturbation for diameters up to 150 µm (mean zero, standard deviation 0.1, limits −0.1 and +0.1). The sampled factor multiplies the terminal velocity after the −0.26 m/s offset.

Paired holograms use sequential transmission masks and angular-spectrum propagation. No optical noise is added. All phase retrieval uses three iterations. Measurement and MinIP use 1,536-pixel padding; collision image features use 2,048-pixel padding. The MinIP propagator samples planes 0,2,…,1022, then explicitly adds plane 1023, preserving the 200-µm recurrence for its first 512 planes.

## Generation and seed control

The canonical command calls `ra3d.simulation.generate` with the arguments in `reference/generation/contract.json`. NumPy generators explicitly use **PCG64**, with seed offsets 0 (main simulation), 113 (velocity sampling) and 197 (diameter sampling). Python's separately seeded sampling routines and all model random states are fixed in code. File traversal and candidate ordering use explicit sorting; hash-randomized set iteration does not select the simulated particle sequence.

```bash
python scripts/generate_holograms.py --seed all --stage labels --output runs/generated
python scripts/generate_holograms.py --seed 260745 --stage render \
  --renderer julia --julia julia --output runs/generated
```

Label generation always creates the complete sequence. A rendering interval can be selected with `--start-frame` and `--end-frame`; the latter is exclusive. Frame IDs are zero-based in tables and one-based in image filenames.

The generator checks its source code and distribution hashes, checks the Python/NumPy/pandas versions, and compares both generated label CSVs against the recorded SHA-256 for each seed. A mismatch raises an error before rendering or fitting. This detects changes that keeping the same integer seed alone would miss. Both Julia rendering commands compare every camera image with `reference/generation/render-sha256.json`.

GPU architectures, CUDA/FFT implementations and image encoders can produce different floating-point or encoded image values even with identical seeds. The reference pipeline detects those differences rather than silently treating them as the same inputs. Use the pinned environment and inspect a mismatch before proceeding. A CPU Python renderer is available for optical exploration; it is not substituted for the Julia renderer in the reference benchmark.

## Complete processing

```bash
python scripts/reproduce_benchmark.py --output runs/benchmark --julia julia
```

The stages are `labels`, `render`, `initialize`, `measurement`, `tracking`, `features`, `train` and `evaluate`. The output contains the generated camera images and truth, per-seed measurements and tracks, intermediate feature tables, newly fitted model directories, predictions and precision-recall curves. The three methods use the same new hologram realizations. Depth-free uses the 3D tracks without depth; MinIP-based 2D independently links the projection measurements.

The particle-measurement networks are fixed. Stage-one and static collision classifiers, out-of-fold base scores, full-feature classifiers, relation attention and localization regressors are fitted from the five training sequences. The test seed is excluded from fitting. Its earlier realization was inspected during sensitivity analysis; final settings were fixed before regenerating the six sequences. This history limits treating the test seed as an entirely untouched source of model-design information.

## Resumption

Completed label and rendered-image hashes are checked before reuse. The top-level command locks its output directory against a second invocation. For example, after all six sequences are rendered:

```bash
python scripts/reproduce_benchmark.py --output runs/benchmark --from-stage initialize --julia julia
```

Use the same output directory, code and environment for continuation. Later stages retain completion records. After an interrupted measurement or training stage, inspect the log and incomplete outputs before restarting that stage; a completion record must not be copied from another run. A fresh output directory is the appropriate choice after changing scientific settings.

`--smoke` processes the first 128 camera pairs, constructs tracks and candidates, and calculates 32 candidate feature vectors for each image-feature implementation. It is an integration test and is excluded from training and performance tables.
