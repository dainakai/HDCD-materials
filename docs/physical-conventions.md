# Count, exposure and uncertainty calculations

[Repository home](../README.md) · [Figure reproduction](figures.md) · [Physical diagnostics](physics.md)

The default figure command uses the `saved-figure` calculation below to reproduce Figures 14–15. Counts, observation volume-time and uncertainty must use the same convention when comparing rates.

## 1. Paper-figure calculation

| Setting | Value |
| --- | --- |
| Experimental sequences | 14 |
| Confirmed reviewed collisions | 185 |
| Frame range per sequence | 0–10,903 inclusive (10,904 frames) |
| Frame rate | 4,000 Hz |
| Volume per frame | 10.71644672 cm³ |
| Total observation time | 38.164 s |
| Pooled volume-time | 4.0898247262208 × 10⁻⁴ m³ s |
| Density/kernel grid | 5 µm |
| Display bins | Hard 25 µm diameter bins; 66 unordered pairs |
| Density resampling | 11 temporal blocks per sequence; 4,000 resamples; seed 260724 |

A count is divided by the corresponding volume-time to obtain a rate in m⁻³ s⁻¹. The 185 confirmed events give 4.5234 × 10⁵ m⁻³ s⁻¹. These are reviewed-event counts; completeness over unreviewed physical collisions is not known.

The gravitational expectation uses the bundled Hall collision-efficiency table and Beard terminal velocities. The kernel is evaluated at 5 µm bin centers before aggregation into the 25 µm display bins. Unordered pairs within the same display bin receive half-weight. Unequal droplets within one display bin can have a nonzero gravitational kernel.

For expected count `lambda` and standard deviation `sd_density` from temporal density resampling, the displayed count standard deviation is:

```text
sd_total = sqrt(lambda + sd_density**2)
```

The pooled expected count is 210.057, density-resampling SD 1.978, and total SD 14.628. Individual scene and diameter-pair values are recalculated by the figure script. Input tables are under `data/paper_figures/gravity/`; reference values are under `data/paper_figures/reference/`.

```bash
python scripts/reproduce_figures.py --figures 14 15 \
  --output runs/paper-counts --verify-rendered
```

Read `figure14/scene_counts.csv` and `figure15/diameter_pair_counts.csv` for full-precision results. `paper_reference_comparison.json` compares counts, rates, exposure and uncertainty with the independently saved reference tables.

## 2. Optional alternative calculations

| Option | Density and exposure | Count SD |
| --- | --- | --- |
| `saved-figure` (default) | Settings above | `sqrt(lambda + sd_density**2)` |
| `poisson-only` | Same density and exposure as the default | `sqrt(lambda)` |
| `manuscript` | All 10,917 frames and nominal 10.24 × 10.24 × 102.3 mm³ volume | `sqrt(lambda)` |

The `manuscript` option uses the counting convention described in the text. It gives expected count 210.119, SD 14.495, and observed rate 4.5136 × 10⁵ m⁻³ s⁻¹. The full-recording density counts are bundled in `gravity/recorded_particle_counts_by_scene_5um.csv`.

```bash
python scripts/reproduce_figures.py --figures 14 15 \
  --gravity-convention manuscript --output runs/alternative-counts
```

These alternative calculations produce different plots from the paper figures and report `COMPUTED_WITH_PAPER_DIFFERENCES`. Use the default for exact paper-figure reproduction. The alternative outputs do not change the reference files.

## 3. Rebuild full-recording density inputs

When the original 14 particle tables are available, create a CSV with columns `raw_scene_num,path`, one row per sequence, then run:

```bash
python scripts/prepare_gravity_counts.py --sources scene_sources.csv \
  --output runs/recorded_particle_counts.csv --audit runs/particle_count_sources.json
```

Paths may be relative to the source CSV. Each source needs `frame` and a measured-diameter column (`final_diameter_um`, `diameter_um` or `slice_diam_pred_um`). The extractor preserves NumPy's histogram convention, including the final right edge at exactly 300 µm. The [recorded source hashes](../provenance/gravity_particle_counts.json) identify the supplied counts. Normal figure reproduction does not require the original tables.

For synthetic optical and coordinate settings, see [acquisition and reconstruction conditions](benchmark-reconstruction.md).
