# Figure reproduction

Complete the [plotting environment setup](environment.md#figures), including the second Matplotlib environment, then run:

```bash
python scripts/reproduce_figures.py --output runs/figures
```

This recalculates experimental summaries and builds eleven figures. Individual vector PDFs are in `runs/figures/figures/`, raster previews in `runs/figures/preview/`, and the combined PDF is `runs/figures/figures.pdf`. Numerical comparisons are written to `verification.json`. The included figures in `paper/` use the same numerical inputs and layouts.

| Figure | Inputs and calculation |
|---|---|
| 4 | Per-sequence diameter densities and per-frame population summaries; 50,000 bootstrap replicates, seed 260715 |
| 5 | Per-track median diameters and velocities, with a common logarithmic count normalization |
| 7 | Particle PR, matched coordinate/diameter errors and trajectory lengths from the collision test sequence |
| 8 | Three model prediction tables, truth events, event PR and false-positive composition |
| 9 | N1 true collision and D4 shared-particle artifact; supplied floating crops, measured tracks and stored classifier scores |
| 10 | Confirmed collisions S0015 rank 1 and S0022 rank 2; five image times and 3D tracks |
| 11 | Scores and labels for the complete top-80 review cohort |
| 12 | Equal per-recording review budgets; vertical guides at 24 and 34, with black count/precision labels |
| 13 | Diameter and velocity window measurements and eligibility rules |
| 14 | Per-recording collision counts and gravitational model predictions |
| 15 | Parent-diameter pair counts and fine-bin gravitational predictions |

Fig.9 uses 128 × 128-pixel fields. Its display times are −0.25, 0, +0.25 ms for N1 and −0.50, 0, +0.50 ms for D4. Each image is normalized between its first and 99th intensity percentiles; all crops lie within the hologram boundaries. The artifact's two observed tracks correspond to the same truth particle. The examples illustrate individual score changes and were selected for visibility after examining the results.

Fig.10 places a 3D trajectory panel on the left and five images on the right. The first example displays a 96-pixel field and the second a 120-pixel field within their 128-pixel source crops. Each example uses a fixed field across its five times (−2, −1, 0, +1, +2 ms). Both parents remain visible in the second example's earliest frame. Each 500-µm scale bar spans 50 source pixels.

The figure commands do not require the full experimental raw recordings. After a complete benchmark, recalculate Fig.7 particle metrics from its new detections and tracks:

```bash
HDCD_RUN_ROOT=runs/benchmark HDCD_GPU_PYTHON="$PWD/.venv-gpu/bin/python" \
  .venv/bin/python scripts/benchmark/figure7.py
```

The CPU environment needs the plotting dependencies, and the GPU interpreter performs the low-confidence detection pass. The summary inputs distributed for Fig.7 were calculated from the same measurements and tracks as the supplied collision test features.

English LaTeX captions are in `paper/captions/`. They use `\unit` notation from siunitx. Fig.8 uses Matplotlib 3.11.1; the other plots use 3.10.9. Fonts, panel geometry and composition code are bundled, while PDF rasterization can differ across Poppler versions.
