# Physical consistency and collision rates

The collision classifier is accompanied by physical diagnostics. These checks describe consistency of reviewed events and expected collision counts; they do not turn a selectively reviewed sample into a complete recall measurement.

For the paper figures, use [the figure workflow](figures.md), which already supplies the correct inputs and exposure. The examples below are for new analyses: replace the placeholder input filenames with your own tables after [core installation](environment.md#core-environment).

## Diameter and vertical momentum

```bash
collision-physics consistency --candidates reviewed-events.parquet \
  --tracks tracks.parquet --pixel-pitch-um 10 --fps 4000 \
  --output runs/physical-consistency.parquet
```

Filter your reviewed events as needed before calling the command. It uses the raw candidate frame, the last ten valid diameter observations at or before f−2, and the first twenty after f. The expected coalesced diameter is `(D1**3 + D2**3)**(1/3)`. Vertical velocities are medians of observed y-displacement/frame intervals over the corresponding windows. Downward y is positive. At least three observations per velocity window are required by default.

The expected post-event vertical velocity is `(D1**3*v1 + D2**3*v2)/(D1**3 + D2**3)`. Missing evidence produces NaN and explicit availability/count columns. Output JSON reports sample counts, errors and correlations. No new classification labels are assigned by this diagnostic.

## Hall-corrected gravitational rate

```bash
collision-physics rate --particles measured-particles.parquet \
  --diameter-column final_diameter_um --recorded-frames 10917 \
  --volume-cm3 10.72693248 --fps 4000 \
  --output runs/gravitational-pair-counts.parquet
```

Use the frame-wise measured particle table, not one row per trajectory. The default includes every recorded frame, giving 10,917 frames and 2.72925 s for this example. Supply your actual recording length and volume for other acquisitions. Number density and expected counts use the same volume and frame interval. The estimator uses 5 µm density bins from 25 to 300 µm (the final right edge is included, matching the historical histogram), reports unordered 25 µm diameter-pair bins, and applies a factor of one half within a pair-bin diagonal to avoid double counting.

The implementation preserves the paper's Davis–Jonas–Hall table interpolation and Beard terminal-velocity calculation at 900 hPa and 307.182512 K. Its numerical table comes from the BSD-licensed [SCALE-SDM source archive](https://doi.org/10.5281/zenodo.3841674), accompanying [Shima et al. (2020)](https://doi.org/10.5194/gmd-13-4107-2020). Coalescence efficiency is one. The table, atmospheric metadata, and original SCALE notice are bundled under `config/physics`; source-symbol hashes are recorded separately.

Uncertainty is conditional Poisson only: SD is the square root of the expected count, with measured density and kernel fixed. No temporal density variation is included. For pooled predictions, sum expected counts first and then take the square root; do not sum standard deviations.

The `rate` command defaults to full-recording counts and Poisson uncertainty. The figure-reproduction script defaults to `saved-figure`, which uses 10,904 frames, 11 temporal blocks and 4,000 density resamples. To use that convention with `rate`, supply both `--convention saved-figure` and `--volume-cm3 10.71644672`. The `poisson-only` option uses the same 10,904-frame interval with only Poisson uncertainty. See [count, exposure and uncertainty settings](physical-conventions.md).

Visually confirmed event counts are lower bounds when missed detections have not been measured. Comparing these counts to a gravitational kernel is a consistency check, not an efficiency-corrected experimental collision-kernel measurement.
