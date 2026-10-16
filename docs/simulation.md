# Regenerate the test particle states

The data-preparation command regenerates the complete 6,000-frame particle-state table for seed 260745. This supplies the particle truth for Figure 7 without distributing the large finished table.

## Determinism contract

The seed is one part of the recipe. `reference/generation/contract.json` also fixes particle count, planned-event count, mixture allocation, diameter and velocity distributions, source-code hashes, random-generator algorithm and numerical-library versions. The wrapper validates those inputs before generating data.

The original generator uses NumPy PCG64 with the recorded seed offsets. The tested environment is Python 3.12, NumPy 2.4.6 and pandas 3.0.3. The same complete-frame recipe is used even when inspecting only a portion of the result.

```bash
python scripts/prepare_data.py --profile figures --output runs/figure-inputs.json
```

This compares the generated particle table with its original SHA-256. It stops if the complete-file checksum differs. Numerical measurements from a particle detector, trajectories and image features are separate recorded inputs; this simulation does not produce them.

The original generator and optical source are retained for provenance. The supported route here generates numerical particle truth; hologram rendering, upstream measurement weights and image-to-track reproduction are outside this distribution.
