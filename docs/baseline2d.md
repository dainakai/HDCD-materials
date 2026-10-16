# Comparison methods

## Method comparison

| Method | Measurement source | Collision inputs |
| --- | --- | --- |
| Ours | Holographic 3D particle measurements and trajectories | 77 trajectory and 15 image features |
| Depth-free variant | Same holographic measurements, with explicit depth-related features excluded | 49 trajectory and 15 minimum-intensity-projection (MinIP) image features |
| MinIP-based 2D | Particle measurements and trajectories obtained from MinIP images | 49 trajectory and 15 MinIP image features |

Each method has its own fitted model and final scoring table. The supplied candidate-coverage tables retain the complete 566,261-row shortlist positions for each method. They permit coverage checks without distributing each method's full earlier candidate pools.

## Prepared-feature-table inputs

The depth-free reference features retain their original preparation, including the reviewed examples used in the research code's reference-feature stage. Those prepared values are supplied; this distribution does not rerun that stage. The two final comparison tables are recovered exactly and used by the existing prediction routines. Training labels are discarded before inference.

```bash
python scripts/reproduce.py --suite terminal --output runs/three-method-evaluation
```

[Model files](models.md) · [Evaluation definitions](paper-validation.md#evaluation-definitions). Full comparison-model retraining and MinIP image processing require data outside this distribution.
