# Included collision models

All weights needed for the supported evaluation are included. No fitting or model download occurs during test inference.

| Method | Model directory | Final classifier inputs |
| --- | --- | --- |
| Ours | `data/models/paper/` | 77 trajectory + 15 image features |
| Depth-free variant | `data/models/baseline2d/` | 49 trajectory + 15 MinIP image features |
| MinIP-based 2D | `data/models/minip2d/` | 49 trajectory + 15 MinIP image features |

Each bundle includes fitted trees, relation-attention weights, normalization, feature ordering and event-localization regressors. Checksums and scikit-learn version are recorded in `model.json`. [Feature catalog](features.md) · [Algorithm](algorithm.md).

All three models were fitted using synthetic seeds 260740–260744. Test seed 260745 and experimental review labels were excluded from fitting. The training sequences are outside this reduced distribution. The Ours bundle is the verified refit used by this material's existing reproduction workflow.

Run all three models with:

```bash
python scripts/reproduce.py --suite terminal --output runs/pretrained-test
```

[Use cases](use-cases.md) describe outputs and scope. HoloD3 particle-measurement weights and the separate experimental deployment model are not included.
