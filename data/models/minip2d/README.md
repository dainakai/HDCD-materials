# MinIP-based 2D trained model

License: MIT. Fitted on seeds 260740–260744, excluding test seed 260745 and experimental review labels. The complete model contains 13-feature screening, 49-feature trajectory and 64-feature final classifiers, 41-input relation attention, normalization, and time/x/y localization regressors. The 15 additional classifier inputs are MinIP image features.

Load with `ra3d.minip.load` and scikit-learn 1.7.1. Test AP is 0.858; [the comparison guide](../../../docs/baseline2d.md) provides the supported test-evaluation workflow. `model.json` records feature order, model-file hashes and training input hashes. Out-of-fold base scores were recomputed using classifiers fitted without each scored training scene.
