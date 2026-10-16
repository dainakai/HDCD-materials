# Trained 3D collision model for the synthetic test

License: MIT. This is the 3D collision model trained on synthetic development seeds 260740–260744. Terminal seed 260745 and experimental review labels were excluded from all fits.

The bundle includes 24-feature screening, 77-feature static, three 92-feature full classifiers, four coordinate residual regressors, relation attention and its training normalization. Use scikit-learn 1.7.1 and `ra3d.models.load_model` to check hashes and load it. No arbitrary pickle loading is needed.

The final relation fit used an NVIDIA RTX 4080 SUPER and reproduced projected-XY terminal AP 0.917, with ±5-frame/350-µm maximum event matching and no Z gate. All five independently refitted base out-of-fold arrays (scores generated with the scored scene excluded from fitting) matched the historical paper arrays bit for bit. Training provenance and file hashes are in model.json.

Inputs must follow the frozen candidate selection and missing-image policy. The model score is a within-scene ranking, not a calibrated collision probability. Validate transfer to a new imaging system. See [model selection](../../../docs/models.md) and [the runnable test workflow](../../../docs/use-cases.md#a-evaluate-the-pretrained-models-on-the-test-seed).
