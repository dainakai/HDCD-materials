# Terms used in this release

[Repository home](../README.md) · [Reproducibility guide](paper-validation.md)

| Term | Meaning here |
| --- | --- |
| Scene / sequence | One independent simulation seed or experimental recording. Split training and testing by whole scenes, not by adjacent frames. |
| Ground truth | The simulator's complete list of physical particles and prescribed coalescence events, before measurement errors are introduced. Experimental visual reviews are a selected sample, not exhaustive ground truth. |
| Candidate | A hypothesis that a terminating trajectory merges into a continuing trajectory at a particular frame. Many candidates can describe the same physical event. |
| Terminating / continuing track | The observed track that ends near the candidate time / the track whose ID continues afterward. File columns call these `vanished_track_id` and `survivor_track_id`. |
| Feature | A numerical measurement used by a model, such as separation, diameter change or shared-image-component count. `static77` means 77 trajectory features; `paper92` adds 15 image-derived features, preserving the paper's missing values. |
| Checkpoint / weights | Saved fitted model parameters. Loading them performs inference without training again. A complete collision model also includes feature order, normalization and event-position/time corrections. |
| Frozen | Saved reference inputs or outputs. Evaluating frozen predictions checks the evaluator; running weights on feature tables checks model inference too. |
| True positive (TP) / false positive (FP) | A prediction assigned to a true event by one-to-one matching / a remaining unmatched prediction. Experimental plots use these labels for visually confirmed/rejected candidates instead. |
| Precision / recall | Matched predictions divided by all predictions / matched true events divided by all true events. The matching gate and score threshold must be specified. |
| Precision–recall (PR) curve | Precision and recall as the minimum accepted score is lowered. Equal scores enter together. |
| Average precision (AP) | A summary of a PR curve. This release uses an all-points interpolated precision envelope; see the exact [formula](paper-validation.md#evaluation-definitions). |
| F1 | The harmonic mean of precision and recall: `2 * precision * recall / (precision + recall)`. “Maximum F1” selects the best score threshold on the evaluated curve. |
| Out-of-fold (OOF) prediction | A prediction for a scene made by a model trained without that scene. For example, scores for seed 260740 use the other four training seeds. These held-out scores train the next model stage without using in-sample base scores. |
| Nested evaluation | An outer scene is reserved for evaluation. When preparing scores to train the next stage, the base model also excludes the inner scene being scored. The published upstream candidate tables remain fixed; the entire image-processing chain is not refitted inside every split. |
| Relation attention | A model that compares competing candidates sharing a terminating track and adjusts their scores using that context. |
| HGB | Histogram gradient boosting: the tree-based classifiers and residual regressors used here, implemented by scikit-learn. |
| Residual correction | A regression estimate added to a selected event's initial time or coordinate. The uncorrected candidate key remains its identity. |
| Duplicate suppression / NMS | Selection of representatives among predictions close in time and space. NMS means non-maximum suppression. |
| MinIP | Minimum-intensity projection: at each image pixel, the smallest intensity over reconstructed depth planes. HoloD3 detects droplets in this projection. |
| Z-GMM | The axial smoother's two-component Gaussian mixture noise model, used to reduce the effect of depth outliers. |
| LWC | Liquid water content: total droplet mass per observation volume. |
| SD / CI / bootstrap | Standard deviation / confidence interval / estimation by repeated resampling. A CI for a mean and an SD of counts describe different quantities. |
| Parquet / CSV / NaN | A typed columnar table format / a plain-text table format / a missing floating-point measurement. Do not replace NaNs with zero when reproducing the trained models. |

The user commands are `collision`, `collision-2d` and `collision-physics`.
The Python import directory remains `ra3d` for compatibility with saved models and existing code; it is an implementation identifier, not an additional method to interpret.
Names such as `validation`, `static77`, `paper92` and `training_eligible` are preserved on disk so existing data remain traceable.
Their scientific meanings are explained in the [data guide](data.md) and [review dictionary](figures.md#5-visual-review-labels).
