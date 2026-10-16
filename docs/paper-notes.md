# Analysis conventions

This page defines choices needed when reusing the paper's numerical inputs. For commands, start with [the reproducibility guide](paper-validation.md).

## Experimental review cohort and ambiguity

The released cohort is the top 80 candidates per sequence: 1,120 reviews, comprising 185 collisions, 896 non-collisions and 39 ambiguous cases. At 30 reviews per sequence, the 420 candidates include 177 collisions, 225 non-collisions and 18 ambiguous cases. Precision is `177/420 = 42.1%`. Ambiguous cases count toward review workload but are excluded from binary histograms and physical comparisons. [CSV dictionary](figures.md#5-visual-review-labels).

## Measurement and count settings

[Acquisition and reconstruction conditions](benchmark-reconstruction.md) specify the stage-specific optics, coordinates and Figure 7 inputs. [Count and rate calculations](physical-conventions.md) specify the default Figure 14–15 exposure, density resampling and uncertainty. The default figure command uses the settings for the paper figures.

## Training labels and test matching

The 3D training-label implementation uses ±5 frames, 350 µm in XY and 15,000 µm in Z before identity-based selection/fallback. Test evaluation uses only the time and XY gates with maximum one-to-one matching. Preserve the training association when refitting; evaluation matches are not interchangeable with training labels. See [labels.py](../src/ra3d/_core/labels.py) and [matching.py](../src/ra3d/_core/matching.py).

## Rank penalty and stage comparison

The terminating-track rank is computed among temporal-mode representatives before coordinate suppression and preserved afterward. Surviving ranks may therefore have gaps. The policy is in [policy.py](../src/ra3d/_core/policy.py).

Every Table 4 row uses temporal aggregation, at most five modes per terminating track, the rank penalty and the same fitted event-localization regressors. The final row adds coordinate suppression. The first row uses trajectory-only scoring with those shared event operations. [Table workflow](paper-validation.md#2-recalculate-the-performance-and-stage-comparison-tables).

## Depth-free comparison

The depth-free variant uses the same holographic measurements and trajectories as Ours, excludes explicit depth-related features, and adds MinIP image features. MinIP-based 2D measures and tracks droplets on MinIP images. Each comparison has its own trained model. Follow [the comparison guide](baseline2d.md) when reproducing or modifying it.

## Metrics and rounding

Figure 8(c) uses absolute false-positive counts. At recall 0.8 these are 67, 188 and 368 for Ours, the depth-free variant and MinIP-based 2D. Event AP uses an interpolated precision envelope; [the evaluation definition](paper-validation.md#evaluation-definitions) specifies the formula.

The LaTeX tables round metrics to three decimal places using half-up rounding. Table 3 reports differences in parentheses as method minus Ours, using the displayed values; for example, `0.901 - 0.917 = -0.016`. Full-precision operating points remain in the CSVs. Use those values for subsequent numerical analysis.
