# Verification scope

The supported workflows operate on test seed 260745 and measured numerical inputs for Figures 4, 5, 7, 8 and 11–15. The [verification record](../provenance/reduced_distribution_verification.json) records the checks performed for this distribution.

## Data preparation

The entire 566,261-row 3D shortlist is recalculated from the supplied trajectories and candidate-generation fields. Recorded bitwise differences recover the exact reference serialization. Prepared original tables must match their original SHA-256, including row order, missing values and Parquet metadata. The test particle-state simulation is also checked against the original full-file hash.

The two comparison methods' final feature tables, image-derived measurements and figure measurements are retained inputs. Their recovery is verified independently of calculating the 3D trajectory features. The omitted training and supplementary data cannot be recovered by this preparation command.

## Inference and evaluation

All three supplied model bundles are applied to the full test scoring sets. Event identities, coordinates and corrected frames are compared with reference predictions. Score comparison allows 1e-5 absolute difference; coordinate comparison allows 1e-7 µm. Event AP must agree with the recorded reference within 1e-12. [Evaluation definitions](paper-validation.md#evaluation-definitions).

The tolerance on prediction scores reflects the existing model replay contract. It does not relax the exact checksum check on prepared inputs.

## Tables and figures

Tables 3–4 are recalculated and all 45 displayed numerical values are checked. All nine figures are recalculated. Their 18 panels are compared with reference page dimensions and rendered pixels at 150 dpi using the specified plotting environments and Poppler 22.02.0. [Figure setup](figures.md).

## Scope limits

No retraining or image-to-track run is performed by these workflows. Five-seed training data, supplementary seed 260913, full experimental sequence replay inputs and measurement weights are outside the distribution. Original images and independent visual relabeling are also outside its scope.
