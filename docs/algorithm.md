# Collision detection and evaluation

## Measurement and trajectories

The particle detector operates on the minimum-intensity projection. Fixed depth and diameter networks measure detected particles from phase-retrieved holograms. A two-stage assignment links detections; axial Gaussian-mixture smoothing and in-plane Savitzky–Golay smoothing provide trajectories. Raw observations generate candidates, and smoothed observations provide kinematic and reconstructed-image features.

Raw trajectory files retain one-based axial slice coordinates. Physical coordinates use `(slice - 1) * 100` µm, with the observation volume spanning 0–102,300 µm. In-plane coordinates are pixels; the pixel pitch is 10 µm. Frame IDs are zero-based and the frame rate is 4,000/s.

## Candidate labels and training

At candidate time t, identity supervision requires both parent observations at t−1 and the daughter observation at t+1. Missing observations are not interpolated. Identity-consistent candidates take precedence. For an event with an identity-consistent candidate, other nearby candidates are neutral and receive zero loss weight. If no identity-consistent candidate exists, spatial correspondence supplies the fallback positive set. Ours uses ±15,000 µm in depth for training correspondence; the two comparison methods omit depth.

Labels are recomputed within the candidate set actually used at each learning stage. This includes the top 20 candidates supplied to relation attention. Positive weights balance events across the training sequences; class balancing is applied separately. The test labels and experimental review labels are excluded from fitting.

Stage-one and static classifiers use scene-wise cross-fitting to prepare downstream training inputs. The final full-feature ensemble combines three gradient-boosting classifiers. Relation attention operates over each terminating track's retained candidate set. Localization regressors correct event time and position after selection. Random states and feature order are included in the code and model metadata.

Ours uses 77 trajectory features and 15 image features. Depth-free uses 49 trajectory features and 15 MinIP image features from the same measured tracks with depth removed. MinIP-based 2D uses an independent in-plane linking of projection detections and the same 64-feature family. See [the feature catalog](features.md).

## Ranking and duplicate suppression

Temporal modes are formed for each track pair, with at most five modes per terminating track. The lineage rank is fixed **before coordinate suppression**, and the lineage-dependent log-odds penalty uses that rank. Coordinate non-maximum suppression then removes nearby alternatives. The fixed rule is common to all three methods.

Table 5 is a scoring-stage comparison. Temporal-mode selection, the five-mode limit, the pre-suppression lineage penalty and trained localization regressors are common downstream operations. The trajectory row uses the static shortlist, while later rows use the scoring union. Missing image features remain missing in the trajectory row. The final row adds coordinate suppression. These rows therefore include candidate-selection and available-feature effects; they are not isolated interventions on classifier scores alone.

Adding image features increases AP from 0.892678 to 0.897689, while false positives at recall 0.8 increase from 177 to 183. Improvements are not uniform across every metric and stage.

## Event metrics

For each score threshold, predictions and truth events form a bipartite graph. Edges require a time difference of at most five frames and in-plane distance of at most 350 µm. The maximum one-to-one matching determines the number of true positives. No depth gate is used for the common evaluation.

Average precision is the integral of the all-points interpolated precision envelope, weighted by recall increments. A separate non-interpolated recall-weighted precision value is recorded by the full benchmark evaluator. Maximum F1 is taken over the score-threshold curve. The operating point at target recall is the first threshold attaining that recall; reported false positives are predictions minus matched events.

Particle detection in Fig.7 uses confidence-ranked nearest-unmatched correspondence within 8 pixels. Particle position errors use raw in-plane Hungarian assignment within the same gate, followed by errors calculated from the smoothed tracks used for collision features. These particle metrics and the collision metrics use the same 6,000-frame test sequence.
