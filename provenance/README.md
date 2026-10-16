# Provenance and verification

The root [manifest](../manifest.json) inventories distributed files. `data_sources/recipes.json` records preparation dependencies and output SHA-256 values. `reference/expected_outputs.json` and `DATASET_INDEX.csv` describe the selected prepared tables.

[Reduced-distribution verification](reduced_distribution_verification.json) records the completed preparation, inference, table and figure checks. Numerical outputs and panel comparisons use the same scientific routines as the complete materials.

Model-specific hashes, feature order and training settings remain in each `data/models/*/model.json`. `paper_figures.json` records original scientific/plotting function hashes and figure references. `review_cohort.json` identifies the top-80 review cohort. These provenance records describe the origin of included artifacts; they do not add omitted training or experimental replay inputs.
