# AMT revision preflight, 2026-10-30

This experiment isolates image phase retrieval, training association rules, training Z gates, and the timing of the lineage-rank penalty.
It is a sensitivity study on fixed upstream data, not a new released benchmark.
The [report](../../docs/revision-preflight-20261030.md) distinguishes verified changes from the subsequent full regeneration.

## Inputs and split

- Five training sequences: seeds 260740–260744.
- Independent test: seed 260745, 6,000 frames and 2,000 collisions.
- Existing 74.9 mm-front particle measurements: phase retrieval 3, FFT padding 1536.
- Collision image features: phase retrieval 12 versus 3, FFT padding 2048.
- Original 102.4 mm truth and holograms are retained within the controlled comparisons.
- Candidate membership, upstream scores, and image-feature selection are fixed across arms.
- The proposed 102.3 mm generation is checked separately and must not be combined with those original images.

`protocol.json` was saved before fitting the comparison arms.
`revision-profile.json` records the proposed new release configuration; it is not a release manifest.
The `validation/` directory in the source archive is the independent test split, never a fitting or parameter-selection split.

## Comparisons

The primary test matcher uses maximum one-to-one association within ±5 frames and 350 µm in XY, without Z.
AP integrates the precision envelope.
All operating-point counts use the highest threshold achieving the specified recall.

The identity-consistent rule observes both parents at candidate time `t-1` and the daughter at `t+1`, without filling missing observations.
It applies the same identity-first/fallback rule to classifiers and relation attention, excludes neutral spatial candidates from loss, and assigns equal aggregate positive/negative weight across the training scenes.
The top-20 relation inputs are relabeled within the actual attention input set.
The legacy comparison keeps the earlier identity windows, spatial relation labels, and relation weighting.
Thus the legacy-to-consistent contrast evaluates a coherent labeling correction, not one isolated subcomponent.

Training Z gates are 5, 15, 30 mm and absent.
These are absolute tolerances (a 5 mm gate means ±5 mm), not full window widths.
Candidate Z uses the physical `(slice-1)*100` origin during association, while the fixed input features keep their recorded representation.
Secondary test Z gates are 2.5, 5, 10, 15, 30 mm and absent; they describe different evaluation criteria and do not replace the common primary metric.

For R4, both rankings use exactly the same surviving predictions and corrected coordinates.
Only the rank-dependent score changes.
The paired bootstrap uses six consecutive 1,000-frame blocks, omits five boundary frames on either side, and draws 2,000 paired block samples with seed 20261030.
Its interval is conditional on this one test sequence and these fitted models.

## Reproduction on the verification host

Run from the repository root.
`common.py` records the external input paths and confines new outputs to the sibling `audit/revision-preflight-20261030/` directory.
Those external archives and paired holograms are required and are not included in this Git branch.
The Fig.7 script reuses the previously verified particle-detection curve under `audit/results/published-figures/figure07/`.
The gravity script compares freshly calculated numbers with `audit/results/manuscript-gravity/` and copies its already rendered PNGs after that numerical check.
These prior-audit references are also required to rerun those two scripts exactly on this host; their final review outputs are included in `results/`.
`frozen_engine/` preserves the exact experimental engine derived from the preceding manuscript-alignment audit, including the stricter label rule.
It is installed into the audit directory by `prepare_local.py`, which refuses to overwrite different files.
The inherited module description of XY/time-only eligibility in `strict_labels.py` describes its no-Z case; its explicit `z_tolerance_um` argument governs the gated arms.

```bash
PY=.venv/bin/python
GPU_PY=/media/dai-server/M2Disk2TB/repos/260712/.venv/bin/python
$PY experiments/amt_preflight_20261030/prepare_local.py
$PY experiments/amt_preflight_20261030/geometry.py
for seed in 260740 260741 260742 260743 260744 260745; do
  $PY experiments/amt_preflight_20261030/generate_revision.py --seed "$seed"
done
$PY experiments/amt_preflight_20261030/verify_generation.py --seeds 260740 260741 260742 260743 260744 260745
$GPU_PY experiments/amt_preflight_20261030/images.py --seeds 260740 260741 260742 260743 260744 260745 --device cuda:0
$GPU_PY experiments/amt_preflight_20261030/train_compare.py --phase 12 --z-mm 15 --rule strict --device cuda:0
$GPU_PY experiments/amt_preflight_20261030/train_compare.py --phase 12 --z-mm 15 --rule legacy --device cuda:0
$GPU_PY experiments/amt_preflight_20261030/queue_train.py
$PY experiments/amt_preflight_20261030/fig7.py
$PY experiments/amt_preflight_20261030/gravity.py
$PY experiments/amt_preflight_20261030/verify_results.py
$PY experiments/amt_preflight_20261030/rank_bootstrap.py
$PY experiments/amt_preflight_20261030/collect_results.py
```

`queue_train.py` starts the five independent phase-3 fits after the five training feature tables pass verification.
Test feature extraction can finish independently; evaluation waits for its verified completion marker.
They use four CPU threads each and share GPUs 0 and 1; the verification host has 28 logical CPUs and 125 GB RAM.
The image tasks may be distributed across GPUs using non-overlapping seed lists.
`images.py` resumes its frame checkpoints; `train_compare.py` resumes verified model checkpoints.
`generate_revision.py` refuses to overwrite an already generated scientific output.
Run the generation loop in a fresh audit directory; on this existing verification workspace, retain the generated sequences and use `verify_generation.py` instead.
Image resumption time in `images3.json` is not an end-to-end speed benchmark.
The Z-gate-free phase-3 fit resumed after an unrelated process exhausted GPU 0 memory.
It retained its CPU tree checkpoints and ran the remaining work with `CUDA_VISIBLE_DEVICES=1` and logical `--device cuda:0`; all seven fits and evaluations completed.
Completion durations can refer to resumed work and are not cross-arm runtime comparisons.

The portable `results/` directory contains scalar metrics, input hashes, label counts, verification results, and PNG figures.
Full image features, model weights, predictions, and revised particle truth remain in the audit directory.
The main repository's historical release inputs and weights are preserved.
