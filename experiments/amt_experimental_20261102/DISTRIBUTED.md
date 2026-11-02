# Distributed experimental processing — 2026-11-02

The user authorized all stages on both local GPUs and daiPCatKIT, with GPU-dependent numerical differences recorded. Scene-level scheduling preserves the fixed scientific functions, model, acquisition parameters and top-80 review rule. Tracking still uses its original CPU implementation on each assigned machine. No model is retrained.

## Admission and identical code

`runtime_attestation.py` checks the actual imported modules, not a package label or the presence of a checkout:

- All 68 files in the frozen upstream production manifest, including code, configuration and weights.
- All collision-engine files and model payloads against their frozen manifests.
- The four scientific campaign scripts against `processing-source-identity.json`.
- The numerical MinIP helper and the common execution adapters.
- The actual `holo_linker`, `two_stage_lap` and `depth3d_online_link` import paths and SHA-256 hashes. Each must resolve inside the frozen production snapshot.
- Python build, PyTorch/CUDA/cuDNN, Torch-TensorRT, TensorRT, Triton and the imported Python dependency versions and entry-point hashes; Julia and the used Julia package versions and entry-point hashes.

The GPU Python installation was copied into a new private runtime on KIT, with the same fallback Python packages and required Julia package sources. Version/entry-point checks are not a claim that every shared-library byte was independently hashed. GPU model, UUID and NVIDIA driver are recorded separately and may differ. Worker paths also differ by host.

The common implementation is the **bundled fixed-lag Holo-Linker**, not a package resolved from an unrelated installed copy. The full local scene 12 and KIT smoke diagnostics have identical tracking arguments after excluding input/output paths: `linking_mode=fixed-lag`, `history_predictor=true`, `online_max_gap=12`, `gap_gate=1.2`, and all other parameters. The old `legacy` preset is not selected.

Holo-Linker source identities:

| File | SHA-256 |
| --- | --- |
| holo_linker.py | 2b4e727c89d6b7f85d7bd5b939ac856a11dfdf84b79e12f1dcc3d9427d7874f3 |
| two_stage_lap.py | c35429776c0b42bcf8e91175bc3ca7006e4496ffb6e23ffb62b5199c5d445c2a |
| depth3d_online_link.py | e2bbd0e3f0f214715a1edc523960392ab98c188fe3185afbf7fcd9e2128c2a8b |

The admitted runtimes use Python 3.12.12, PyTorch 2.12.0+cu130, CUDA 13.0, torchvision 0.27.0+cu130, Torch-TensorRT 2.12.1, TensorRT 10.16.1.11, Triton 3.7.0 and Julia 1.12.6. The existing measurement configuration selects TensorRT with FP16; its compiled GPU engines are hardware-specific and are not transferred between architectures. See private `distributed/{expected-runtime,local-gpu1-runtime,kit-runtime,code-equivalence}.json` for complete identities and absolute import paths.

Missing runtime identity, different imported code or a different dependency identity rejects the job. Every worker checks runtime identity at startup, then checks the scientific files, adapters, upstream files, numerical helper and fixed models at every stage boundary. The dispatcher also refuses changes to its active execution scripts. Do not edit those scripts during the campaign.

## Measured GPU differences

The controlled 128-frame probe is separate from production and human review. KIT completed all six stages successfully; strict numerical equivalence remains `DIFFERENT` and is not relabeled as equality.

| Check | Local GPU0 baseline | KIT RTX 5060 Ti |
| --- | ---: | ---: |
| Background image equality | reference | 256 / 256 bit-identical |
| Particle measurement rows | 12,848 | 12,847 |
| Tracks | 211 | 210 |
| Collision predictions | 94 | 92 |
| Fixed-measurement/tracking inference control | 94 | 94; same ordered candidates and top 80; max score difference 0 |

MinIP changed by at most one gray level in about 0.00617% of pixels. Reusing the exact GPU0 backgrounds and MinIP on KIT still changed one measured particle. Thus the downstream differences cannot be explained solely by rounded MinIP images. The fixed-input collision-inference control matched exactly. These are bounded observations for the probe, not a guarantee that full scenes have the same differences.

The local GPU1 probe completed background and MinIP. Measurement ran out of memory when an unrelated GPU1 job grew. Its partial measurement directory was archived under `distributed/probe-recovery/gpu1-oom-20261102/`, and completed stages were retained. The dispatcher waits for no compute processes and at least 10,000 MiB free for 60 seconds, resumes the missing probe stages with unchanged scientific parameters, then admits GPU1 only after successful completion. It never stops the unrelated job.

## Scheduling and transfer

`dispatch.py` owns the durable scene queue. A GPU receives the next unclaimed scene after completing its current one. A failed scene is reserved for inspection and is not silently restarted. Each local scene/stage is locked. Source/parameter identity is stored with stage receipts.

The handoff pauses only the old serial controller, allowing its current scene 13 measurement child to finish. After the PASS marker appears, the dispatcher retires that controller and resumes the remaining stages through the new worker. Completed stages and scene 12 remain intact.

`remote_task.py` copies only the assigned scene and necessary calibration to `/home/dai/amt-distributed-20261102`. It remaps host-dependent paths while keeping the input contract unchanged. All raw filenames/sizes and the original sampled content hashes are checked after rsync. This is not a claim of separately hashing every raw input.

After processing, every remote output file is SHA-256 checked after transfer. Import refuses to overwrite a different existing file. The local review manifest appears only after verification, with exactly 80 full-scene candidates. Task-owned remote raw copies and bulky results are cleaned only after the verified local import acknowledgment. The read-only source HDD is never cleaned or modified. Remote launch/status and output receipts remain available.

## Operational records

Paths below are relative to `../audit/revision-experimental-20261102/`:

- `campaign-state.json`: current state and all three workers; also exposed as a safe aggregate in the HTML report.
- `distributed/dispatcher-launch.json`, `dispatcher-process.json`, `dispatcher.log`: controller PID, command, code hashes and log.
- `distributed/legacy-handoff.json`: original controller and draining child identity.
- `distributed/assignments.json`: active/latest job per worker; prior job receipts and per-scene stage receipts preserve history.
- `distributed/kit-sceneNNNN.json`, `gpuN-sceneNNNN.json`: per-job state and completion.
- `scenes/sceneNNNN/stages/*.json`: scientific results and worker runtime identities.
- `scenes/sceneNNNN/remote-all-execution.json`: imported remote manifest, including the remote worker identity.
- `distributed/execution-policy.json`: exact user authorization and accepted hardware differences.

Distributed controller PID after the final runtime audit: **2317226** (replacing initial PID 2307386 without interrupting a scientific stage). Always check the live receipt and `/proc/PID/cmdline` before operating on a PID; identifiers can be reused.

To stop assigning new work, terminate only the verified dispatcher PID. Already launched workers continue. If `legacy-handoff.json` says `draining`, keep the legacy parent suspended until its child finishes and a replacement dispatcher completes the handoff; do not resume the old full serial queue while other scenes are assigned. To resume scheduling after confirming no dispatcher is alive, run:

```bash
.venv/bin/python -u experiments/amt_experimental_20261102/dispatch.py
```

Use a detached process with output directed to `distributed/dispatcher.log`. The dispatcher lock prevents a duplicate controller. Inspect failures and existing output receipts before changing assignments or retrying a job.

The existing report uses port **8769**, the protected review backend uses **8770**, and both retain the same Quick Tunnel:

https://messenger-recognition-forgotten-valium.trycloudflare.com

Its current server/tunnel/monitor PIDs, logs and review URL are in `../audit/revision-review-20261101/preview-processes.json`. `../audit/revision-review-20261101/stop_preview.py` stops only the preview processes, not scientific workers. A Quick Tunnel stops working when its process exits; recreating it changes the URL. Do not expose the private review key, whole repository or raw research data.

## Validation

Six contract tests check rejection of stale Holo-Linker, changed dependencies/adapters, missing identity, accepted host/hardware differences, and non-destructive imports. Python syntax and the existing HTML allowlist/link checks pass. The JavaScript renderer was exercised with published multi-worker data, including the GPU1 waiting state and Holo-Linker verification. HTTPS checks cover the changed report resources and live progress. Browser automation is disabled in the current Work settings, so these checks do not claim a rendered smartphone UI inspection.
