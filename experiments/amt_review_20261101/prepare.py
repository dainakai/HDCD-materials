"""Read-only inventory and review preparation; never fit models or move scientific data."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
OUT = WORKSPACE / "audit/revision-review-20261101"
PREFLIGHT = REPO / "experiments/amt_preflight_20261030"
RESULTS = PREFLIGHT / "results"
EXPERIMENT = Path("/media/dai-server/DATAM2/siderepos/260804/ra3dv1-experimental-260715/inference")
SCENES = [12, 13, 15, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def table(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preparation_table(path, rows, columns=None):
    """Do not erase annotations or correspondence entered after preparation."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns or list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    expected = buffer.getvalue().encode()
    path = Path(path)
    if path.exists():
        if path.read_bytes() != expected:
            raise ValueError(f"Preserving modified preparation table; use a new version directory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)


def image_files(path):
    path = Path(path)
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".bmp", ".png", ".tif", ".tiff"})


def main():
    evidence = OUT / "evidence"
    private = OUT / "experimental-preparation"
    evidence.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    manifest = read(RESULTS / "manifest.json")
    errors = [name for name, expected in manifest["files"].items() if sha(RESULTS / name) != expected]
    assert not errors, errors
    verification = read(RESULTS / "verification.json")
    assert verification["status"] == "PASS" and len(verification["arms"]) == 7
    metrics = list(csv.DictReader((RESULTS / "primary_metrics.csv").open()))
    assert len(metrics) == 14
    indexed = {(r["arm"], r["rank"]): r for r in metrics}
    contrasts = []
    for label, a, b, rank_a, rank_b in [
        ("位相回復12→3（旧規則）", "legacy_phase12_z15", "legacy_phase3_z15", "before", "before"),
        ("位相回復12→3（ID統一）", "strict_phase12_z15", "strict_phase3_z15", "before", "before"),
        ("旧規則→ID統一（3回）", "legacy_phase3_z15", "strict_phase3_z15", "before", "before"),
        ("抑制前順位→抑制後順位", "strict_phase3_z15", "strict_phase3_z15", "before", "after"),
        ("Z±15→±5 mm", "strict_phase3_z15", "strict_phase3_z5", "before", "before"),
        ("Z±15→±30 mm", "strict_phase3_z15", "strict_phase3_z30", "before", "before"),
        ("Z±15→なし", "strict_phase3_z15", "strict_phase3_znone", "before", "before"),
    ]:
        x, y = indexed[a, rank_a], indexed[b, rank_b]
        contrasts.append({"change": label, "from_arm": a, "to_arm": b, "rank_from": rank_a, "rank_to": rank_b,
                          "ap_from": float(x["event_ap"]), "ap_to": float(y["event_ap"]),
                          "delta_ap": float(y["event_ap"]) - float(x["event_ap"]),
                          "fp80_from": int(x["fp_r0.8"]), "fp80_to": int(y["fp_r0.8"]),
                          "fp90_from": int(x["fp_r0.9"]), "fp90_to": int(y["fp_r0.9"])})
    table(evidence / "controlled_deltas.csv", contrasts)
    generation = []
    for seed in range(260740, 260746):
        root = WORKSPACE / f"audit/revision-preflight-20261030/generation/seed{seed}"
        frames = list(csv.DictReader((root / "frames.csv").open()))
        assert len(frames) == 6000
        pairs = sum(all((root / row[k]).is_file() for k in ["primary", "secondary"]) for row in frames)
        generation.append({"seed": seed, "planned_pairs": len(frames), "rendered_pairs": pairs,
                           "frames_manifest_sha256": sha(root / "frames.csv")})
    table(evidence / "synthetic_render_inventory.csv", generation)

    labels_path = REPO / "data/paper_figures/review/visual_review_labels.csv"
    labels = list(csv.DictReader(labels_path.open()))
    assert len(labels) == 1120
    assert sorted({int(x["scene_num"]) for x in labels}) == SCENES
    counts = Counter(x["status"] for x in labels)
    assert counts == {"collision": 185, "non_collision": 896, "ambiguous": 39}
    rows, baseline, private_scenes = [], [], []
    for n in SCENES:
        source = EXPERIMENT / f"C000H001S{n:04d}"
        command_path = source / "image_feature_command.json"
        command = read(command_path)["command"]
        track_dir = Path(command[command.index("--track-dir") + 1])
        run = track_dir.parents[1]
        proc_path = run / "minip/proc_summary.json"
        proc = read(proc_path)
        full = read(run / "full_pipeline_summary.json")
        frames = read(source / "prepared_data_summary.json")["frame_contract"]["cameras"]
        sources = []
        for camera in frames:
            raw = Path(camera["directory"])
            old_bkg = run / "data/01_bkgrem" / raw.name
            all_bkg = source / "prepared_data/01_bkgrem" / raw.name
            raws, retained, processed = image_files(raw), image_files(old_bkg), image_files(all_bkg)
            sources.append({"camera": camera["camera"], "raw_directory": str(raw), "raw_files": len(raws),
                            "retained_directory": str(old_bkg), "retained_files": len(retained),
                            "former_prepared_directory": str(all_bkg), "former_prepared_files": len(processed),
                            "expected_frames": 10917})
        subset = [x for x in labels if int(x["scene_num"]) == n]
        tally = Counter(x["status"] for x in subset)
        for x in subset:
            baseline.append({"baseline_version": "v1.0.1-historical-top80",
                             "historical_key": f"260715:S{n:04d}:{x['event_id']}",
                             "scene_num": n, "event_id": x["event_id"], "candidate_frame_raw": x["candidate_frame_raw"],
                             "vanished_track_id": x["vanished_track_id"], "survivor_track_id": x["survivor_track_id"],
                             "old_rank": x["review_score_rank"], "old_score": x["score"], "old_label": x["status"],
                             "new_candidate_key": "", "new_label": "", "match_status": "unmatched",
                             "reuse_requires_identity_and_image_verification": True})
        rows.append({"scene_num": n, "reviewed": len(subset), "collision": tally["collision"],
                     "non_collision": tally["non_collision"], "ambiguous": tally["ambiguous"],
                     "raw_C001": sources[0]["raw_files"], "raw_C002": sources[1]["raw_files"],
                     "retained_C001": sources[0]["retained_files"], "retained_C002": sources[1]["retained_files"],
                     "historical_minip_phase": proc["priter"], "historical_front_um": proc["z0_um"],
                     "requested_front_um": 74900, "front_shift_um": 74900 - proc["z0_um"],
                     "historical_tracks_exist": track_dir.is_dir()})
        scene = {"scene_num": n, "sources": sources, "historical_track_directory": str(track_dir),
                 "historical_minip": proc, "historical_minip_summary_sha256": sha(proc_path),
                 "calibration_scene": full["calibration_scene"], "calibration_source": full["calibration_resolution"]["directory"],
                 "calibration_source_exists": Path(full["calibration_resolution"]["directory"]).is_dir(),
                 "image_feature_command_sha256": sha(command_path), "pipeline_frame_to_raw_index": 1,
                 "requested_front_um": 74900, "historical_front_um": proc["z0_um"],
                 "phase_iterations_target": 3, "measurement_padding_target": 1536, "collision_padding_target": 2048,
                 "output_model_fingerprint": None, "output_predictions_fingerprint": None,
                 "readiness": "pending_final_synthetic_model_and_full_images"}
        private_scenes.append(scene)
        save(private / f"scenes/scene{n:04d}.json", scene)
    preparation_table(private / "historical_review_crosswalk.csv", baseline)
    table(evidence / "experimental_inventory.csv", rows)
    preparation_table(private / "new_review_annotations.csv", [], ["run_id", "model_sha256", "prediction_sha256", "scene_num",
        "candidate_id", "candidate_frame_raw", "review_rank", "label", "note", "reviewed_at", "asset_sha256",
        "historical_match_key", "historical_match_verified"])
    save(private / "review_contract.json", {
        "status": "PREPARED_NOT_STARTED", "historical_cohort_sha256": sha(labels_path),
        "historical_counts": dict(counts), "historical_reviews": 1120,
        "target_scenes": SCENES, "planned_new_budget_per_scene": 80, "planned_new_budget_is_not_completed": True,
        "label_values": ["collision", "non_collision", "ambiguous"],
        "record_identity": ["run_id", "model_sha256", "prediction_sha256", "scene_num", "candidate_id"],
        "historical_match_policy": "No rank-based transfer. Physical-event correspondence plus image verification; retain old and new IDs and labels separately.",
        "blinding": "Hide historical labels during new review; unblind only for adjudication of correspondence/disagreements.",
        "main_cohort": "Frozen new-model top 80 per scene; ties deterministic; no replacing difficult or ambiguous candidates.",
        "additional_review": "Old-only events and optional lower-score samples are separate strata with recorded inclusion probabilities; never mix their denominator into top-80 precision.",
        "experimental_training": "No training on these review labels during the synthetic-only experimental evaluation.",
        "metrics": ["confirmed/all-reviewed precision with ambiguous count", "binary-only precision as separate secondary metric",
                    "top-N confirmed-event coverage conditional on reviewed cohort, not absolute recall", "old/new matched-event overlap and rank changes",
                    "diameter/velocity consistency with valid sample counts", "Poisson gravity comparison using full frames and nominal volume"],
        "generation_distribution_policy": "Freeze the existing empirical distribution as historical generation input. Recomputing experimental statistics must not silently regenerate synthetic training data.",
        "launch_prerequisites": ["Final synthetic release and all three methods validated", "Final model/code/config fingerprints frozen",
                                "Full synchronized images and calibration validated", "New predictions and review image assets complete",
                                "New review store bound to model and predictions; old annotations preserved"],
        "notes": "This preparation neither imports old labels as new labels nor starts an experimental review."
    })
    save(evidence / "readiness.json", {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "scientific_source_commit": "c7110c85c9e3f711566d2631ecf9be692d8ecefe",
        "paper_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE / "CollisionPaper", text=True).strip(),
        "paper_source_sha256": sha(WORKSPACE / "CollisionPaper/本番フォーマット/template.tex"),
        "verified_portable_artifacts": len(manifest["files"]), "artifact_hash_errors": errors,
        "completed_controlled_models": len(verification["arms"]), "controlled_comparison_verification": "PASS",
        "revised_truth_sequences": 6, "revised_paired_hologram_frames": sum(x["rendered_pairs"] for x in generation),
        "required_paired_hologram_frames": 36000, "paper_replacement_ready": False, "experimental_review_ready": False,
        "remaining_synthetic": ["paired holograms", "endpoint-inclusive MinIP", "new measurements and tracks", "all-stage labels and upstream selection",
                                "three separately trained methods", "Fig7-9 and both numerical performance tables", "method text and independent-test wording"],
        "experimental_scenes": len(rows), "historical_review_counts": dict(counts),
        "full_raw_camera_directories": sum(x["raw_files"] == 10917 for r in private_scenes for x in r["sources"]),
        "expected_raw_camera_directories": 28,
        "historical_minip_phase_iterations": sorted({r["historical_minip_phase"] for r in rows}),
        "requested_front_differs_from_historical_scenes": [r["scene_num"] for r in rows if r["front_shift_um"]],
        "public_scope": "Report, relevant plots and aggregate evidence only; no raw images, model weights or event annotations."
    })
    print(json.dumps(read(evidence / "readiness.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
