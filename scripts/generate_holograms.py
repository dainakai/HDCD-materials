#!/usr/bin/env python3
"""Recreate full benchmark labels and paired holograms using bundled inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from generation_contract import validate_contract, label_command

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def command(args):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def validate_output(path):
    path = path.resolve()
    for protected in [ROOT / "data", ROOT / "src", ROOT / "reference", ROOT / "vendor", ROOT / "provenance"]:
        if path == protected or protected in path.parents:
            raise ValueError("Choose an output outside the bundled input directories")
    return path


def acquisition(output, seed, renderer):
    import pandas as pd
    import yaml
    split = "validation" if seed == 260745 else "train"
    source = ROOT / "data/acquisition" / split / f"seed{seed}" / "acquisition.yaml"
    config = yaml.safe_load(source.read_text())
    config["frames"] = "frames.csv"
    directories = {"primary": "01_bkgrem/C002H001S0011", "secondary": "01_bkgrem/C001H001S0011"}
    if renderer == "python":
        directories = {k: "holograms/" + k for k in directories}
    # A short render must not advertise the unrendered portion of the scene.
    # Preserve acquisition frame IDs even when the first available image is later than frame 0.
    rows = []
    for frame in range(6000):
        paths = {key: f"{directory}/{frame + 1:06d}.png" for key, directory in directories.items()}
        exists = [(output / value).is_file() for value in paths.values()]
        if any(exists) and not all(exists):
            raise ValueError(f"Incomplete rendered camera pair at frame {frame}")
        if all(exists):
            rows.append({"frame": frame, **paths})
    if not rows:
        raise ValueError("No rendered image pairs found for the acquisition manifest")
    pd.DataFrame(rows).to_csv(output / "frames.csv", index=False)
    (output / "acquisition.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    optics = config["optics"]
    upstream = {
        "schema_version": 1, "name": f"paper-seed-{seed}",
        "description": "Canonical paired synthetic holograms; no background subtraction or distortion.",
        "mode": "dual_phase_retrieval",
        "frames": {"primary_holograms": directories["primary"], "secondary_holograms": directories["secondary"]},
        "optics": {"wavelength_um": optics["wavelength_um"], "pixel_pitch_um": 10.0,
                   "image_size_px": 1024, "reconstruction_start_um": optics["reconstruction_start_um"],
                   "slice_spacing_um": 100.0, "slice_count": 1024,
                   "phase_retrieval_distance_um": optics["phase_distance_um"]},
        # Tracking measurement and collision-image features have distinct
        # historical reconstruction recipes; only their start plane is shared.
        "reconstruction": {"phase_retrieval_iterations": 3,
                           "fft_padding_side": 1536, "minip_slice_step": 2},
        "calibration": {"secondary_distortion_coefficients": None},
        "transforms": {"primary": [], "secondary": [], "minip": []},
    }
    (output / "holod3-acquisition.yaml").write_text(yaml.safe_dump(upstream, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default="260745", choices=["all"] + [str(s) for s in range(260740, 260746)])
    parser.add_argument("--output", type=Path, required=True, help="New parent directory; each seed gets a subdirectory")
    parser.add_argument("--stage", choices=["labels", "render", "all"], default="labels")
    parser.add_argument("--renderer", choices=["julia", "python"], default="julia")
    parser.add_argument("--device", default="cpu", help="Python renderer device; CPU works without CUDA")
    parser.add_argument("--julia-device", type=int, default=1, help="One-based visible CUDA device for original Julia renderer")
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=6000, help="Exclusive; labels always cover all 6000 frames")
    parser.add_argument("--published-labels", action="store_true", help="Render the bundled exact particle table without resimulating")
    parser.add_argument("--allow-unverified-environment", action="store_true",
                        help="Explicitly allow a different label-generation environment; verify labels afterward")
    args = parser.parse_args()
    if not 0 <= args.start_frame < args.end_frame <= 6000:
        parser.error("Require 0 <= start-frame < end-frame <= 6000")
    if args.published_labels and args.stage == "labels":
        parser.error("--published-labels is only for rendering")
    _, identity = validate_contract(allow_unverified_environment=(
        args.allow_unverified_environment or args.stage == "render" or args.published_labels))
    seeds = range(260740, 260746) if args.seed == "all" else [int(args.seed)]
    parent = validate_output(args.output)
    original = ROOT / "reference/generation"
    for seed in seeds:
        output = parent / f"seed{seed}"
        particles = output / "labels/particles.csv"
        if args.stage in {"labels", "all"} and not args.published_labels:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Label output is not empty: {output}")
            command(label_command(seed, output, sys.executable))
        if args.stage in {"render", "all"}:
            if args.published_labels:
                import pandas as pd
                if particles.exists():
                    raise FileExistsError(f"Choose a fresh render directory: {particles}")
                split = "validation" if seed == 260745 else "train"
                table = pd.read_parquet(ROOT / "data/labels" / split / f"seed{seed}" / "particles.parquet")
                particles.parent.mkdir(parents=True, exist_ok=True)
                table.to_csv(particles, index=False)
            if not particles.is_file():
                raise FileNotFoundError("Generate labels first, or pass --published-labels")
            (output / "metadata").mkdir(exist_ok=True)
            if args.renderer == "julia":
                command([args.julia, f"--project={original}", original / "render_holograms.jl",
                    f"--out-dir={output}", f"--device={args.julia_device}",
                    f"--frame-start={args.start_frame}", f"--frame-end={args.end_frame}"])
            else:
                # Use a script path so the bundled package is sufficient without an editable install.
                from ra3d.simulation.render import main as render
                render(["--particles", str(particles), "--output", str(output / "holograms"),
                        "--device", args.device, "--start-frame", str(args.start_frame),
                        "--end-frame", str(args.end_frame)])
            acquisition(output, seed, args.renderer)
        (output / "reproduction_request.json").write_text(json.dumps({
            "seed": seed, "stage": args.stage, "renderer": args.renderer,
            "labels": "published" if args.published_labels else "original_generator",
            "render_start": args.start_frame, "render_end_exclusive": args.end_frame,
            "label_frame_count": 6000, "event_count": 2000,
            "generation_identity": identity,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
