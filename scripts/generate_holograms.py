#!/usr/bin/env python3
"""Recreate full benchmark labels and paired holograms using bundled inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from generation_contract import validate_contract, label_command

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def command(args):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True,
                   env={**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'PYTHONHASHSEED': '0'})


def validate_output(path):
    path = path.resolve()
    for protected in [ROOT / "data", ROOT / "src", ROOT / "reference", ROOT / "vendor", ROOT / "provenance"]:
        if path == protected or protected in path.parents:
            raise ValueError("Choose an output outside the bundled input directories")
    return path


def acquisition(output, seed, renderer, *, planned=False):
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
        if not planned and any(exists) and not all(exists):
            raise ValueError(f"Incomplete rendered camera pair at frame {frame}")
        if planned or all(exists):
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
        # Measurement and collision-image features use different padding sizes.
        "reconstruction": {"phase_retrieval_iterations": 3,
                           "fft_padding_side": 1536, "minip_slice_step": 2},
        "calibration": {"secondary_distortion_coefficients": None},
        "transforms": {"primary": [], "secondary": [], "minip": []},
    }
    (output / "holod3-acquisition.yaml").write_text(yaml.safe_dump(upstream, sort_keys=False))


def verify_labels(output, seed):
    from prepare_data import digest
    contract = json.loads((ROOT / 'reference/generation/contract.json').read_text())
    checks = {}
    for name, expected in contract['labels_sha256'][str(seed)].items():
        path = output / 'labels' / (name + '.csv')
        actual = digest(path)
        if actual != expected:
            raise ValueError(f'Generated {name} differs for seed {seed}: {actual}. '
                             'Use the pinned generation environment; do not continue to rendering or fitting.')
        checks[name] = actual
    (output / 'labels/verified.json').write_text(json.dumps(checks, indent=2) + '\n')
    return checks


def verify_images(output, seed, start, end):
    from prepare_data import digest
    frames = json.loads((ROOT / 'reference/generation/render-sha256.json').read_text())[str(seed)]
    for frame in range(start, end):
        for name, expected in frames[str(frame)].items():
            if digest(output / name) != expected:
                raise ValueError(f'Hologram differs for seed {seed}, frame {frame}: {name}')


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
    parser.add_argument("--allow-unverified-environment", action="store_true",
                        help="Explicitly allow a different label-generation environment; verify labels afterward")
    args = parser.parse_args()
    if not 0 <= args.start_frame < args.end_frame <= 6000:
        parser.error("Require 0 <= start-frame < end-frame <= 6000")
    _, identity = validate_contract(allow_unverified_environment=(
        args.allow_unverified_environment or args.stage == "render"))
    seeds = range(260740, 260746) if args.seed == "all" else [int(args.seed)]
    parent = validate_output(args.output)
    original = ROOT / "reference/generation"
    for seed in seeds:
        output = parent / f"seed{seed}"
        particles = output / "labels/particles.csv"
        if args.stage in {"labels", "all"}:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Label output is not empty: {output}")
            command(label_command(seed, output, sys.executable))
            verify_labels(output, seed)
            acquisition(output, seed, args.renderer, planned=True)
        if args.stage in {"render", "all"}:
            if not particles.is_file():
                raise FileNotFoundError("Generate labels first with --stage labels")
            verify_labels(output, seed)
            (output / "metadata").mkdir(exist_ok=True)
            if args.renderer == "julia":
                command([args.julia, f"--project={original}", original / "render_holograms.jl",
                    f"--out-dir={output}", f"--device={args.julia_device}",
                    f"--frame-start={args.start_frame}", f"--frame-end={args.end_frame}"])
                verify_images(output, seed, args.start_frame, args.end_frame)
            else:
                # Use a script path so the bundled package is sufficient without an editable install.
                from ra3d.simulation.render import main as render
                render(["--particles", str(particles), "--output", str(output / "holograms"),
                        "--device", args.device, "--start-frame", str(args.start_frame),
                        "--end-frame", str(args.end_frame)])
            acquisition(output, seed, args.renderer)
        (output / "reproduction_request.json").write_text(json.dumps({
            "seed": seed, "stage": args.stage, "renderer": args.renderer,
            "labels": "ra3d.simulation.generate",
            "render_start": args.start_frame, "render_end_exclusive": args.end_frame,
            "label_frame_count": 6000, "event_count": 2000,
            "generation_identity": identity,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
