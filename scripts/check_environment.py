#!/usr/bin/env python3
"""Check the installed numerical environment before following a reproducibility scenario."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["core"], default="core")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    expected = dict(line.split("==") for line in (ROOT / "requirements-core-reference.txt").read_text().splitlines()
                    if line and not line.startswith("#"))
    required = ["ra3d"]
    installed, errors = {}, []
    for name in [*expected, *required]:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing package: {name}")
            continue
        if name in expected and installed[name].split("+")[0] != expected[name]:
            errors.append(f"{name}: expected {expected[name]}, installed {installed[name]}")
    if sys.version_info[:2] != (3, 12):
        errors.append("Use Python 3.12")
    import torch
    cuda = torch.cuda.is_available()
    if args.require_cuda and not cuda:
        errors.append("CUDA is not available to PyTorch; check the GPU driver or select the CPU route")
    report = {"status": "FAIL" if errors else "PASS", "profile": args.profile,
              "python": sys.version.split()[0], "packages": installed, "cuda_available": cuda, "errors": errors}
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
