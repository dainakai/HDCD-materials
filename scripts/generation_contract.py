"""Validate the immutable inputs and reference environment of the six paper seeds."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "reference/generation/contract.json"


def validate_contract(*, allow_unverified_environment=False):
    contract = json.loads(CONTRACT.read_text())
    for name, expected in contract["source_sha256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Canonical generation input changed: {name}. Restore the release input.")
    versions = {"python_major_minor": ".".join(platform.python_version_tuple()[:2]),
                "numpy": np.__version__, "pandas": importlib.metadata.version("pandas")}
    matches = versions == contract["reference_environment"]
    if not matches and not allow_unverified_environment:
        raise ValueError(
            f"Generation requires the reference environment {contract['reference_environment']}; got {versions}. "
            "Install requirements-core-reference.txt, or explicitly select --allow-unverified-environment "
            "and compare the generated labels before use."
        )
    if type(np.random.default_rng(0).bit_generator).__name__ != contract["random_generator"]:
        raise ValueError("The NumPy default RNG algorithm differs from the published PCG64 contract")
    return contract, {"contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
                      "generation_inputs_verified": True, "environment": versions,
                      "environment_matches_reference": matches, "random_generator": contract["random_generator"]}


def label_command(seed, output, executable):
    contract = json.loads(CONTRACT.read_text())
    if seed not in contract["seeds"]:
        raise ValueError("The canonical recipe only covers the six published seeds")
    return [executable, str(ROOT / "reference/generation/generate_collision_benchmark.py"),
            "--project-root", str(ROOT), "--out-dir", str(output), "--seed", str(seed),
            *[arg.format(root=ROOT) for arg in contract["label_arguments"]]]


if __name__ == "__main__":
    _, evidence = validate_contract()
    print(json.dumps(evidence, indent=2))
