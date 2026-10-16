"""Regression checks for the release's partial-image and deterministic-generation entry points."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from ra3d.acquisition import Acquisition

ROOT = Path(__file__).resolve().parents[1]


def script_module(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_render_manifest_preserves_nonzero_frames_and_rejects_unpaired_images(tmp_path, monkeypatch):
    generation = script_module("generate_holograms", monkeypatch)
    for camera in ["C002H001S0011", "C001H001S0011"]:
        directory = tmp_path / "01_bkgrem" / camera
        directory.mkdir(parents=True)
        for filename in ["000101.png", "000102.png"]:
            (directory / filename).touch()
    generation.acquisition(tmp_path, 260745, "julia")
    acquisition = Acquisition.load(tmp_path / "acquisition.yaml")
    assert acquisition.frames.index.tolist() == [100, 101]
    assert acquisition.validate()["frames"] == 2
    # Both workflows share the recorded physical front, but not the optical
    # iteration/padding recipe. Confusing these silently changes measurements.
    upstream = yaml.safe_load((tmp_path / "holod3-acquisition.yaml").read_text())
    collision = yaml.safe_load((tmp_path / "acquisition.yaml").read_text())
    assert upstream["optics"]["reconstruction_start_um"] == collision["optics"]["reconstruction_start_um"] == 64900
    assert upstream["reconstruction"]["phase_retrieval_iterations"] == 3
    assert upstream["reconstruction"]["fft_padding_side"] == 1536
    assert collision["optics"]["phase_iterations"] == 12
    assert collision["optics"]["padding_px"] == 2048
    (tmp_path / "01_bkgrem/C001H001S0011/000102.png").unlink()
    with pytest.raises(ValueError, match="Incomplete rendered camera pair at frame 101"):
        generation.acquisition(tmp_path, 260745, "julia")


def test_generation_contract_rejects_changed_inputs_even_with_environment_override(tmp_path, monkeypatch):
    contract = script_module("generation_contract", monkeypatch)
    source = tmp_path / "distribution.csv"
    source.write_text("diameter,density\n25,1\n")
    definition = {"source_sha256": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}}
    source.write_text("diameter,density\n25,2\n")
    contract_file = tmp_path / "contract.json"
    contract_file.write_text(json.dumps(definition))
    monkeypatch.setattr(contract, "ROOT", tmp_path)
    monkeypatch.setattr(contract, "CONTRACT", contract_file)
    with pytest.raises(ValueError, match="Canonical generation input changed"):
        contract.validate_contract(allow_unverified_environment=True)


def test_generation_contract_requires_explicit_environment_override(tmp_path, monkeypatch):
    contract = script_module("generation_contract", monkeypatch)
    definition = {"source_sha256": {}, "random_generator": "PCG64",
                  "reference_environment": {"python_major_minor": "0.0", "numpy": "0.0", "pandas": "0.0"}}
    contract_file = tmp_path / "contract.json"
    contract_file.write_text(json.dumps(definition))
    monkeypatch.setattr(contract, "CONTRACT", contract_file)
    with pytest.raises(ValueError, match="requires the reference environment"):
        contract.validate_contract()
    _, evidence = contract.validate_contract(allow_unverified_environment=True)
    assert not evidence["environment_matches_reference"]
