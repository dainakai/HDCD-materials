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
    # padding recipe. Confusing these silently changes measurements.
    upstream = yaml.safe_load((tmp_path / "holod3-acquisition.yaml").read_text())
    collision = yaml.safe_load((tmp_path / "acquisition.yaml").read_text())
    assert upstream["optics"]["reconstruction_start_um"] == collision["optics"]["reconstruction_start_um"] == 74900
    assert upstream["reconstruction"]["phase_retrieval_iterations"] == 3
    assert upstream["reconstruction"]["fft_padding_side"] == 1536
    assert collision["optics"]["phase_iterations"] == 3
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


def test_source_contracts_reference_only_distributed_files():
    distributed = {entry['path'] for entry in json.loads((ROOT / 'manifest.json').read_text())['files']}
    for filename, field, prefix in [
        ('reference/engine-manifest.json', 'files', ''),
        ('reference/generation/contract.json', 'source_sha256', ''),
        ('reference/measurement/manifest.json', 'files', 'reference/measurement/'),
    ]:
        for name, expected in json.loads((ROOT / filename).read_text())[field].items():
            relative = prefix + name
            assert relative in distributed, f'Undistributed source in {filename}: {relative}'
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_acquisition_defaults_match_the_explicit_benchmark_optics(tmp_path):
    (tmp_path / 'primary.png').touch()
    (tmp_path / 'secondary.png').touch()
    (tmp_path / 'frames.csv').write_text('frame,primary,secondary\n0,primary.png,secondary.png\n')
    config = {'schema_version': 1, 'units': {}, 'frames': 'frames.csv'}
    (tmp_path / 'acquisition.yaml').write_text(yaml.safe_dump(config))
    actual = Acquisition.load(tmp_path / 'acquisition.yaml')
    expected = yaml.safe_load((ROOT / 'data/acquisition/validation/seed260745/acquisition.yaml').read_text())
    for field, value in expected['optics'].items():
        assert getattr(actual, field) == value


@pytest.mark.parametrize('jobs', [None, 2])
def test_benchmark_cpu_concurrency_keeps_gpu_work_sequential(tmp_path, monkeypatch, jobs):
    import sys
    benchmark = script_module('reproduce_benchmark', monkeypatch)
    commands = []
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    arguments = ['reproduce_benchmark.py', '--output', str(tmp_path),
                 '--from-stage', 'features', '--through-stage', 'features', '--julia', sys.executable]
    if jobs is not None:
        arguments += ['--cpu-jobs', str(jobs)]
    monkeypatch.setattr(sys, 'argv', arguments)
    benchmark.main()
    assert len(commands) == 12
    for command in commands:
        method = command[command.index('--method') + 1]
        image_gpu = command[2] == 'images' and method == 'ours'
        assert command[command.index('--jobs') + 1] == str(1 if image_gpu else (jobs or 1))
        assert command[command.index('--device') + 1] == ('cuda:0' if image_gpu else 'cpu')


@pytest.mark.parametrize('jobs', ['0', '-1'])
def test_benchmark_rejects_nonpositive_cpu_concurrency(monkeypatch, jobs):
    import sys
    benchmark = script_module('reproduce_benchmark', monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['reproduce_benchmark.py', '--cpu-jobs', jobs])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


def test_full_benchmark_compares_results_and_records_comparison_failure(tmp_path, monkeypatch):
    import subprocess
    import sys
    benchmark = script_module('reproduce_benchmark', monkeypatch)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if Path(command[1]).name == 'verify_benchmark_results.py':
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(benchmark.subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['reproduce_benchmark.py', '--output', str(tmp_path),
                                     '--from-stage', 'evaluate', '--julia', sys.executable])
    with pytest.raises(subprocess.CalledProcessError):
        benchmark.main()
    assert len(commands) == 4
    assert Path(commands[-1][1]).name == 'verify_benchmark_results.py'
    state = json.loads((tmp_path / 'state.json').read_text())
    assert state['status'] == 'FAIL' and state['stage'] == 'evaluate'


@pytest.mark.parametrize('change', [None, 'metric', 'prediction', 'missing'])
def test_complete_run_reference_comparison_detects_scientific_differences(tmp_path, monkeypatch, change):
    import shutil
    import pandas as pd
    verification = script_module('verify_benchmark_results', monkeypatch)
    for method in verification.METHODS:
        directory = tmp_path / 'results' / method
        directory.mkdir(parents=True)
        for name in ['metrics.json', 'predictions.parquet']:
            shutil.copyfile(ROOT / 'reference/evaluation' / method / name, directory / name)
    target = tmp_path / 'results/ours'
    if change == 'metric':
        metric = json.loads((target / 'metrics.json').read_text())
        metric['operating_at_recall_0.80']['false_positive_events'] += 1
        (target / 'metrics.json').write_text(json.dumps(metric))
    elif change == 'prediction':
        frame = pd.read_parquet(target / 'predictions.parquet')
        frame.loc[0, 'vanished_track_id'] += 1
        frame.to_parquet(target / 'predictions.parquet')
    elif change == 'missing':
        (target / 'predictions.parquet').unlink()
    report = verification.verify(tmp_path)
    assert report['status'] == ('PASS' if change is None else 'MISMATCH')
    assert report['methods']['ours']['status'] == (
        'PASS' if change is None else 'MISSING' if change == 'missing' else 'MISMATCH')
