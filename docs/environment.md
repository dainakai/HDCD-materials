# Environment setup

## CPU evaluation

The numerical reference environment is Linux x86-64, Python 3.12.12, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1, scikit-learn 1.7.1 and PyTorch 2.12.0. Use the exact versions in `requirements-core-reference.txt`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install uv==0.10.6
.venv/bin/python -m uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch==2.12.0
.venv/bin/python -m uv pip install --python .venv/bin/python \
  -r requirements-core-reference.txt -e .
source .venv/bin/activate
python scripts/check_environment.py
python scripts/prepare_data.py --profile test
```

Scikit-learn model deserialization checks its version and the checksums of all model files. `prepare_data.py` verifies each compressed part and the complete decoded file. It reuses matching files and refuses to replace a different existing file. All commands use paths relative to the checkout or the requested output directory.

## Hologram generation and particle measurement

Use an NVIDIA GPU with at least 16 GB VRAM and a CUDA-13-compatible driver. The reference computation used an RTX 4080 SUPER. Allow at least 32 GB RAM and 150 GB of disk space. A single GPU can run the complete pipeline; choose another visible device with `--gpu`.

Create the GPU environment with the full pinned requirements:

```bash
python3.12 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install uv==0.10.6
.venv-gpu/bin/python -m uv pip install --python .venv-gpu/bin/python \
  -r requirements-benchmark-reference.txt -e .
source .venv-gpu/bin/activate
python scripts/check_environment.py --profile benchmark --require-cuda
python scripts/prepare_data.py --profile measurement
```

The detector, depth estimators and diameter estimator are bundled as compressed objects. No measurement-model download is needed. The measurement backend uses PyTorch, CUDA Python, Ultralytics and TensorRT; the listed versions preserve its execution path.

Install **Julia 1.12.6** and make `julia` available on `PATH`. Instantiate both included Julia environments once:

```bash
julia --startup-file=no --project=reference/generation -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=reference/measurement -e 'using Pkg; Pkg.instantiate()'
```

The committed `Project.toml` and `Manifest.toml` files pin the renderer and trajectory smoother. Do not run `Pkg.update()` for a reference reproduction. Package installation requires network access; the scientific inputs, pretrained weights and generation distributions are supplied in the checkout.

Run the integration check before the complete benchmark:

```bash
python scripts/reproduce_benchmark.py --smoke --output runs/smoke --julia julia
python scripts/reproduce_benchmark.py --output runs/benchmark --julia julia
```

If using separate environments for CPU and GPU work, start with the CPU Python and pass `--gpu-python .venv-gpu/bin/python`. The child interpreter must contain the complete benchmark requirements. An absolute Julia path can be passed through `--julia`.

## Figures

Install the plotting requirements in the CPU environment, and install Poppler (`poppler-utils` on Debian/Ubuntu):

```bash
.venv/bin/python -m uv pip install --python .venv/bin/python -r requirements-figures-reference.txt
```

Fig.8 uses a separate Python 3.12 environment with Matplotlib 3.11.1 to preserve the established rendering:

```bash
python3.12 -m venv .venv-figures311
.venv-figures311/bin/python -m pip install uv==0.10.6
.venv-figures311/bin/python -m uv pip install --python .venv-figures311/bin/python \
  -r requirements-figures311-reference.txt
```

Then run:

```bash
.venv/bin/python scripts/reproduce_figures.py --output runs/figures
```

Supply a different plotting interpreter with `--figure8-python`. Other figures use Matplotlib 3.10.9. The fonts and composition functions are bundled. The reference raster checks use Poppler 22.02.0; another Poppler version can change antialiasing without changing numerical results.
