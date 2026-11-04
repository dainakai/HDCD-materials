# Environment setup

## CPU evaluation

The numerical reference environment is Linux x86-64, Python 3.12.12, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1, scikit-learn 1.7.1 and PyTorch 2.12.0. These versions specify the reproducibility environment. The supplied models, evaluation tables and figure calculations run without a GPU.

Start with Git and curl installed, and clone the repository as shown in the [README](../README.md). Install uv and the exact Python version below; skip the first three commands if that setup is already complete. uv installs Python separately from the operating system's Python. Run all commands from the repository root.

```bash
curl -LsSf https://astral.sh/uv/0.10.6/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12.12
uv venv --python 3.12.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch==2.12.0
uv pip install --python .venv/bin/python \
  -r requirements-core-reference.txt -e .
source .venv/bin/activate
python scripts/check_environment.py
python scripts/prepare_data.py --profile test
```

Scikit-learn model deserialization checks its version and the checksums of all model files. `prepare_data.py` verifies each compressed part and the complete decoded file. It reuses matching files and refuses to replace a different existing file. All commands use paths relative to the checkout or the requested output directory.

## Hologram generation and particle measurement

The reference renderer and particle-measurement backend use one NVIDIA GPU. The pinned CUDA 13 / TensorRT 10.16 stack requires a compatible driver and GPU architecture: CUDA 13 uses driver series R580 or newer, and TensorRT supports compute capability 7.5 or newer. The reference computation used an RTX 4080 SUPER with driver 595.84. These are the tested hardware and software, not a 16 GB VRAM minimum. See [resource usage](#resource-usage) for measured memory use. Select another visible device with `--gpu`. Driver and architecture compatibility are described in NVIDIA's [CUDA compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) and [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html).

Create the GPU environment with the full pinned requirements:

```bash
uv venv --python 3.12.12 .venv-gpu
uv pip install --python .venv-gpu/bin/python \
  -r requirements-benchmark-reference.txt -e .
source .venv-gpu/bin/activate
python scripts/check_environment.py --profile benchmark --require-cuda
python scripts/prepare_data.py --profile measurement
```

The detector, depth estimators and diameter estimator are bundled as compressed objects. No measurement-model download is needed. The measurement backend uses PyTorch, CUDA Python, Ultralytics and TensorRT; the listed versions preserve its execution path.

Install **Julia 1.12.6** with Juliaup:

```bash
curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.12.6
export PATH="$HOME/.juliaup/bin:$PATH"
julia --version
```

For an existing Juliaup installation, use `juliaup add 1.12.6` and `juliaup default 1.12.6` instead of running the installer. Instantiate both included Julia environments once:

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

## Resource usage

The following GPU memory peaks were sampled with `nvidia-smi` during the 128-frame integration check on the reference GPU. They include process context, workspace and allocator caches. TensorRT engines were compiled during the measurement. The image-feature check used 32 candidates per method.

| Operation | Observed GPU memory |
|---|---:|
| Julia hologram generation | 0.37 GiB |
| Minimum-intensity projection | 0.51 GiB |
| Particle detection, batch size 8 | 13.81 GiB with abundant free memory |
| Same particle detection with about 8 GiB free | 3.26 GiB |
| Depth and diameter estimation | 1.63 GiB |
| Collision image features | 0.52 GiB |

The second detection run used a temporary allocation in an independent process to leave 7.96 GiB free. It retained the same batch sizes, thresholds and model weights, and completed particle measurement, tracking and feature calculation. The higher unrestricted peak therefore does not establish a 16 GB requirement. This check does not establish a universal minimum GPU capacity or qualify the full benchmark on every 8 GB card. Run the integration check on the intended GPU before starting all six sequences.

Changing the available workspace changed the detection count from 26,288 to 26,287 in this check. cuDNN algorithm selection and floating-point rounding can affect detections near a threshold. Successful execution alone does not establish numerical equivalence; the complete benchmark writes a comparison with the published results. The generated particle labels and holograms remain subject to their exact SHA-256 checks.

The complete CPU evaluation of all three supplied models peaked at 2.37 GiB of process RAM. The 128-frame integration check peaked at 4.06 GiB across its process tree; these figures exclude the operating system and other applications. Full candidate generation and model fitting can require more RAM. CPU feature jobs run sequentially by default; `--cpu-jobs 3` increases concurrency and memory use. No fixed 32 GB RAM minimum has been established.

The completed six-sequence run stored approximately 100.3 GiB of generated inputs and results: 44.7 GiB of holograms and generation labels, 21.6 GiB of particle measurements, 15.1 GiB of MinIP images, 2.4 GiB of tracks, 16.2 GiB of feature/truth tables and 0.3 GiB of fitted models. Allow additional space for environments, downloads and caches; roughly 150 GB free is a practical full-run budget. CPU evaluation of the supplied data uses much less disk space. Detailed observations and measurement scope are recorded in [resource-usage.json](../reference/resource-usage.json).

## Figures

Install the plotting requirements in the CPU environment, and install Poppler (`poppler-utils` on Debian/Ubuntu):

```bash
uv pip install --python .venv/bin/python -r requirements-figures-reference.txt
```

Fig.8 uses a separate Python 3.12 environment with Matplotlib 3.11.1 to preserve the established rendering:

```bash
uv venv --python 3.12.12 .venv-figures311
uv pip install --python .venv-figures311/bin/python \
  -r requirements-figures311-reference.txt
```

Then run:

```bash
.venv/bin/python scripts/reproduce_figures.py --output runs/figures
```

Supply a different plotting interpreter with `--figure8-python`. Other figures use Matplotlib 3.10.9. The fonts and composition functions are bundled. The reference raster checks use Poppler 22.02.0; another Poppler version can change antialiasing without changing numerical results.
