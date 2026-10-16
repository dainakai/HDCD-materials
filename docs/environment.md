# Set up the environment

The documented workflows use Linux, Python 3.12 and CPU execution. Windows and macOS were not tested. Run commands from the repository root.

## Core environment

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install uv==0.10.6
.venv/bin/python -m uv pip install --python .venv/bin/python \
  -r requirements-core-reference.txt -e .
source .venv/bin/activate
python scripts/check_environment.py --profile core
```

The checker should report `PASS`. `cuda_available: false` is valid: all documented workflows run on a CPU. Keep the pinned numerical versions when comparing with the reference.

If Python 3.12 is unavailable, an optional bootstrap using an existing Python installation is:

```bash
python3 -m venv runs/setup-tools
runs/setup-tools/bin/python -m pip install uv==0.10.6
runs/setup-tools/bin/uv python install 3.12.12
runs/setup-tools/bin/uv venv --python 3.12.12 --seed .venv
```

Then skip the first core command and continue with the installation commands. Interpreter and package installation need network access. Data preparation and evaluation use included inputs and need no data-service account or download.

## Prepare the data

```bash
python scripts/prepare_data.py --profile all --output runs/data-preparation.json
python scripts/verify_release.py --tables
```

Allow 6 GB of additional free disk space for the prepared data and temporary trajectory/simulation files. Environments require separate space. Several GB of RAM are needed; 8 GB of available RAM is a practical starting point. [Preparation profiles](data-preparation.md) let you prepare only evaluation or figure inputs.

## Figure environments

The [figure guide](figures.md#1-set-up-the-two-plotting-environments) adds two pinned plotting environments and Poppler. The second environment is for plotting only. No GPU or TeX installation is required for the Python figure workflow.

Write results under `runs/`, which is excluded from the release inventory and Git. Use a new output directory for each evaluation or figure run.
