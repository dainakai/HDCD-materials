# Third-party notices

The collision-detection software, trajectory adapters, simulation routines and scientific plotting code are authored by Dai Nakai and released under MIT. The collision-model weights in `data/models/` also use MIT. Particle-measurement and tracking routines are included in `reference/measurement/`, `src/holod3_linker/` and `src/holod3_smoother/`; their dependency versions are recorded in the environment files.

The particle detector uses the Ultralytics implementation, which retains its AGPL-3.0 license. TensorRT, CUDA, PyTorch and other installed dependencies retain their respective licenses. Installing this project does not relicense those components. The supplied detector and measurement checkpoints are used with those implementations.

The Julia renderer uses ParticleHolography 0.2.4. Its source and MIT license are retained under `reference/ParticleHolography/`. Three explanatory comments in that snapshot have been translated into English; numerical code is unchanged. The alternative Python renderer uses PyTorch. Trajectory smoothing uses the odd-reflection convention of the MIT-licensed SavitzkyGolay.jl package; the Python implementation computes coefficients with SciPy.

Liberation Sans and Mono 2.1.5 in `reference/figure_fonts/` retain the SIL Open Font License 1.1 and original notice in `LICENSE.txt`. The 1.07.4 fonts used in Figure 7 retain GPL version 2 with the document-embedding exception. Their notices, GPL text and corresponding build sources are in `reference/figure_fonts/liberation-1.07.4/`. The font licenses are distinct from the project's MIT license.

The Davis–Jonas–Hall collision-efficiency table and physical interpolation conventions derive from SCALE-SDM 0.2.5-2.2.2, archived at [Zenodo](https://doi.org/10.5281/zenodo.3841674). Copyright (c) 2012–2015, Team SCALE. The BSD-style two-clause redistribution notice is retained in `src/ra3d/config/physics/LICENSE-SCALE.txt`. Cite [Shima et al. (2020)](https://doi.org/10.5194/gmd-13-4107-2020) when using this component.

Numerical research data use CC BY 4.0, as stated in `LICENSE-DATA.md`. The project license does not replace any of the third-party notices above.
