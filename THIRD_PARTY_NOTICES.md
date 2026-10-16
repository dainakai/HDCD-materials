# Third-party notices

Collision-detection code is authored by Dai Nakai and released under MIT. Numerical kernels were ported from the author's research code, with source hashes and symbol provenance in `src/ra3d/_core/provenance.json` and `two_d_provenance.json`. The Python synthetic label generator and half-mixture allocation preserve the author's original implementation; their provenance is in `src/ra3d/simulation/provenance.json`.

HoloD3, HoloD3-Linker and HoloD3-Smoother were used to obtain the original particle measurements. Their source snapshots, upstream measurement weights and crop-training datasets are not part of this reduced distribution. The included trained collision models use MIT.

The Python Z-GMM implementation follows the author's `zsmooth_new.jl`. Its Savitzky–Golay endpoint behavior follows the odd-reflection convention in [SavitzkyGolay.jl](https://github.com/lnacquaroli/SavitzkyGolay.jl), an MIT-licensed package; numerical coefficients are computed using SciPy. Hologram propagation follows the angular-spectrum formulation used by ParticleHolography.jl. This deposit includes the original Julia rendering script and ParticleHolography 0.2.4 source under `reference/`, with the upstream license preserved. The alternative Python renderer uses PyTorch.

Direct dependencies include NumPy, pandas, SciPy, scikit-learn, PyTorch, Pillow, PyArrow, PyYAML, Hugging Face Hub, Flask, threadpoolctl, zstandard, skops, safetensors, and Matplotlib. Each dependency retains its own license. The local review page is implemented in this project and does not load JavaScript or CSS from a CDN.

Figure reproduction additionally uses Polars and pypdf, retaining their own licenses.
The author's original figure/scientific functions are identified by source hashes in `provenance/paper_figures.json`.
Liberation Sans and Mono 2.1.5 in `reference/figure_fonts/` retain the SIL Open Font License 1.1 and original notice in `LICENSE.txt`.
The 1.07.4 fonts needed for Figure 7(a–d) retain GPL version 2 with the original document-embedding exception; their notices, GPL text and corresponding build sources are in `reference/figure_fonts/liberation-1.07.4/`.
These are distinct font releases with distinct licenses; neither is relicensed under this project's MIT license.
The upstream [Liberation 1.07 project](https://github.com/liberationfonts/liberation-1.7-fonts) identifies the source release and its license.

The numerical data in this distribution use CC BY 4.0, as stated in `LICENSE-DATA.md`. Third-party software retains the licenses above.
## SCALE-SDM physical tables and kernel conventions

The Davis–Jonas–Hall collision-efficiency table and associated physical interpolation conventions derive from SCALE-SDM 0.2.5-2.2.2, archived at https://doi.org/10.5281/zenodo.3841674. Copyright (c) 2012–2015, Team SCALE. The original BSD-style two-clause redistribution notice is retained in `src/ra3d/config/physics/LICENSE-SCALE.txt` and included in installed distributions. Cite Shima et al. (2020), https://doi.org/10.5194/gmd-13-4107-2020, when using this component. This project's MIT license does not replace that notice.
