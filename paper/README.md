# Paper figures and tables

These figures and tables are generated from the included numerical data, using the paper's design and English captions.

[Open the 11-page preview](preview.pdf): Figures 4, 5, 7, 8 and 11–15, plus Tables 3–4, using the paper's English captions and LaTeX layout. Figure 9 is outside this workflow.

- `figures/` contains the 18 regenerated PDF panels at the original manuscript paths.
- `figure4.tex`, `figure5.tex`, `figure7.tex`, `figure8.tex`, `figure11.tex` through `figure15.tex` are the corresponding inclusion blocks.
- [Table 3](table3.tex) and [Table 4](table4.tex) contain freshly computed values in the original table layouts.
- [preview.tex](preview.tex) assembles these parts and retains the paper's section and citation numbers.

## Regenerate

From the release root, after [environment setup](../docs/figures.md#1-set-up-the-two-plotting-environments):

```bash
python scripts/reproduce_tables.py --output runs/paper-tables
python scripts/reproduce_figures.py --output runs/paper-figures --verify-rendered
python scripts/assemble_paper_preview.py \
  --figures runs/paper-figures --tables runs/paper-tables \
  --output runs/paper-preview --latex /path/to/lualatex
```

Omit `--latex` to generate the source layout without compiling. LuaLaTeX needs the packages in `reference/figure_layouts.tex`, including the original `ltjsarticle` class; the public captions and reader instructions are English. Numerical reproduction and the HTML gallery do not require TeX. Use new output directories for each run.

All panel page dimensions and 150 dpi rendered pixels were checked against the reference figures using Poppler 22.02.0. PDF bytes may differ through metadata. All 45 displayed numerical table values match the references. Table 3 compares Ours, the depth-free variant and MinIP-based 2D; its parenthetical differences are calculated from the displayed rounded method values, while CSVs retain full precision.

[Verification records](../provenance/README.md) · [Archive guide](../README.md)
