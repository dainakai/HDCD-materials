# Paper figures and tables

[View all eleven figures](figures.pdf). Individual figures are named `fig04.pdf`, `fig05.pdf`, and `fig07.pdf` through `fig15.pdf`; PNG previews use the same stems. English captions are in `captions/`. `table4.tex` compares the three detection methods and `table5.tex` compares scoring stages.

The source commands and figure-to-data mapping are in [the figure guide](../docs/figures.md). Table values are recalculated with:

```bash
python scripts/reproduce_tables.py --output runs/tables
```

The supplied LaTeX tables retain the manuscript's labels and rounding. The generated CSV tables retain full precision; generated LaTeX files provide a standalone tabular representation of those values.
