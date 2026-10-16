#!/usr/bin/env python3
"""Assemble regenerated figures and tables in the paper's English LaTeX layout."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, required=True, help="Output from reproduce_figures.py")
    parser.add_argument("--tables", type=Path, required=True, help="Output from reproduce_tables.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latex", type=Path, help="Optional LuaLaTeX executable; otherwise only write LaTeX")
    args = parser.parse_args()
    output = args.output.absolute()
    if output == ROOT or any(output.is_relative_to(ROOT / name)
                             for name in ("data", "src", "scripts", "reference", "vendor", "provenance")):
        parser.error("Use a new output directory outside release inputs")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Choose a new or empty output directory")
    figure_report = json.loads((args.figures / "verification.json").read_text())
    table_report = json.loads((args.tables / "verification.json").read_text())
    if figure_report["status"] != "PASS" or table_report["status"] != "PASS":
        parser.error("Figure and table verification must both report PASS")
    required = {"4", "5", "7", "8", "11", "12", "13", "14", "15"}
    if set(figure_report["figures"]) != required or not {"table3", "table4"} <= table_report.keys():
        parser.error("Generate all nine figures and both tables before assembling the preview")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.figures / "paper-assets/figures", output / "figures")
    layout = (args.figures / "paper-assets/figures.tex").read_text()
    preamble = layout.split(r"\begin{document}")[0]
    blocks = re.findall(r"\\setcounter\{figure\}\{(\d+)\}\s*(\\begin\{figure\}.*?\\end\{figure\})",
                        layout, re.S)
    parts = []
    for previous_number, block in blocks:
        number = int(previous_number) + 1
        (output / f"figure{number}.tex").write_text(block + "\n")
        parts.append(r"\setcounter{figure}{" + previous_number + "}\n" +
                     r"\input{figure" + str(number) + ".tex}\n")
        if number in (7, 8):
            table_number = number - 4
            shutil.copyfile(args.tables / f"table{table_number}.tex", output / f"table{table_number}.tex")
            parts.append(r"\setcounter{table}{" + str(table_number - 1) + "}\n" +
                         r"\input{table" + str(table_number) + ".tex}\n")
    preview = preamble + r"\begin{document}" + "\n" + ("\n" + r"\clearpage" + "\n").join(parts)
    preview += r"\end{document}" + "\n"
    (output / "preview.tex").write_text(preview)
    if args.latex:
        # Build files and font caches are run outputs, never part of paper assets.
        build = ROOT / "runs" / ("latex-" + output.name)
        build.mkdir(parents=True, exist_ok=True)
        executable = args.latex.absolute()  # Keep the name lualatex, not its luahbtex symlink target.
        environment = os.environ.copy()
        environment["PATH"] = str(executable.parent) + os.pathsep + environment.get("PATH", "")
        for variable, name in [("TEXMFVAR", "texmf-var"), ("TEXMFCACHE", "texmf-cache"),
                               ("TEXMFCONFIG", "texmf-config")]:
            environment[variable] = str(build / name)
        for iteration in (1, 2):
            with (build / f"compile-{iteration}.log").open("w") as log:
                subprocess.run([str(executable), "-interaction=nonstopmode", "-halt-on-error",
                                "-file-line-error", f"-output-directory={build}", "preview.tex"],
                               cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        from pypdf import PdfReader
        if len(PdfReader(build / "preview.pdf").pages) != 11:
            raise AssertionError("Expected 11 preview pages: nine figures and two tables")
        shutil.copyfile(build / "preview.pdf", output / "preview.pdf")
    print(json.dumps({"status": "PASS", "figures": len(blocks), "tables": 2,
                      "compiled": bool(args.latex), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
