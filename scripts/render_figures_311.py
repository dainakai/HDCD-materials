#!/usr/bin/env python3
"""Plot the panels originally made with Matplotlib 3.11.1 (called by reproduce_figures.py)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import pandas as pd

from figure_support.fonts import register_fonts

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/paper_figures"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, choices=[7, 8, 12], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if mpl.__version__ != "3.11.1":
        parser.error("This plotting stage requires Matplotlib 3.11.1; see requirements-figures311-reference.txt")
    output = args.output
    mpl.rcdefaults()
    register_fonts()
    if args.figure == 7:
        from figure_support import particle_density as density, trajectory_length as length
        density.SOURCE, density.OUTPUT = DATA / "fig07/number_density_by_diameter_5um.csv", output
        density.main()
        mpl.rcdefaults()
        length.SOURCE, length.OUTPUT = DATA / "fig07", output
        length.main()
    elif args.figure == 8:
        from figure_support import collision_plot as plot
        curves = pd.read_csv(output / "event_pr.csv", float_precision="round_trip")
        summaries = pd.read_csv(output / "event_summary.csv", float_precision="round_trip")
        targets = pd.read_csv(output / "target_recall.csv", float_precision="round_trip")
        composition = pd.read_csv(output / "fp_composition.csv")
        plot.OUTPUT = output
        plot.configure_matplotlib()
        plot.plot_precision_recall(curves, summaries)
        plot.plot_topn(curves)
        plot.plot_fp_burden(targets)
        plot.plot_fp_composition(composition)
    else:
        from figure_support import review_budget
        curve = pd.read_csv(output / "review_curve.csv", float_precision="round_trip")
        operating = pd.read_csv(output / "review_operating_points.csv", float_precision="round_trip")
        suggested = int(operating.loc[operating.suggested, "reviews_per_scene"].iloc[0])
        review_budget.plot_figure(curve, operating, suggested, output / "review_budget_operating_points_260715")


if __name__ == "__main__":
    main()
