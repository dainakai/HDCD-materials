"""Particle-table adapter for the included trajectory linker."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .io import read_table, write_table, write_json, sha256
from .config import Units
from .tracks import read_tracks

LINKER_COMMIT = "fbe935a5d7fcd6b0c04ad0b2b2b9786c23f08cdf"


def link(particles, output, *, pixel_pitch_um=10.0, slice_spacing_um=100.0, extra_args=()):
    Units(pixel_pitch_um=pixel_pitch_um, slice_spacing_um=slice_spacing_um)
    try:
        from holod3_linker import link_holod3_output
    except ImportError as exc:
        raise RuntimeError('Install the project: pip install -e .') from exc
    particles = Path(particles).resolve()
    # The upstream adapter accepts CSV or a pipeline directory. Preserve Parquet
    # values when adapting a selectively downloaded detection table to that API.
    with tempfile.TemporaryDirectory(prefix="ra3d-link-input-") as temporary:
        source = particles
        if particles.suffix == ".parquet":
            source = Path(temporary) / "particles_3d.csv"
            write_table(read_table(particles), source)
        result = link_holod3_output(
            source, output_dir=output, pixel_pitch_um=pixel_pitch_um, linker_args=extra_args
        )
    table = read_tracks(result.trajectory_dir)
    # Link costs use physical Z, but trajectory exports retain the input slice.
    # Therefore slice spacing must not rescale the exported coordinates.
    write_table(table, Path(output) / "trajectory3d.parquet")
    write_json(
        {
            "source": str(particles),
            "sha256": sha256(particles) if particles.is_file() else None,
            "pixel_pitch_um": pixel_pitch_um,
            "slice_spacing_um": slice_spacing_um,
            "linker_commit": LINKER_COMMIT,
        },
        Path(output) / "ra3d_link.json",
    )
    return result
