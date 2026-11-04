#!/usr/bin/env python3
"""Recalculate and draw Figures 4-5 and 7-15 with the supplied panel layouts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--figure8-python', type=Path, default=REPO / '.venv-figures311/bin/python')
    p.add_argument('--pdftoppm', default='pdftoppm')
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Choose an empty output directory')
    if not a.figure8_python.is_file():
        p.error('Install the Figure 8 plotting environment (docs/environment.md#figures)')
    from figure_support.delivery import check_renderer
    check_renderer(a.pdftoppm)
    a.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(a.output.resolve() / 'matplotlib-cache'))
    from reproduce_experimental import calculate
    from figure_support import experimental, synthetic_particles, synthetic_detection, synthetic_examples
    from pypdf import PdfWriter
    calculate(a.output)
    experimental.main(a.output)
    synthetic_particles.main(REPO / 'data/figure7', a.output / 'figure7-panels', a.output / 'figures')
    synthetic_detection.main(a.output, a.figure8_python)
    synthetic_examples.main(a.output)
    writer = PdfWriter()
    preview = a.output / 'preview'
    preview.mkdir()
    numbers = [4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for n in numbers:
        source = a.output / f'figures/fig{n:02d}.pdf'
        subprocess.run([a.pdftoppm, '-singlefile', '-scale-to', '1800', '-png', str(source), str(preview / f'fig{n:02d}')], check=True)
        writer.append(source, outline_item=f'Figure {n}')
    writer.write(a.output / 'figures.pdf')
    (a.output / 'figures.json').write_text(json.dumps({'status': 'PASS', 'figures': numbers}, indent=2) + '\n')
    print('Created Figures 4-5 and 7-15 and the combined PDF.')


if __name__ == '__main__':
    main()
