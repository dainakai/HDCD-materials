"""Assemble figure previews and compare panels with the paper's rendering hashes."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pypdf import PageObject, PdfReader, PdfWriter, Transformation

ROOT = Path(__file__).resolve().parents[2]


def check_renderer(executable):
    """Fail before numerical work if the optional system executable cannot run."""
    try:
        result = subprocess.run([executable, "-v"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Install Poppler (poppler-utils), or pass --pdftoppm /path/to/pdftoppm") from error
    return (result.stderr or result.stdout).splitlines()[0]


def render(pdf, png, executable, dpi):
    subprocess.run([executable, "-f", "1", "-singlefile", "-r", str(dpi), "-png", str(pdf), str(png.with_suffix(""))],
                   capture_output=True, check=True)
    with Image.open(png) as image:
        pixels = np.asarray(image.convert("RGB"))
    return {"width": pixels.shape[1], "height": pixels.shape[0], "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()}


def compose(panels, number, output):
    """Keep panels as vector PDF objects; standalone previews follow the paper's ordering."""
    target = output / f"figure{number:02d}.pdf"
    if len(panels) == 1:
        shutil.copyfile(panels[0], target)
        return target
    pages = [PdfReader(path).pages[0] for path in panels]
    page_width, margin, gap = 720.0, 8.0, 20.0
    widths = [0.45 * page_width, 0.48 * page_width] if number == 4 else [(page_width - 2 * margin - gap) / 2] * 2
    heights = [float(page.mediabox.height) * widths[i % 2] / float(page.mediabox.width) for i, page in enumerate(pages)]
    caption_height = 34.0 if number == 4 else 22.0
    row_heights = [max(heights[i:i + 2]) + caption_height for i in range(0, len(pages), 2)]
    height = sum(row_heights) + gap * (len(row_heights) - 1) + 2 * margin
    canvas = PageObject.create_blank_page(width=page_width, height=height)
    captions = []
    top = height - margin
    for row, row_height in enumerate(row_heights):
        for col in range(2):
            i = 2 * row + col
            page = pages[i]
            scale = widths[col] / float(page.mediabox.width)
            x = margin + (widths[0] + gap if col else 0)
            y = top - heights[i]
            canvas.merge_transformed_page(page, Transformation().scale(scale).translate(x, y))
            label = f"({chr(97 + i)})"
            if number == 4:
                label += [" Droplet number-density distribution", " Droplet count per hologram\nand liquid water content"][i]
            captions.append((x + widths[col] / 2, top - row_height + caption_height / 2, label))
        top -= row_height + gap
    # Caption text is also vector content. Full original LaTeX captions are supplied separately.
    fig = plt.figure(figsize=(page_width / 72, height / 72))
    for x, y, text in captions:
        fig.text(x / page_width, y / height, text, fontsize=11, ha="center", va="center", fontfamily="DejaVu Serif")
    buffer = BytesIO()
    fig.savefig(buffer, format="pdf", transparent=True)
    plt.close(fig)
    buffer.seek(0)
    canvas.merge_page(PdfReader(buffer).pages[0])
    writer = PdfWriter()
    writer.add_page(canvas)
    writer.add_metadata({"/Title": f"Figure {number}", "/Subject": "Recomputed panel assets; see paper-assets/figures.tex for original captions"})
    writer.write(target)
    return target


def assemble_and_verify(output, figures, *, verify, pdftoppm):
    reference = json.loads((ROOT / "provenance/paper_figures.json").read_text())
    report = {"renderer": check_renderer(pdftoppm), "reference_renderer": reference["renderer"],
              "dpi": reference["render_dpi"], "strict_pixels_requested": verify, "panels": []}
    gallery = []
    for number in figures:
        paths = []
        for panel in reference["panels"]:
            if panel["figure"] != number:
                continue
            pdf = output / f"figure{number:02d}" / Path(panel["paper_path"]).name
            page = PdfReader(pdf).pages[0]
            geometry = [float(page.mediabox.width), float(page.mediabox.height)]
            geometry_matches = bool(np.allclose(geometry, panel["page_points"], rtol=0, atol=1e-7))
            png = pdf.with_name(pdf.stem + "_paper_check.png")
            rendered = render(pdf, png, pdftoppm, reference["render_dpi"])
            exact = rendered == panel["render"]
            report["panels"].append({"figure": number, "file": str(pdf.relative_to(output)),
                "page_points": geometry, "page_geometry_matches": geometry_matches,
                "render": rendered, "reference_pixels_match": exact})
            mapped = output / "paper-assets" / panel["paper_path"]
            mapped.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pdf, mapped)
            paths.append(pdf)
        combined = compose(paths, number, output)
        render(combined, combined.with_suffix(".png"), pdftoppm, 110)
        gallery.append(f'<section><h2>Figure {number}</h2><a href="figure{number:02d}.pdf">Vector PDF</a>'
                       f'<img src="figure{number:02d}.png" alt="Figure {number}"></section>')
    shutil.copyfile(ROOT / "reference/figure_layouts.tex", output / "paper-assets/figures.tex")
    report["all_page_geometry_matches"] = all(row["page_geometry_matches"] for row in report["panels"])
    report["all_reference_pixels_match"] = all(row["reference_pixels_match"] for row in report["panels"])
    if not report["all_page_geometry_matches"] or (verify and not report["all_reference_pixels_match"]):
        report["status"] = "FAIL"
    else:
        report["status"] = "PASS" if report["all_reference_pixels_match"] else "DIFFERENT"
    (output / "rendering_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    title = "Reproduced paper figures"
    (output / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        f'<title>{title}</title><style>body{{max-width:1100px;margin:2rem auto;font:17px system-ui;padding:0 1rem}}'
        'img{display:block;max-width:100%;height:auto}section{margin:2rem 0;border-top:1px solid #ccc}</style>'
        f'<h1>{title}</h1>'
        '<p>Figures generated from the supplied numerical data. Each figure is available as a vector PDF. '
        'Individual panels and the paper’s English captions and LaTeX layout are in '
        '<a href="paper-assets/figures.tex">paper-assets/</a>.</p>'
        '<p><a href="numerical_verification.json">Numerical checks</a> record the calculation settings and results. '
        '<a href="rendering_verification.json">Rendering checks</a> compare panel dimensions and pixels '
        'with the paper figures. The combined PDFs below are previews without the full captions.</p>'
        + "\n".join(gallery) + '</html>')
    if report["status"] == "FAIL":
        raise RuntimeError(f"Figure rendering differs; inspect {output / 'rendering_verification.json'}. "
                           "Use the default --gravity-convention saved-figure to reproduce the paper. Exact pixel "
                           "checks also require the reference numerical/font/Poppler versions.")
    return report
