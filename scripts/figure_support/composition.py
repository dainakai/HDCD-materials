"""Compose vector panels with the manuscript dimensions and labels."""
from io import BytesIO
from pathlib import Path
import json
import os
import re
import shutil
import subprocess

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ArrayObject, ByteStringObject, ContentStream, TextStringObject
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LABEL_SIZE_PT = 8.0  # PDF points at the size used in template.tex, not source-PDF points.
TEXT_WIDTH_PT = 177 * 72 / 25.4
TEXT_HEIGHT_PT = (660 - (37 - 11.4) * 72.27 / 25.4) * 72 / 72.27


def register_label_font():
    candidates = [
        str(HERE.parents[1] / "reference/figure_fonts/LiberationSans-Regular.ttf"),
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            pdfmetrics.registerFont(TTFont("PanelLabel", candidate))
            return
    raise FileNotFoundError("Set AMT_LABEL_FONT to an Arial-compatible regular TTF font.")


def display_scale(width, height, width_fraction):
    return min(TEXT_WIDTH_PT * width_fraction / width, TEXT_HEIGHT_PT * 0.78 / height)


def remove_panel_text(page, reader, prefix_only=False):
    """Remove existing panel captions, or only their prefixes, from PDF text.

    Edits text-showing operators instead of covering text with white rectangles;
    all plot paths and embedded images are preserved.
    """
    stream = ContentStream(page.get_contents(), reader)
    removed = []
    for operands, operator in stream.operations:
        if operator not in (b"TJ", b"Tj"):
            continue
        parts = operands[0] if operator == b"TJ" else [operands[0]]
        text = "".join(str(part) for part in parts if isinstance(part, TextStringObject))
        match = re.match(r"^\(([a-f])\)\s*", text)
        if not match:
            continue
        removed.append(match.group(1))
        if not prefix_only:
            operands[0] = ArrayObject() if operator == b"TJ" else TextStringObject("")
            continue
        remaining = len(match.group())
        kept = ArrayObject()
        for part in parts:
            if remaining and isinstance(part, TextStringObject):
                take = min(remaining, len(part))
                remaining -= take
                if take < len(part):
                    raw = part.original_bytes
                    stride = 2 if b"\x00" in raw else 1
                    kept.append(ByteStringObject(raw[take * stride:]))
            elif not remaining:
                kept.append(part)
        operands[0] = kept if operator == b"TJ" else kept[0]
    page.replace_contents(stream)
    return removed


def panel_label(index, subtitles):
    text = f"({chr(97 + index)})"
    if subtitles:
        text += " " + subtitles[index]
    return text


def relabel(source, target, centres, width_fraction, subtitles=None):
    reader = PdfReader(ROOT / source)
    writer = PdfWriter()
    page = writer.add_page(reader.pages[0])
    found = remove_panel_text(page, writer)
    assert found == [chr(97 + i) for i in range(len(centres))], (source, found)
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    font_size = LABEL_SIZE_PT / display_scale(width, height, width_fraction)
    labels = BytesIO()
    overlay = canvas.Canvas(labels, pagesize=(width, height))
    overlay.setFont("PanelLabel", font_size)
    for i, (x, baseline) in enumerate(centres):
        overlay.drawCentredString(x, baseline, panel_label(i, subtitles))
    overlay.save()
    page.merge_page(PdfReader(labels).pages[0])
    writer.write(target)


def combine(sources, target, columns, width_fraction, strip_inner_labels=False, subtitles=None, vertical_align="top", column_widths=None, panel_y_offsets=None):
    """Place existing PDF pages and panel labels without rasterizing plots."""
    if vertical_align not in ("top", "bottom"):
        raise ValueError(f"Unsupported panel alignment: {vertical_align}")
    readers = [PdfReader(ROOT / source) for source in sources]
    # Clone pages into writers so content-stream edits are owned by a writer.
    owners = [PdfWriter() for _ in readers]
    pages = [owner.add_page(reader.pages[0]) for owner, reader in zip(owners, readers)]
    if strip_inner_labels:
        for page, owner in zip(pages, owners):
            assert remove_panel_text(page, owner, prefix_only=True) == list("abcdef")
    gap, label_band = 16.0, 24.0
    widths = column_widths or [360.0] * columns
    offsets = panel_y_offsets or [0.0] * len(pages)
    if len(widths) != columns or len(offsets) != len(pages) or min(widths) <= 0 or min(offsets) < 0:
        raise ValueError("Invalid panel widths or vertical offsets")
    heights = [float(p.mediabox.height) * widths[i % columns] / float(p.mediabox.width) for i, p in enumerate(pages)]
    width = sum(widths) + (columns - 1) * gap
    for _ in range(12):
        row_heights = [max(heights[j] + offsets[j] for j in range(i, min(i + columns, len(pages)))) + label_band for i in range(0, len(pages), columns)]
        height = sum(row_heights) + (len(row_heights) - 1) * gap
        font_size = LABEL_SIZE_PT / display_scale(width, height, width_fraction)
        label_band = font_size * 2
    writer = PdfWriter()
    result = writer.add_blank_page(width=width, height=height)
    labels = BytesIO()
    overlay = canvas.Canvas(labels, pagesize=(width, height))
    overlay.setFont("PanelLabel", font_size)
    for i, page in enumerate(pages):
        row, col = divmod(i, columns)
        top = height - sum(row_heights[:row]) - gap * row
        cell_width = widths[col]
        x = sum(widths[:col]) + col * gap
        y = top - row_heights[row] + label_band if vertical_align == "bottom" else top - heights[i]
        y += offsets[i]
        scale = cell_width / float(page.mediabox.width)
        transform = Transformation().translate(-float(page.mediabox.left), -float(page.mediabox.bottom)).scale(scale).translate(x, y)
        result.merge_transformed_page(page, transform)
        overlay.drawCentredString(x + cell_width / 2, top - row_heights[row] + font_size * 0.5, panel_label(i, subtitles))
    overlay.save()
    result.merge_page(PdfReader(labels).pages[0])
    writer.write(target)
