"""Select the exact Liberation font revision used by each original plot."""
from pathlib import Path

from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]


def register_fonts(version="2.1.5"):
    directory = ROOT / "reference/figure_fonts"
    if version == "1.07.4":
        directory /= "liberation-1.07.4"
    elif version != "2.1.5":
        raise ValueError(f"Unknown reference font revision: {version}")
    # Equal family/style scores otherwise let a machine's installed version win.
    manager = font_manager.fontManager
    manager.ttflist = [font for font in manager.ttflist if font.name not in {"Liberation Sans", "Liberation Mono"}]
    for path in sorted(directory.glob("*.ttf")):
        manager.addfont(path)
    manager._findfont_cached.cache_clear()
