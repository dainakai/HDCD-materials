"""Render vector figures with Poppler."""
import hashlib
import subprocess
import numpy as np
from PIL import Image

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
