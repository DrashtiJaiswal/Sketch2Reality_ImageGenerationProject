"""
preprocess.py - validation and clean-up of the uploaded sketch.

Why this matters (AI pipeline step 1):
Sketch-conditioned models read the *lines* of the control image. Grey paper,
shadows from a phone photo, faint pencil strokes or a transparent background
all weaken that signal. So we normalise every upload into the same form:
dark strokes on clean white paper, a size the API accepts, and a safe aspect
ratio. Only Pillow + NumPy are used (no OpenCV needed).
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MIN_LONG_SIDE = 512          # tiny uploads are enlarged to this (API needs >= 64 px per side)
MAX_ASPECT = 2.5             # Stability Control Sketch accepts 1:2.5 .. 2.5:1
SAFE_ASPECT = 2.4            # pad to this when an image is too wide/tall
MAX_PIXELS_INPUT = 60_000_000
MAX_PAYLOAD_BYTES = 9 * 1024 * 1024   # API limit is 10 MiB for the whole request


class InvalidImageError(ValueError):
    """Raised with a user-friendly message when the upload cannot be used."""


@dataclass
class PreparedSketch:
    image: Image.Image                 # RGB, cleaned, API-ready
    png_bytes: bytes
    original_size: tuple[int, int]
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 1. Validation + loading
# --------------------------------------------------------------------------- #
def load_and_validate(path: str | os.PathLike, max_upload_mb: int = 10) -> Image.Image:
    if not path:
        raise InvalidImageError("No sketch uploaded yet. Drop an image into the sketch box first.")
    p = Path(path)
    if not p.is_file():
        raise InvalidImageError("The uploaded file could not be found. Please upload it again.")
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise InvalidImageError(
            f"Unsupported file type '{p.suffix}'. Use PNG, JPG, WEBP, BMP, GIF or TIFF."
        )
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > max_upload_mb:
        raise InvalidImageError(
            f"The file is {size_mb:.1f} MB; the limit is {max_upload_mb} MB. Please use a smaller image."
        )
    try:
        with Image.open(p) as probe:
            probe.verify()                      # detects truncated / corrupt files
        img = Image.open(p)
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise InvalidImageError("This file is not a valid image (it may be corrupt). Try exporting it again as PNG or JPG.")
    w, h = img.size
    if w * h > MAX_PIXELS_INPUT:
        raise InvalidImageError("The image has too many pixels. Please downscale it to under ~60 megapixels.")
    if min(w, h) < 32:
        raise InvalidImageError(f"The image is only {w}x{h}px - too small to contain a usable sketch.")
    return ImageOps.exif_transpose(img)         # honour phone-camera rotation


# --------------------------------------------------------------------------- #
# 2. Normalisation helpers
# --------------------------------------------------------------------------- #
def _flatten_alpha(img: Image.Image) -> Image.Image:
    """Put transparent PNGs onto white paper and return an RGB image."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        paper = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(paper, rgba).convert("RGB")
    return img.convert("RGB")


def _estimate_paper(gray: Image.Image) -> np.ndarray:
    """Estimate the (uneven) paper brightness: remove thin strokes, then blur."""
    w, h = gray.size
    sw, sh = max(64, w // 4), max(64, h // 4)
    small = gray.resize((sw, sh), Image.BILINEAR)
    small = small.filter(ImageFilter.MaxFilter(7))                  # strokes vanish, paper remains
    small = small.filter(ImageFilter.GaussianBlur(radius=max(sw, sh) / 20))
    return np.asarray(small.resize((w, h), Image.BICUBIC), dtype=np.float32)


def _clean_paper_and_lines(gray: Image.Image, notes: list[str]) -> Image.Image:
    arr = np.asarray(gray, dtype=np.float32)
    paper = np.maximum(_estimate_paper(gray), 32.0)
    norm = np.clip(arr / paper, 0.0, 1.0)                           # flat-field correction
    lo = min(float(np.percentile(norm, 0.5)), 0.7)                  # darkest strokes
    x = np.clip((norm - lo) / max(0.97 - lo, 1e-3), 0.0, 1.0)
    x = x ** 1.6                                                    # darken faint pencil lines
    x[x > 0.9] = 1.0                                                # snap paper noise to pure white
    if float(np.percentile(norm, 1)) > 0.93:
        notes.append("The drawing looks very faint or almost blank - results may ignore it. Try darker lines.")
    notes.append("Flattened uneven paper lighting and strengthened faint lines.")
    return Image.fromarray((x * 255).astype(np.uint8), mode="L")


def _pad_extreme_aspect(img: Image.Image, notes: list[str]) -> Image.Image:
    w, h = img.size
    if max(w, h) / min(w, h) <= MAX_ASPECT:
        return img
    long_side = max(w, h)
    short_target = int(round(long_side / SAFE_ASPECT))
    new_size = (w, short_target) if w >= h else (short_target, h)
    canvas = Image.new("RGB", new_size, (255, 255, 255))
    canvas.paste(img, ((new_size[0] - w) // 2, (new_size[1] - h) // 2))
    notes.append("Padded with white paper because the aspect ratio was beyond the API limit (2.5:1).")
    return canvas


def _encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# 3. Public entry point
# --------------------------------------------------------------------------- #
def prepare_sketch(
    path: str | os.PathLike,
    *,
    max_side: int = 1024,
    max_upload_mb: int = 10,
    clean_paper: bool = True,
) -> PreparedSketch:
    """Validate and normalise an uploaded sketch for the generation API."""
    notes: list[str] = []
    raw = load_and_validate(path, max_upload_mb)
    original_size = raw.size

    rgb = _flatten_alpha(raw)
    gray = rgb.convert("L")

    # White-on-dark drawings (chalkboard / blueprint) -> invert to dark-on-light.
    if float(np.asarray(gray).mean()) < 110:
        gray = ImageOps.invert(gray)
        notes.append("Detected light lines on a dark background and inverted them.")

    # Resize: large photos are shrunk (faster, cheaper); tiny ones are enlarged.
    w, h = gray.size
    long_side = max(w, h)
    if long_side > max_side:
        scale = max_side / long_side
    elif long_side < MIN_LONG_SIDE:
        scale = MIN_LONG_SIDE / long_side
    else:
        scale = 1.0
    if scale != 1.0:
        gray = gray.resize((max(64, round(w * scale)), max(64, round(h * scale))), Image.LANCZOS)
        notes.append(f"Resized {w}x{h} -> {gray.size[0]}x{gray.size[1]}.")

    if clean_paper:
        gray = _clean_paper_and_lines(gray, notes)
    else:
        gray = ImageOps.autocontrast(gray, cutoff=0.5)
        notes.append("Paper clean-up skipped (auto-contrast only).")

    result = _pad_extreme_aspect(gray.convert("RGB"), notes)

    png = _encode_png(result)
    while len(png) > MAX_PAYLOAD_BYTES and min(result.size) > 128:     # extremely unlikely safeguard
        result = result.resize((int(result.size[0] * 0.8), int(result.size[1] * 0.8)), Image.LANCZOS)
        png = _encode_png(result)
    return PreparedSketch(image=result, png_bytes=png, original_size=original_size, notes=notes)
