"""
offline_stylizer.py - NON-GENERATIVE fallback (classical image processing).

This is NOT AI image generation. It never invents detail: it only thickens the
sketch lines, colours enclosed regions with a style palette (flood fill) and adds
simple lighting/paper grain. It exists so the app is usable with no API key and
so you can demonstrate the difference between classical processing and a
diffusion model. Hand-drawn gaps in outlines can make colour "leak" - a known limit.
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PALETTES: dict[str, dict] = {
    "photorealistic": {"bg": (214, 226, 236), "fills": [(176, 132, 98), (120, 140, 160), (198, 170, 140), (96, 120, 92)]},
    "architectural": {"bg": (200, 222, 240), "fills": [(228, 220, 205), (150, 98, 78), (92, 118, 140), (110, 140, 96)]},
    "product": {"bg": (236, 236, 238), "fills": [(196, 160, 96), (70, 74, 82), (160, 168, 178)]},
    "interior": {"bg": (238, 230, 218), "fills": [(176, 138, 100), (120, 140, 126), (214, 190, 160), (96, 104, 120)]},
    "minimalist": {"bg": (244, 243, 240), "fills": [(206, 200, 190), (170, 176, 170), (226, 222, 214)]},
    "concept": {"bg": (36, 42, 66), "fills": [(214, 112, 88), (82, 148, 172), (232, 190, 96), (126, 98, 168)]},
}
MAX_WORK_SIDE = 768        # flood fill is pure Python, so work at a modest size


def stylize(sketch: Image.Image, style_key: str = "photorealistic", seed: int = 0) -> Image.Image:
    rng = random.Random(seed or None)
    pal = PALETTES.get(style_key, PALETTES["photorealistic"])
    orig_size = sketch.size
    work = sketch.convert("L")
    scale = min(1.0, MAX_WORK_SIDE / max(work.size))
    if scale < 1.0:
        work = work.resize((round(work.size[0] * scale), round(work.size[1] * scale)), Image.LANCZOS)
    w, h = work.size

    # 1. binary line mask, thickened so small gaps in hand-drawn outlines are closed
    binary = work.point(lambda p: 0 if p < 170 else 255)
    closed = binary.filter(ImageFilter.MinFilter(5))
    canvas = closed.convert("RGB")

    # 2. background first, then flood-fill the enclosed regions on a coarse grid
    if canvas.getpixel((0, 0)) == (255, 255, 255):
        ImageDraw.floodfill(canvas, (0, 0), pal["bg"], thresh=0)
    step = max(8, min(w, h) // 40)
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if canvas.getpixel((x, y)) == (255, 255, 255):
                ImageDraw.floodfill(canvas, (x, y), rng.choice(pal["fills"]), thresh=0)

    # 3. gentle top-light gradient, vignette and paper grain (numpy)
    arr = np.asarray(canvas, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    light = 1.06 - 0.22 * (yy / h)
    vignette = 1.0 - 0.18 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    grain = np.random.default_rng(seed or None).normal(0, 3.0, (h, w, 1))
    arr = arr * (light * vignette)[..., None] + grain

    # 4. ink the original lines on top
    ink = (np.asarray(binary, dtype=np.float32) / 255.0)[..., None]       # 0 on lines, 1 elsewhere
    arr = arr * (0.25 + 0.75 * ink)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    return out.resize(orig_size, Image.LANCZOS) if out.size != orig_size else out
