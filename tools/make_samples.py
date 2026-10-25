"""Generate the three demo sketches in ../samples (hand-drawn look: wobbly pencil lines).

Run from the project folder:  python tools/make_samples.py
Pillow only. Safe to re-run; it overwrites the sample files.
"""
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "samples"
W, H = 768, 576


def wobble_line(draw, p0, p1, rng, width=3, passes=2, amp=2.2):
    """Draw a slightly shaky line, twice, like a pencil stroke."""
    for _ in range(passes):
        n = max(4, int(math.dist(p0, p1) / 18))
        pts = []
        for i in range(n + 1):
            t = i / n
            x = p0[0] + (p1[0] - p0[0]) * t + rng.uniform(-amp, amp)
            y = p0[1] + (p1[1] - p0[1]) * t + rng.uniform(-amp, amp)
            pts.append((x, y))
        draw.line(pts, fill=(45, 45, 50), width=width, joint="curve")


def poly(draw, pts, rng, closed=True, **kw):
    seq = pts + ([pts[0]] if closed else [])
    for a, b in zip(seq, seq[1:]):
        wobble_line(draw, a, b, rng, **kw)


def circle(draw, cx, cy, r, rng, width=3):
    pts = [(cx + (r + rng.uniform(-1.5, 1.5)) * math.cos(a / 20 * 2 * math.pi),
            cy + (r + rng.uniform(-1.5, 1.5)) * math.sin(a / 20 * 2 * math.pi)) for a in range(21)]
    draw.line(pts, fill=(45, 45, 50), width=width, joint="curve")


def finish(img, name):
    img = img.filter(ImageFilter.GaussianBlur(0.7))
    img.save(OUT / name)
    print("wrote", OUT / name)


def house(rng):
    img = Image.new("RGB", (W, H), (252, 252, 250)); d = ImageDraw.Draw(img)
    poly(d, [(170, 250), (170, 470), (600, 470), (600, 250)], rng, closed=False)         # walls
    poly(d, [(140, 260), (385, 120), (630, 260)], rng, closed=False)                       # roof
    poly(d, [(360, 470), (360, 360), (430, 360), (430, 470)], rng, closed=False)           # door
    poly(d, [(215, 310), (300, 310), (300, 390), (215, 390)], rng)                         # windows
    poly(d, [(470, 310), (555, 310), (555, 390), (470, 390)], rng)
    wobble_line(d, (257, 310), (257, 390), rng, width=2); wobble_line(d, (215, 350), (300, 350), rng, width=2)
    wobble_line(d, (512, 310), (512, 390), rng, width=2); wobble_line(d, (470, 350), (555, 350), rng, width=2)
    poly(d, [(500, 150), (500, 205)], rng, closed=False); poly(d, [(540, 150), (540, 230)], rng, closed=False)
    poly(d, [(500, 150), (540, 150)], rng, closed=False)                                   # chimney
    wobble_line(d, (90, 472), (690, 472), rng)
    finish(img, "house.png")


def chair(rng):
    img = Image.new("RGB", (W, H), (252, 252, 250)); d = ImageDraw.Draw(img)
    poly(d, [(250, 120), (520, 120), (530, 300), (240, 300)], rng)                         # backrest
    poly(d, [(220, 300), (550, 300), (575, 360), (195, 360)], rng)                         # seat
    for x0, x1 in [(215, 205), (550, 560)]:
        poly(d, [(x0, 360), (x1, 500)], rng, closed=False, width=4)                         # front legs
    for x0, x1 in [(260, 270), (500, 495)]:
        poly(d, [(x0, 360), (x1, 470)], rng, closed=False, width=3)                         # back legs
    for y in (160, 210, 260):
        wobble_line(d, (262, y), (510, y), rng, width=2)                                   # slats
    wobble_line(d, (150, 505), (620, 505), rng, width=2)
    finish(img, "chair.png")


def car(rng):
    img = Image.new("RGB", (W, H), (252, 252, 250)); d = ImageDraw.Draw(img)
    poly(d, [(90, 360), (110, 300), (260, 285), (330, 215), (520, 215), (610, 290), (690, 310), (700, 360)], rng, closed=False)
    wobble_line(d, (90, 360), (700, 360), rng)
    poly(d, [(290, 290), (345, 235), (440, 235), (440, 290)], rng)                         # windows
    poly(d, [(455, 290), (455, 235), (515, 235), (570, 290)], rng)
    circle(d, 210, 365, 55, rng, width=4); circle(d, 210, 365, 24, rng, width=3)
    circle(d, 570, 365, 55, rng, width=4); circle(d, 570, 365, 24, rng, width=3)
    wobble_line(d, (60, 425), (730, 425), rng, width=2)
    finish(img, "car.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    rng = random.Random(7)
    house(rng); chair(rng); car(rng)
