"""
evaluation.py - academic evaluation log.

Three kinds of values are kept strictly apart (see README):
  * MEASURED  : generation time, provider, seed, estimated credits - recorded automatically.
  * HUMAN     : quality / structure ratings (1-5) - entered by YOU, blank until you rate.
  * PROXY     : an automatic edge-based structure score - experimental, not validated.
Nothing is ever pre-filled or invented. The optional CLIP column stays blank unless you type a value.
"""
from __future__ import annotations

import csv
import threading
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from config import BASE_DIR

LOG_PATH = BASE_DIR / "evaluation" / "evaluation_log.csv"

COLUMNS = [
    "run_id", "timestamp_utc", "sketch_category", "sketch_file", "prompt", "style", "provider", "model",
    "control_strength", "seed_used", "generation_time_s", "credits_estimated",
    "structure_proxy_edge_recall", "structure_proxy_chance_level",
    "human_visual_quality_1to5", "human_structure_preservation_1to5",
    "clip_similarity_optional_manual", "notes",
]


@dataclass
class EvalRecord:
    run_id: str = ""
    timestamp_utc: str = ""
    sketch_category: str = ""
    sketch_file: str = ""
    prompt: str = ""
    style: str = ""
    provider: str = ""
    model: str = ""
    control_strength: str = ""
    seed_used: str = ""
    generation_time_s: str = ""
    credits_estimated: str = ""
    structure_proxy_edge_recall: str = ""
    structure_proxy_chance_level: str = ""
    human_visual_quality_1to5: str = ""
    human_structure_preservation_1to5: str = ""
    clip_similarity_optional_manual: str = ""
    notes: str = ""


assert [f.name for f in fields(EvalRecord)] == COLUMNS


def edge_structure_proxy(sketch: Image.Image, output: Image.Image) -> tuple[float, float] | None:
    """
    EXPERIMENTAL automatic proxy for structure preservation.

    recall = fraction of the sketch's line pixels that lie within a few pixels of a strong
    edge in the generated image. chance_level = fraction of the image covered by that
    "near an edge" zone, i.e. the score a random line would get. Only recall clearly
    ABOVE chance_level is informative. Textures create extra edges, so this is a rough
    indicator and must not replace the human rating.
    """
    sketch = sketch.convert("L")
    out = output.convert("L").resize(sketch.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.0))
    g = np.asarray(out, dtype=np.float32)
    dx = np.zeros_like(g); dy = np.zeros_like(g)
    dx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    dy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(dx, dy)
    if float(mag.max()) <= 0:
        return None
    strong = mag >= np.percentile(mag, 92)                       # strongest 8% of pixels
    radius = max(2, round(0.005 * max(sketch.size)))
    zone = np.asarray(
        Image.fromarray((strong * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * radius + 1))
    ) > 0
    lines = np.asarray(sketch) < 128
    if int(lines.sum()) < 50:
        return None
    return float((lines & zone).sum() / lines.sum()), float(zone.mean())


class EvalLog:
    """Small thread-safe CSV-backed log (no database needed)."""

    def __init__(self, path: Path = LOG_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.records: list[EvalRecord] = []
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            with open(self.path, newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    self.records.append(EvalRecord(**{c: row.get(c, "") or "" for c in COLUMNS}))
        except (OSError, csv.Error, TypeError):
            self.records = []

    def _write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:     # BOM => opens cleanly in Excel
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            for r in self.records:
                writer.writerow(asdict(r))

    # -- operations --------------------------------------------------------
    def next_run_id(self) -> str:
        nums = [int(r.run_id[1:]) for r in self.records if r.run_id[1:].isdigit()]
        return f"R{(max(nums) + 1) if nums else 1:04d}"

    def add(self, **values) -> EvalRecord:
        with self._lock:
            rec = EvalRecord(**{k: ("" if v is None else str(v)) for k, v in values.items()})
            rec.run_id = self.next_run_id()
            rec.timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            self.records.append(rec)
            self._write(self.path)
            return rec

    def get(self, run_id: str) -> EvalRecord | None:
        return next((r for r in self.records if r.run_id == run_id), None)

    def rate(self, run_id: str, quality: int | None, structure: int | None,
             clip: str = "", notes: str = "") -> EvalRecord:
        with self._lock:
            rec = self.get(run_id)
            if rec is None:
                raise KeyError(run_id)
            if quality is not None:
                rec.human_visual_quality_1to5 = str(int(quality))
            if structure is not None:
                rec.human_structure_preservation_1to5 = str(int(structure))
            if clip.strip():
                float(clip)                                   # raises ValueError if not numeric
                rec.clip_similarity_optional_manual = clip.strip()
            if notes.strip():
                rec.notes = notes.strip()
            self._write(self.path)
            return rec

    def clear(self) -> None:
        with self._lock:
            self.records = []
            if self.path.exists():
                self.path.unlink()

    def rows(self) -> list[list[str]]:
        return [[getattr(r, c) for c in COLUMNS] for r in self.records]

    def export(self, dest: Path) -> Path:
        with self._lock:
            self._write(Path(dest))
        return Path(dest)
