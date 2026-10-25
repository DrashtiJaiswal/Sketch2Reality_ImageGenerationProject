"""
pipeline.py - the generation workflow, independent of the web UI (so it can be unit-tested).

    upload -> validate/clean (preprocess.py) -> build prompt (prompts.py)
           -> provider.generate() (providers.py) -> measure + log (evaluation.py)
"""
from __future__ import annotations

import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from config import Settings
from evaluation import EvalLog, edge_structure_proxy
from preprocess import InvalidImageError, PreparedSketch, prepare_sketch
from prompts import STYLES, PromptError, build_prompt, validate_prompt
from providers import BaseProvider, GenerationRequest, ProviderError, get_provider

# Generated images live in the OS temp folder (not in the project folder).
OUTPUT_DIR = Path(tempfile.gettempdir()) / "sketch2reality"
RUN_ASSETS: dict[str, dict[str, str]] = {}          # run_id -> {"sketch": path, "result": path}
_usage_lock = threading.Lock()
USAGE = {"images": 0, "credits": 0.0}


@dataclass
class VariantOutput:
    run_id: str
    image_path: str
    seed: Optional[int]
    elapsed_s: float
    proxy_recall: Optional[float]
    proxy_chance: Optional[float]


@dataclass
class Outcome:
    kind: str                                   # success | partial | error | setup
    title: str
    message: str = ""
    prepared: Optional[PreparedSketch] = None
    sketch_path: str = ""
    variants: list[VariantOutput] = field(default_factory=list)
    provider_label: str = ""
    model_label: str = ""
    credits: float = 0.0
    notes: list[str] = field(default_factory=list)


def estimated_cost_text(provider: BaseProvider, variations: int, settings: Settings) -> str:
    if provider.credits_per_image() <= 0:
        return "This run is free (runs locally, no API credits)."
    n = variations if provider.capabilities.variations else 1
    credits = provider.credits_per_image() * n
    return (f"This run uses {n} x {provider.credits_per_image():g} = {credits:g} credits "
            f"(about ${credits * settings.credit_price_usd:.2f}). Failed generations are not billed by Stability AI.")


def _resolve_preset(choice: str, style_key: str) -> Optional[str]:
    if choice == "auto":
        return STYLES[style_key].stability_preset if style_key in STYLES else None
    return None if choice == "none" else choice


def run_generation(
    sketch_path: Optional[str],
    prompt: str,
    style_key: str,
    category: str,
    provider_key: str,
    *,
    settings: Settings,
    eval_log: EvalLog,
    control_strength: float = 0.7,
    negative_prompt: str = "",
    seed: int = 0,
    variations: int = 1,
    clean_paper: bool = True,
    style_preset_choice: str = "auto",
    progress: Callable[[float, str], None] = lambda frac, msg: None,
) -> Outcome:
    provider = get_provider(provider_key, settings)
    caps = provider.capabilities

    # Fail early and politely if the API key is missing (the app itself still starts fine).
    if not provider.is_configured():
        return Outcome("setup", "API key needed", provider.setup_message())

    try:
        progress(0.05, "Checking and cleaning the sketch...")
        prepared = prepare_sketch(
            sketch_path, max_side=settings.max_image_side,
            max_upload_mb=settings.max_upload_mb, clean_paper=clean_paper,
        )
        clean_prompt = validate_prompt(prompt) if caps.uses_prompt else (prompt or "").strip()
        full_prompt = build_prompt(clean_prompt, style_key) if caps.uses_prompt else clean_prompt
    except (InvalidImageError, PromptError) as exc:
        return Outcome("error", "Cannot start", str(exc))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n = max(1, min(int(variations), settings.max_variations)) if caps.variations else 1
    base_seed = int(seed or 0)
    outcome = Outcome("success", "Done", prepared=prepared, provider_label=provider.label,
                      model_label=provider.model_label, notes=list(prepared.notes))

    sketch_tmp = OUTPUT_DIR / f"sketch_{id(prepared):x}.png"
    prepared.image.save(sketch_tmp)
    outcome.sketch_path = str(sketch_tmp)

    for i in range(n):
        progress(0.15 + 0.8 * i / n, f"Generating variation {i + 1} of {n} ({provider.key})...")
        req = GenerationRequest(
            sketch_image=prepared.image, sketch_png=prepared.png_bytes, prompt=full_prompt,
            negative_prompt=negative_prompt if caps.negative_prompt else "",
            style_key=style_key, style_preset=_resolve_preset(style_preset_choice, style_key) if caps.style_preset else None,
            control_strength=float(control_strength),
            seed=min(base_seed + i, 4294967294) if base_seed > 0 else 0,
        )
        try:
            result = provider.generate(req)
        except ProviderError as exc:
            if outcome.variants:                       # keep what we already paid for
                outcome.kind = "partial"
                outcome.title = f"{len(outcome.variants)} of {n} variations finished"
                outcome.message = str(exc)
                return outcome
            return Outcome("setup" if exc.kind == "setup" else "error",
                           {"auth": "API key rejected", "credits": "Out of credits", "rate_limit": "Rate limit reached",
                            "timeout": "Request timed out", "network": "No connection"}.get(exc.kind, "Generation failed"),
                           str(exc), prepared=prepared, sketch_path=outcome.sketch_path,
                           provider_label=provider.label, model_label=provider.model_label, notes=outcome.notes)

        proxy = edge_structure_proxy(prepared.image, result.image)
        rec = eval_log.add(
            sketch_category=category, sketch_file=Path(sketch_path).name, prompt=clean_prompt,
            style=STYLES[style_key].label if style_key in STYLES else style_key,
            provider=provider.key, model=result.model,
            control_strength=f"{control_strength:.2f}" if caps.control_strength else "",
            seed_used=result.seed if result.seed is not None else "",
            generation_time_s=f"{result.elapsed_s:.2f}", credits_estimated=f"{result.credits_estimated:g}",
            structure_proxy_edge_recall=f"{proxy[0]:.3f}" if proxy else "",
            structure_proxy_chance_level=f"{proxy[1]:.3f}" if proxy else "",
        )
        out_path = OUTPUT_DIR / f"{rec.run_id}.png"
        result.image.save(out_path)
        RUN_ASSETS[rec.run_id] = {"sketch": outcome.sketch_path, "result": str(out_path)}
        outcome.variants.append(VariantOutput(rec.run_id, str(out_path), result.seed, result.elapsed_s,
                                              proxy[0] if proxy else None, proxy[1] if proxy else None))
        outcome.credits += result.credits_estimated
        with _usage_lock:
            USAGE["images"] += 1
            USAGE["credits"] += result.credits_estimated

    progress(1.0, "Done")
    outcome.title = "Generation complete"
    return outcome
