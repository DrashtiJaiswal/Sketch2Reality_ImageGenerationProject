"""
Sketch2Reality - sketch-to-realistic-design generator (Multimodal Generative AI mini-project).

Run with:   python app.py        then open http://127.0.0.1:7860
Heavy AI work happens on a hosted API (see providers.py); this app needs no GPU and no model download.
"""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")      # no telemetry from the local app

import html
import re
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd

from config import APP_NAME, BASE_DIR, load_settings, redact
from evaluation import EvalLog
from pipeline import RUN_ASSETS, USAGE, Outcome, estimated_cost_text, run_generation
from prompts import (DEFAULT_NEGATIVE_PROMPT, PROMPT_EXAMPLES, SKETCH_CATEGORIES, STABILITY_STYLE_PRESETS,
                     STYLE_CHOICES, STYLES)
from providers import get_provider, provider_choices
from ui_text import ABOUT_MD, CSS, EVAL_INTRO_MD, HERO_HTML

SETTINGS = load_settings()
EVAL_LOG = EvalLog()
THEME = gr.themes.Base(
    primary_hue="teal", neutral_hue="slate", radius_size="md",
    font=[gr.themes.GoogleFont("Instrument Sans"), "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"],
)
SAMPLES = [
    ("house.png", "House / building"), ("chair.png", "Furniture"), ("car.png", "Vehicle"),
]
TABLE_COLUMNS = [
    ("run_id", "Run"), ("timestamp_utc", "Time (UTC)"), ("sketch_category", "Category"), ("prompt", "Prompt"),
    ("style", "Style"), ("provider", "Provider"), ("generation_time_s", "Time (s)"),
    ("human_visual_quality_1to5", "Quality (1-5)"), ("human_structure_preservation_1to5", "Structure (1-5)"),
    ("structure_proxy_edge_recall", "Edge proxy"), ("structure_proxy_chance_level", "Proxy chance"),
]


# --------------------------------------------------------------------------- #
# Small HTML helpers
# --------------------------------------------------------------------------- #
def _fmt(text: str) -> str:
    """Escape text for HTML and turn `code` spans into <code>."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(text))


def status_html(kind: str, title: str, message: str = "", notes: list[str] | None = None) -> str:
    body = f"<div>{_fmt(message)}</div>" if message else ""
    if notes:
        body += "<ul>" + "".join(f"<li>{_fmt(n)}</li>" for n in notes) + "</ul>"
    return f'<div class="s2r-status {kind}"><strong>{_fmt(title)}</strong>{body}</div>'


def setup_banner() -> str:
    provider = get_provider(SETTINGS.provider, SETTINGS)
    if provider.is_configured():
        return ""
    return status_html("warn", "One-time setup needed for AI generation", provider.setup_message())


def safe_error(exc: Exception) -> str:
    print("Unexpected error:\n" + redact(traceback.format_exc(), SETTINGS.stability_api_key))
    return status_html("error", "Unexpected error",
                       redact(f"{type(exc).__name__}: {exc}", SETTINGS.stability_api_key)
                       + " - details were printed in the terminal.")


# --------------------------------------------------------------------------- #
# Studio tab handlers
# --------------------------------------------------------------------------- #
def cost_md(provider_key: str, variations: int) -> str:
    provider = get_provider(provider_key, SETTINGS)
    return f'<div class="s2r-cost">{html.escape(estimated_cost_text(provider, int(variations or 1), SETTINGS))}</div>'


def on_provider_change(provider_key: str, variations: int):
    caps = get_provider(provider_key, SETTINGS).capabilities
    return gr.Group(visible=caps.control_strength), cost_md(provider_key, variations)


def info_markdown(o: Outcome) -> str:
    lines = [f"**Provider:** {o.provider_label}", f"**Model / endpoint:** {o.model_label}"]
    total = sum(v.elapsed_s for v in o.variants)
    lines.append(f"**Generation time:** {total:.1f} s total (measured around the API call)")
    for v in o.variants:
        proxy = (f" - edge proxy {v.proxy_recall:.2f} (chance level {v.proxy_chance:.2f})"
                 if v.proxy_recall is not None else "")
        seed = v.seed if v.seed is not None else "n/a"
        lines.append(f"- `{v.run_id}`: {v.elapsed_s:.1f} s, seed {seed}{proxy}")
    price = o.credits * SETTINGS.credit_price_usd
    lines.append(f"**Credits for this run (estimated from the published price):** {o.credits:g} (about ${price:.2f})")
    lines.append(f"**This session:** {USAGE['images']} image(s), about {USAGE['credits']:g} credits. "
                 "Your real balance is on the Stability AI dashboard.")
    if "non-AI" in o.model_label or "not AI" in o.model_label:
        lines.append("*This result comes from classical image processing, not generative AI.*")
    lines.append("*The edge proxy is experimental; rate the result yourself in the Evaluation tab.*")
    return "\n\n".join(lines)


def render_outcome(o: Outcome):
    """Map an Outcome onto the Studio output components."""
    skip = gr.skip()
    clean_img = o.prepared.image if o.prepared else skip
    if not o.variants:
        kind = "warn" if o.kind == "setup" else "error"
        return (status_html(kind, o.title, o.message, o.notes), skip, skip, skip, skip, clean_img, skip)

    paths = [v.image_path for v in o.variants]
    first = paths[0]
    caption = lambda v: f"{v.run_id} (seed {v.seed if v.seed is not None else 'n/a'})"
    gallery = gr.Gallery(value=[(v.image_path, caption(v)) for v in o.variants], visible=len(paths) > 1)
    if o.kind == "partial":
        status = status_html("warn", o.title, o.message, o.notes)
    else:
        status = status_html("ok", "Generation complete", "Drag the slider to compare your sketch with the result.", o.notes)
    return (status, (o.sketch_path, first), gallery, info_markdown(o),
            gr.DownloadButton(value=first, interactive=True), clean_img,
            {"sketch": o.sketch_path, "paths": paths})


def on_generate(sketch, prompt, style, category, provider_key, strength, negative, seed, variations,
                clean_paper, preset, progress=gr.Progress()):
    try:
        outcome = run_generation(
            sketch, prompt, style, category, provider_key, settings=SETTINGS, eval_log=EVAL_LOG,
            control_strength=strength, negative_prompt=negative or "", seed=int(seed or 0),
            variations=int(variations or 1), clean_paper=bool(clean_paper), style_preset_choice=preset,
            progress=lambda frac, msg: progress(frac, desc=msg),
        )
        return render_outcome(outcome)
    except Exception as exc:                                  # last-resort guard: never crash the UI
        skip = gr.skip()
        return safe_error(exc), skip, skip, skip, skip, skip, skip


def on_gallery_select(evt: gr.SelectData, state):
    if not state or evt.index is None or evt.index >= len(state["paths"]):
        return gr.skip(), gr.skip()
    path = state["paths"][evt.index]
    return (state["sketch"], path), gr.DownloadButton(value=path, interactive=True)


def on_example_prompt(category: str) -> str:
    return PROMPT_EXAMPLES.get(category, "")


# --------------------------------------------------------------------------- #
# Evaluation tab handlers
# --------------------------------------------------------------------------- #
def _table() -> pd.DataFrame:
    rows = [[getattr(r, col) for col, _ in TABLE_COLUMNS] for r in EVAL_LOG.records]
    return pd.DataFrame(rows, columns=[label for _, label in TABLE_COLUMNS])


def _run_choices():
    return [(f"{r.run_id} - {r.sketch_category or 'no category'} - {r.style}", r.run_id)
            for r in reversed(EVAL_LOG.records)]


def refresh_eval():
    choices = _run_choices()
    return _table(), gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


def on_select_run(run_id):
    rec = EVAL_LOG.get(run_id) if run_id else None
    if rec is None:
        return None, "Select a run to see it here."
    assets = RUN_ASSETS.get(run_id)
    pair = (assets["sketch"], assets["result"]) if assets and Path(assets["result"]).is_file() else None
    text = (f"**Prompt:** {rec.prompt or '(none)'}  \n**Style:** {rec.style} | **Provider:** {rec.provider} | "
            f"**Time:** {rec.generation_time_s} s")
    if pair is None:
        text += "\n\n*Images from earlier sessions are not stored; you can still rate this run from your own notes.*"
    return pair, text


def on_save_rating(run_id, quality, structure, clip, notes):
    if not run_id:
        return status_html("warn", "Select a run first")
    if quality is None and structure is None and not (clip or "").strip() and not (notes or "").strip():
        return status_html("warn", "Nothing to save", "Choose at least one rating, or add a note.")
    try:
        EVAL_LOG.rate(run_id, quality, structure, clip or "", notes or "")
    except ValueError:
        return status_html("error", "CLIP value must be a number", "Leave it empty if you did not compute one.")
    except KeyError:
        return status_html("error", "Unknown run", "Refresh the table and select the run again.")
    return status_html("ok", f"Saved ratings for {run_id}")


def on_export():
    path = Path(tempfile.gettempdir()) / "sketch2reality" / f"sketch2reality_evaluation_{datetime.now():%Y%m%d_%H%M%S}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    EVAL_LOG.export(path)
    return gr.File(value=str(path), visible=True)


def on_clear(confirm):
    if not confirm:
        return status_html("warn", "Tick the confirmation box first", "This permanently deletes all logged runs.")
    EVAL_LOG.clear()
    return status_html("ok", "Evaluation log cleared")


def on_batch(files, template, category, style, provider_key, strength, negative, seed, clean_paper, preset,
             confirm, progress=gr.Progress()):
    try:
        if not files:
            return status_html("warn", "No sketches selected", "Add several sketch images to run them together.")
        provider = get_provider(provider_key, SETTINGS)
        if not provider.is_configured():
            return status_html("warn", "API key needed", provider.setup_message())
        credits = provider.credits_per_image() * len(files)
        if credits > 0 and not confirm:
            return status_html("warn", "Please confirm the cost",
                               f"{len(files)} sketches x {provider.credits_per_image():g} credits = {credits:g} credits "
                               f"(about ${credits * SETTINGS.credit_price_usd:.2f}). Tick the confirmation box to continue.")
        lines, stop = [], False
        for i, f in enumerate(files):
            path = f if isinstance(f, str) else getattr(f, "name", str(f))
            stem = Path(path).stem.replace("_", " ").replace("-", " ")
            prompt = (template or "{name}").replace("{name}", stem)
            o = run_generation(path, prompt, style, category, provider_key, settings=SETTINGS, eval_log=EVAL_LOG,
                               control_strength=strength, negative_prompt=negative or "", seed=int(seed or 0),
                               variations=1, clean_paper=bool(clean_paper), style_preset_choice=preset,
                               progress=lambda frac, msg: progress((i + frac) / len(files), desc=f"{Path(path).name}: {msg}"))
            if o.variants:
                lines.append(f"{Path(path).name}: finished as {o.variants[0].run_id} in {o.variants[0].elapsed_s:.1f} s")
            else:
                lines.append(f"{Path(path).name}: FAILED - {o.message}")
                if o.kind == "setup" or o.title in ("API key rejected", "Out of credits", "Rate limit reached"):
                    lines.append("Stopped early so no further credits are spent.")
                    stop = True
            if stop:
                break
        ok = sum("finished as" in ln for ln in lines)
        return status_html("ok" if ok == len(files) else "warn", f"Batch finished: {ok} of {len(files)} succeeded", notes=lines)
    except Exception as exc:
        return safe_error(exc)


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
def build_demo() -> gr.Blocks:
    default_provider = SETTINGS.provider if SETTINGS.provider in dict(
        (k, v) for v, k in provider_choices()) else "stability"
    caps = get_provider(default_provider, SETTINGS).capabilities

    with gr.Blocks(title=f"{APP_NAME} - sketch to realistic design") as demo:
        gr.HTML(HERO_HTML)
        banner = setup_banner()
        if banner:
            gr.HTML(banner)
        state = gr.State(None)

        with gr.Tabs():
            # ---------------------------------------------------------------- Studio
            with gr.Tab("Studio"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        sketch = gr.Image(label="1. Your sketch", type="filepath", sources=["upload", "clipboard", "webcam"],
                                          height=360, buttons=["fullscreen"])
                        category = gr.Dropdown(SKETCH_CATEGORIES, value="House / building", label="Sketch category (for the evaluation log)")
                        prompt = gr.Textbox(label="2. Describe the design (English)", lines=3,
                                            placeholder=PROMPT_EXAMPLES["House / building"])
                        example_btn = gr.Button("Insert an example prompt for this category", size="sm")
                        style = gr.Dropdown(STYLE_CHOICES, value="photorealistic", label="3. Style")
                        provider_dd = gr.Dropdown(provider_choices(), value=default_provider, label="Generator")
                        with gr.Accordion("Advanced settings", open=False):
                            clean_paper = gr.Checkbox(True, label="Clean paper and lighting (recommended for photos of paper sketches)")
                            with gr.Group(visible=caps.control_strength) as api_group:
                                strength = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Sketch adherence (control strength)",
                                                     info="Higher follows your lines more closely; lower gives the model more freedom.")
                                negative = gr.Textbox(DEFAULT_NEGATIVE_PROMPT, label="Negative prompt (what to avoid)", lines=2)
                                preset = gr.Dropdown([("Automatic (from style)", "auto"), ("None", "none")]
                                                     + [(p, p) for p in STABILITY_STYLE_PRESETS],
                                                     value="auto", label="Stability style preset")
                                seed = gr.Number(0, precision=0, minimum=0, maximum=4294967294, label="Seed (0 = random)",
                                                 info="Reuse the seed shown after a run to get a similar result again.")
                                variations = gr.Slider(1, SETTINGS.max_variations, value=1, step=1,
                                                       label="Variations (each one is a separate paid generation)")
                        cost = gr.HTML(cost_md(default_provider, 1))
                        generate_btn = gr.Button("Generate realistic design", variant="primary", size="lg")
                    with gr.Column(scale=6):
                        status = gr.HTML(status_html("info", "Ready", "Upload a sketch (or pick a sample below), write a prompt and press Generate."))
                        compare = gr.ImageSlider(label="Before / after - drag to compare", height=420, show_label=True)
                        gallery = gr.Gallery(label="Variations - click one to compare it", columns=3, height="auto", visible=False)
                        download = gr.DownloadButton("Download result (PNG)", interactive=False, variant="secondary")
                        info = gr.Markdown()
                        with gr.Accordion("What the model received (cleaned sketch)", open=False):
                            clean_view = gr.Image(label="Cleaned sketch", interactive=False, height=260)
                existing = [(str(BASE_DIR / "samples" / f), c) for f, c in SAMPLES if (BASE_DIR / "samples" / f).is_file()]
                if existing:
                    gr.Examples(
                        examples=[[p, c, PROMPT_EXAMPLES[c], "photorealistic"] for p, c in existing],
                        inputs=[sketch, category, prompt, style], label="Sample sketches (click one to load it)",
                        cache_examples=False,
                    )

            # ------------------------------------------------------------ Evaluation
            with gr.Tab("Batch & evaluation"):
                gr.Markdown(EVAL_INTRO_MD)
                with gr.Accordion("Batch run - test several sketches in one go", open=False):
                    gr.Markdown("Uses the style, generator and advanced settings currently chosen in the Studio tab. "
                                "Write one prompt template; `{name}` is replaced by each file name (underscores become spaces).")
                    batch_files = gr.File(label="Sketch images", file_count="multiple", file_types=["image"])
                    batch_template = gr.Textbox(label="Prompt template", value="a realistic {name}, detailed materials, natural light")
                    batch_confirm = gr.Checkbox(False, label="I understand this uses credits for every sketch (one generation each)")
                    batch_btn = gr.Button("Run batch")
                    batch_status = gr.HTML()
                gr.Markdown("### Rate a result")
                with gr.Row():
                    with gr.Column(scale=5):
                        run_dd = gr.Dropdown(choices=_run_choices(), value=(_run_choices()[0][1] if _run_choices() else None),
                                             label="Run to rate")
                        run_info = gr.Markdown("Select a run to see it here.")
                        q_radio = gr.Radio([1, 2, 3, 4, 5], value=None, label="Visual quality (1 poor - 5 excellent)")
                        s_radio = gr.Radio([1, 2, 3, 4, 5], value=None, label="Sketch-structure preservation (1 ignored - 5 faithful)")
                        clip_box = gr.Textbox(label="CLIP similarity (optional, only if you computed it elsewhere)", placeholder="e.g. 0.31")
                        notes_box = gr.Textbox(label="Notes (optional)", lines=2)
                        save_btn = gr.Button("Save ratings", variant="primary")
                        rate_status = gr.HTML()
                    with gr.Column(scale=6):
                        run_view = gr.ImageSlider(label="Sketch vs result", height=360)
                gr.Markdown("### Evaluation log")
                table = gr.Dataframe(value=_table(), interactive=False, wrap=True, max_height=320)
                with gr.Row():
                    refresh_btn = gr.Button("Refresh table")
                    export_btn = gr.Button("Export CSV")
                    clear_confirm = gr.Checkbox(False, label="Confirm: delete the whole log")
                    clear_btn = gr.Button("Clear log", variant="stop")
                export_file = gr.File(label="Exported CSV (click to download)", visible=False)
                clear_status = gr.HTML()

            # ------------------------------------------------------------- About
            with gr.Tab("How it works"):
                gr.Markdown(ABOUT_MD)

        # ------------------------------------------------------------- wiring
        studio_inputs = [sketch, prompt, style, category, provider_dd, strength, negative, seed, variations, clean_paper, preset]
        gen_event = generate_btn.click(
            on_generate, studio_inputs, [status, compare, gallery, info, download, clean_view, state])
        gen_event.then(refresh_eval, None, [table, run_dd])
        gallery.select(on_gallery_select, state, [compare, download])
        example_btn.click(on_example_prompt, category, prompt)
        provider_dd.change(on_provider_change, [provider_dd, variations], [api_group, cost])
        variations.change(cost_md, [provider_dd, variations], cost)

        run_dd.change(on_select_run, run_dd, [run_view, run_info])
        save_btn.click(on_save_rating, [run_dd, q_radio, s_radio, clip_box, notes_box], rate_status).then(
            refresh_eval, None, [table, run_dd])
        refresh_btn.click(refresh_eval, None, [table, run_dd])
        export_btn.click(on_export, None, export_file)
        clear_btn.click(on_clear, clear_confirm, clear_status).then(refresh_eval, None, [table, run_dd])
        batch_btn.click(
            on_batch,
            [batch_files, batch_template, category, style, provider_dd, strength, negative, seed, clean_paper, preset, batch_confirm],
            batch_status,
        ).then(refresh_eval, None, [table, run_dd])
    return demo


def main() -> None:
    demo = build_demo()
    demo.queue(default_concurrency_limit=1)       # one generation at a time -> no accidental double spending
    print(f"\n{APP_NAME} starting on http://127.0.0.1:{SETTINGS.server_port}  (press Ctrl+C to stop)")
    if not get_provider(SETTINGS.provider, SETTINGS).is_configured():
        print("Note: no API key configured yet - the app will show setup instructions. See README.md.")
    demo.launch(
        server_name="127.0.0.1", server_port=SETTINGS.server_port, inbrowser=True,
        theme=THEME, css=CSS, allowed_paths=[str(BASE_DIR / "samples")],
    )


if __name__ == "__main__":
    main()
