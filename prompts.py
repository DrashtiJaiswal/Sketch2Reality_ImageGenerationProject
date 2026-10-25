"""
prompts.py - text conditioning (AI pipeline step 2).

The final text prompt sent to the model = the user's description
+ a style suffix. Text and the sketch together form the multimodal condition:
the sketch controls WHERE things are, the prompt controls WHAT they look like.
"""
from __future__ import annotations

from dataclasses import dataclass

MAX_PROMPT_CHARS = 1500


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class StyleSpec:
    key: str
    label: str
    description: str
    suffix: str                   # appended to the user's prompt
    stability_preset: str | None  # value for Stability's optional `style_preset`


STYLES: dict[str, StyleSpec] = {s.key: s for s in [
    StyleSpec("photorealistic", "Photorealistic",
              "Natural light, real materials, camera-like look.",
              "photorealistic, ultra detailed, natural lighting, realistic materials and textures, sharp focus",
              "photographic"),
    StyleSpec("architectural", "Architectural visualization",
              "Exterior/building render with landscaping and golden-hour light.",
              "architectural visualization, professional architectural photography, golden hour lighting, "
              "realistic materials, landscaped surroundings, high detail",
              "photographic"),
    StyleSpec("product", "Product render",
              "Studio product shot, soft light, clean backdrop.",
              "studio product render, soft box lighting, seamless neutral backdrop, subtle reflections, "
              "premium finish, high detail",
              None),
    StyleSpec("interior", "Interior design",
              "Room photography with window light and styled decor.",
              "interior design photography, natural window light, realistic materials, tasteful styling, high detail",
              "photographic"),
    StyleSpec("minimalist", "Minimalist",
              "Calm, simple, muted palette.",
              "minimalist design, clean composition, muted neutral palette, soft diffuse light, simple background",
              None),
    StyleSpec("concept", "Concept art",
              "Painterly, dramatic, illustration-like.",
              "professional concept art, dramatic lighting, rich colour, detailed painterly rendering",
              "digital-art"),
]}

STYLE_CHOICES = [(s.label, s.key) for s in STYLES.values()]

# Values accepted by Stability's `style_preset` parameter (from the official API reference).
STABILITY_STYLE_PRESETS = [
    "enhance", "anime", "photographic", "digital-art", "comic-book", "fantasy-art", "line-art",
    "analog-film", "neon-punk", "isometric", "low-poly", "origami", "modeling-compound",
    "cinematic", "3d-model", "pixel-art", "tile-texture",
]

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low resolution, jpeg artifacts, distorted proportions, extra parts, warped lines, "
    "watermark, text, logo, pencil texture, paper texture, sketch lines"
)

SKETCH_CATEGORIES = [
    "House / building", "Interior / room", "Furniture", "Product", "Vehicle",
    "Character / creature", "Landscape", "Other",
]

# (category, prompt) pairs used by the "example prompts" buttons.
PROMPT_EXAMPLES = {
    "House / building": "a modern two-storey family house with white walls, large glass windows and a wooden front door, green lawn, clear blue sky",
    "Interior / room": "a cozy Scandinavian living room with a light oak floor, linen sofa and warm afternoon light",
    "Furniture": "a mid-century armchair upholstered in mustard velvet with walnut wooden legs, in a bright studio",
    "Product": "a sleek ceramic desk lamp with a brushed brass base, on a light grey surface",
    "Vehicle": "a red sports car with glossy paint, parked on a wet city street at dusk",
    "Character / creature": "a friendly robot with brushed steel plating and glowing blue eyes",
    "Landscape": "a lakeside cabin surrounded by pine forests and mountains at sunrise",
    "Other": "describe the object, its materials, colours and setting",
}


def validate_prompt(prompt: str | None) -> str:
    text = " ".join((prompt or "").split())
    if len(text) < 3:
        raise PromptError("Please describe the design you want (at least a few words).")
    if len(text) > MAX_PROMPT_CHARS:
        raise PromptError(f"The prompt is too long ({len(text)} characters; maximum {MAX_PROMPT_CHARS}).")
    return text


def build_prompt(user_prompt: str, style_key: str) -> str:
    """Combine the user's description with the selected style suffix."""
    style = STYLES.get(style_key) or STYLES["photorealistic"]
    base = validate_prompt(user_prompt).rstrip(".,; ")
    return f"{base}. {style.suffix}"
