"""
config.py - central configuration for Sketch2Reality.

All settings come from environment variables (optionally loaded from a local
`.env` file). Nothing secret is ever hard-coded, printed or shown in the UI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

APP_NAME = "Sketch2Reality"
APP_VERSION = "1.0.0"


def _get_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except ValueError:
        value = default
    return max(lo, min(hi, value))


def _get_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
    except ValueError:
        value = default
    return max(lo, min(hi, value))


def _looks_like_placeholder(key: str) -> bool:
    k = key.strip().lower()
    return (not k) or k.startswith("your_") or "paste" in k or "xxxx" in k or k == "changeme"


@dataclass(frozen=True)
class Settings:
    provider: str                    # default provider key shown in the UI
    stability_api_key: str           # NEVER log or display this
    stability_api_base: str
    stability_credits_per_image: float
    credit_price_usd: float
    request_timeout_s: int
    max_image_side: int
    max_upload_mb: int
    max_variations: int
    server_port: int

    @property
    def stability_key_configured(self) -> bool:
        return not _looks_like_placeholder(self.stability_api_key)


def load_settings() -> Settings:
    """Read settings from the environment (call again to pick up changes)."""
    return Settings(
        provider=os.getenv("SKETCH2REALITY_PROVIDER", "stability").strip().lower() or "stability",
        stability_api_key=os.getenv("STABILITY_API_KEY", "").strip(),
        stability_api_base=os.getenv("STABILITY_API_BASE", "https://api.stability.ai").strip().rstrip("/"),
        # Published price at the time of writing: Control Sketch = 5 credits, 1 credit = $0.01.
        # Configurable so the cost display stays correct if Stability changes its pricing.
        stability_credits_per_image=_get_float("STABILITY_CREDITS_PER_IMAGE", 5.0, 0.0, 1000.0),
        credit_price_usd=_get_float("CREDIT_PRICE_USD", 0.01, 0.0, 100.0),
        request_timeout_s=_get_int("REQUEST_TIMEOUT_SECONDS", 120, 10, 600),
        max_image_side=_get_int("MAX_IMAGE_SIDE", 1024, 256, 2048),
        max_upload_mb=_get_int("MAX_UPLOAD_MB", 10, 1, 50),
        max_variations=_get_int("MAX_VARIATIONS", 3, 1, 4),
        server_port=_get_int("SERVER_PORT", 7860, 1024, 65535),
    )


def redact(text: str, *secrets: str) -> str:
    """Remove any secret values from a string before it is shown or logged."""
    for secret in secrets:
        if secret and len(secret) >= 6:
            text = text.replace(secret, "[hidden]")
    return text
