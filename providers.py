"""
providers.py - provider abstraction (AI pipeline step 3: conditional generation).

The UI only talks to `BaseProvider.generate()`. To add another hosted API, subclass
BaseProvider, fill in `capabilities`, implement `generate()`, and register it in
PROVIDERS at the bottom of this file. The frontend does not need to change.

Implemented providers
  stability : Stability AI "Stable Image Control: Sketch"  (hosted, sketch-conditioned, paid credits)
  offline   : classical image-processing stylizer          (local, free, NOT generative AI)

Stability request format verified against the official API reference
(https://docs.stability.ai/api-reference/control/sketch):
  POST {base}/v2beta/stable-image/control/sketch      multipart/form-data
  headers: authorization: Bearer <key>, accept: image/*
  fields : image, prompt, [control_strength 0-1], [negative_prompt], [seed], [output_format], [style_preset]
  reply  : image bytes; headers `seed`, `finish-reason`, `x-request-id`
"""
from __future__ import annotations

import io
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import requests
from PIL import Image, UnidentifiedImageError

from config import Settings, redact
from offline_stylizer import stylize


# --------------------------------------------------------------------------- #
# Data classes and errors
# --------------------------------------------------------------------------- #
class ProviderError(Exception):
    """An error whose message is safe and helpful to show to the user."""

    def __init__(self, message: str, *, kind: str = "error"):
        super().__init__(message)
        self.kind = kind          # setup | auth | credits | rate_limit | timeout | network | rejected | server | error


@dataclass
class GenerationRequest:
    sketch_image: Image.Image
    sketch_png: bytes
    prompt: str
    negative_prompt: str = ""
    style_key: str = "photorealistic"
    style_preset: Optional[str] = None
    control_strength: float = 0.7
    seed: int = 0                       # 0 = random (Stability convention)


@dataclass
class GenerationResult:
    image: Image.Image
    provider: str
    model: str
    seed: Optional[int]
    elapsed_s: float
    credits_estimated: float
    request_id: str = ""


@dataclass(frozen=True)
class Capabilities:
    is_generative: bool
    uses_prompt: bool
    control_strength: bool
    negative_prompt: bool
    seed: bool
    style_preset: bool
    variations: bool
    requires_api_key: bool


class BaseProvider(ABC):
    key: str = ""
    label: str = ""
    model_label: str = ""
    capabilities: Capabilities

    def __init__(self, settings: Settings):
        self.settings = settings

    def is_configured(self) -> bool:
        return True

    def credits_per_image(self) -> float:
        return 0.0

    def setup_message(self) -> str:
        return ""

    @abstractmethod
    def generate(self, req: GenerationRequest) -> GenerationResult: ...


# --------------------------------------------------------------------------- #
# Stability AI - Control: Sketch
# --------------------------------------------------------------------------- #
class StabilitySketchProvider(BaseProvider):
    key = "stability"
    label = "Stability AI - Control: Sketch (hosted AI, recommended)"
    model_label = "Stable Image Control: Sketch (v2beta)"
    capabilities = Capabilities(
        is_generative=True, uses_prompt=True, control_strength=True, negative_prompt=True,
        seed=True, style_preset=True, variations=True, requires_api_key=True,
    )
    PATH = "/v2beta/stable-image/control/sketch"

    def is_configured(self) -> bool:
        return self.settings.stability_key_configured

    def credits_per_image(self) -> float:
        return self.settings.stability_credits_per_image

    def setup_message(self) -> str:
        return (
            "No Stability AI API key is configured yet. Create a key at "
            "https://platform.stability.ai (new accounts get 25 free credits), copy `.env.example` to `.env`, "
            "paste the key after `STABILITY_API_KEY=`, then restart the app. "
            "Until then you can try the free 'Offline stylizer' (non-AI fallback)."
        )

    # -- helpers ---------------------------------------------------------------
    def _scrub(self, text: str) -> str:
        return redact(text, self.settings.stability_api_key)

    @staticmethod
    def _error_details(resp: requests.Response) -> tuple[str, str]:
        """Return (error_name, human_message) from Stability's JSON error body."""
        try:
            body = resp.json()
            name = str(body.get("name", ""))
            errors = body.get("errors") or []
            msg = "; ".join(str(e) for e in errors) if errors else str(body)[:300]
            return name, msg
        except ValueError:
            return "", (resp.text or "")[:300].strip()

    def _raise_for_status(self, resp: requests.Response) -> None:
        code = resp.status_code
        name, detail = self._error_details(resp)
        detail = self._scrub(detail)
        low = f"{name} {detail}".lower()

        if code == 401 or (code == 403 and name != "content_moderation"):
            raise ProviderError("Stability AI rejected the API key (authentication failed). Check STABILITY_API_KEY in your .env file, "
                                "make sure there are no spaces or quotes around it, and restart the app.", kind="auth")
        if code == 402 or "insufficient" in low or "balance" in low or ("credit" in low and code >= 400):
            raise ProviderError("Your Stability AI account has insufficient credits. Check your balance and buy credits at "
                                "https://platform.stability.ai/dashboard/billing - or switch to the free offline stylizer.", kind="credits")
        if code == 403 and name == "content_moderation":
            raise ProviderError("Stability AI's content moderation blocked this request (you were not charged). "
                                "Try a different sketch or rephrase the prompt.", kind="rejected")
        if code == 413:
            raise ProviderError("The request was larger than the API's 10 MiB limit. Use a smaller sketch.", kind="rejected")
        if code == 422:
            if name == "invalid_language":
                raise ProviderError("Stability AI only accepts English prompts for this service. Please write the prompt in English.", kind="rejected")
            if name == "public_figure":
                raise ProviderError("The image seems to contain a public figure's likeness, which the API refuses. Use a different sketch.", kind="rejected")
            raise ProviderError(f"The request was rejected: {detail}", kind="rejected")
        if code == 429:
            raise ProviderError("Rate limit reached (the API allows 150 requests per 10 seconds and then pauses you for 60 seconds). "
                                "Wait a minute and try again.", kind="rate_limit")
        if code == 400:
            raise ProviderError(f"Stability AI says the request was invalid: {detail}", kind="rejected")
        if code >= 500:
            raise ProviderError("Stability AI had a temporary server error. You were not charged for a failed generation - try again shortly. "
                                "Status page: https://stabilityai.instatus.com/", kind="server")
        raise ProviderError(f"Unexpected response from Stability AI (HTTP {code}): {detail}", kind="error")

    # -- main call -------------------------------------------------------------
    def generate(self, req: GenerationRequest) -> GenerationResult:
        if not self.is_configured():
            raise ProviderError(self.setup_message(), kind="setup")
        s = self.settings
        url = f"{s.stability_api_base}{self.PATH}"
        headers = {
            "authorization": f"Bearer {s.stability_api_key}",
            "accept": "image/*",
            "stability-client-id": "Sketch2Reality",
            "stability-client-version": "1.0.0",
        }
        data = {
            "prompt": req.prompt,
            "control_strength": f"{req.control_strength:.2f}",
            "output_format": "png",
        }
        if req.negative_prompt.strip():
            data["negative_prompt"] = req.negative_prompt.strip()
        if req.seed and req.seed > 0:
            data["seed"] = str(int(req.seed))
        if req.style_preset:
            data["style_preset"] = req.style_preset

        start = time.perf_counter()
        resp = None
        for attempt in (1, 2):               # one automatic retry, only for 5xx (failed generations are not billed)
            try:
                resp = requests.post(
                    url, headers=headers, data=data,
                    files={"image": ("sketch.png", req.sketch_png, "image/png")},
                    timeout=(10, s.request_timeout_s),
                )
            except requests.exceptions.Timeout:
                raise ProviderError(f"The request timed out after {s.request_timeout_s}s. The service may be busy - try again. "
                                    "(Increase REQUEST_TIMEOUT_SECONDS in .env if this keeps happening.)", kind="timeout")
            except requests.exceptions.ConnectionError:
                raise ProviderError("Could not reach api.stability.ai. Check your internet connection, VPN or firewall.", kind="network")
            except requests.exceptions.RequestException as exc:
                raise ProviderError(self._scrub(f"Network error: {exc}"), kind="network")
            if resp.status_code >= 500 and attempt == 1:
                time.sleep(2)
                continue
            break
        elapsed = time.perf_counter() - start

        assert resp is not None
        if resp.status_code != 200:
            self._raise_for_status(resp)

        if resp.headers.get("finish-reason", "SUCCESS").upper() == "CONTENT_FILTERED":
            raise ProviderError("The generated image was blurred by Stability AI's content filter. Try a different prompt or sketch.", kind="rejected")
        if not resp.headers.get("content-type", "").lower().startswith("image/"):
            raise ProviderError("Stability AI returned something that is not an image. Please try again.", kind="server")
        try:
            image = Image.open(io.BytesIO(resp.content))
            image.load()
        except (UnidentifiedImageError, OSError):
            raise ProviderError("The returned image could not be decoded. Please try again.", kind="server")

        seed_hdr = resp.headers.get("seed", "")
        return GenerationResult(
            image=image.convert("RGB"), provider=self.label, model=self.model_label,
            seed=int(seed_hdr) if seed_hdr.isdigit() else None, elapsed_s=elapsed,
            credits_estimated=self.credits_per_image(), request_id=resp.headers.get("x-request-id", ""),
        )


# --------------------------------------------------------------------------- #
# Offline stylizer (non-generative)
# --------------------------------------------------------------------------- #
class OfflineStylizerProvider(BaseProvider):
    key = "offline"
    label = "Offline stylizer (free, NON-generative fallback)"
    model_label = "Classical image processing (flood-fill + shading) - not AI"
    capabilities = Capabilities(
        is_generative=False, uses_prompt=False, control_strength=False, negative_prompt=False,
        seed=True, style_preset=False, variations=False, requires_api_key=False,
    )

    def generate(self, req: GenerationRequest) -> GenerationResult:
        start = time.perf_counter()
        image = stylize(req.sketch_image, req.style_key, req.seed)
        return GenerationResult(
            image=image, provider=self.label, model=self.model_label,
            seed=req.seed or None, elapsed_s=time.perf_counter() - start, credits_estimated=0.0,
        )


# --------------------------------------------------------------------------- #
# Registry - add new providers here
# --------------------------------------------------------------------------- #
PROVIDERS: dict[str, type[BaseProvider]] = {
    StabilitySketchProvider.key: StabilitySketchProvider,
    OfflineStylizerProvider.key: OfflineStylizerProvider,
}


def get_provider(key: str, settings: Settings) -> BaseProvider:
    cls = PROVIDERS.get(key) or PROVIDERS["stability"]
    return cls(settings)


def provider_choices() -> list[tuple[str, str]]:
    return [(cls.label, cls.key) for cls in PROVIDERS.values()]
