"""
Offline tests for Sketch2Reality.

IMPORTANT: these tests use a SIMULATED Stability AI HTTP response (unittest.mock). They verify our
request format, error handling, preprocessing and logging. They do NOT prove the live API accepts the
request - only a real run with your API key can do that.

Run from the project folder:   python -m pytest -q
"""
import io
import sys
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                   # noqa: E402
import evaluation                              # noqa: E402
import pipeline                                # noqa: E402
import preprocess                              # noqa: E402
import prompts                                 # noqa: E402
import providers                               # noqa: E402

SAMPLE = ROOT / "samples" / "house.png"
SETTINGS = replace(config.load_settings(), stability_api_key="sk-test-1234567890abcdef",
                   stability_api_base="https://api.stability.ai")


def png_bytes(size=(256, 256), color=(120, 160, 90)):
    buf = io.BytesIO(); Image.new("RGB", size, color).save(buf, "PNG"); return buf.getvalue()


def fake_response(status=200, content=None, headers=None, json_body=None):
    r = mock.Mock(spec=requests.Response)
    r.status_code = status
    r.content = content if content is not None else b""
    r.headers = {"content-type": "image/png", "seed": "4242", "finish-reason": "SUCCESS", "x-request-id": "req-1"}
    r.headers.update(headers or {})
    if json_body is not None:
        r.json.return_value = json_body
        r.text = str(json_body)
    else:
        r.json.side_effect = ValueError("no json")
        r.text = ""
    return r


def make_request():
    prep = preprocess.prepare_sketch(SAMPLE)
    return providers.GenerationRequest(sketch_image=prep.image, sketch_png=prep.png_bytes,
                                       prompt="a house. photorealistic", negative_prompt="blurry",
                                       style_preset="photographic", control_strength=0.7, seed=99)


# ----------------------------------------------------------------- preprocessing
def test_sample_preprocess_is_api_ready():
    p = preprocess.prepare_sketch(SAMPLE)
    assert p.image.mode == "RGB" and max(p.image.size) <= 1024 and min(p.image.size) >= 64
    assert len(p.png_bytes) < 10 * 1024 * 1024
    arr = np.asarray(p.image.convert("L"))
    assert arr.max() == 255 and arr.min() < 100                       # white paper, dark lines


def test_uneven_lighting_is_flattened(tmp_path):
    img = Image.new("L", (600, 400), 255); d = ImageDraw.Draw(img)
    d.rectangle((100, 100, 500, 300), outline=40, width=4)
    arr = np.asarray(img, dtype=np.float32)
    shade = np.linspace(0.45, 1.0, 600)[None, :]                       # dark on the left like a bad phone photo
    path = tmp_path / "photo.jpg"; Image.fromarray((arr * shade).astype(np.uint8)).save(path)
    out = np.asarray(preprocess.prepare_sketch(path).image.convert("L"))
    assert np.median(out[:, :60]) >= 250 and np.median(out[:, -60:]) >= 250      # paper white on both sides


def test_transparent_and_inverted_inputs(tmp_path):
    rgba = Image.new("RGBA", (400, 300), (0, 0, 0, 0)); ImageDraw.Draw(rgba).line((20, 20, 380, 280), fill=(0, 0, 0, 255), width=5)
    (tmp_path / "t.png").write_bytes(b""); rgba.save(tmp_path / "t.png")
    assert np.asarray(preprocess.prepare_sketch(tmp_path / "t.png").image.convert("L")).mean() > 200
    chalk = Image.new("L", (400, 300), 15); ImageDraw.Draw(chalk).line((20, 20, 380, 280), fill=240, width=5)
    chalk.save(tmp_path / "c.png")
    res = preprocess.prepare_sketch(tmp_path / "c.png")
    assert np.asarray(res.image.convert("L")).mean() > 200 and any("inverted" in n for n in res.notes)


def test_extreme_aspect_is_padded_and_tiny_is_enlarged(tmp_path):
    Image.new("RGB", (2000, 200), "white").save(tmp_path / "wide.png")
    w, h = preprocess.prepare_sketch(tmp_path / "wide.png").image.size
    assert max(w, h) / min(w, h) <= 2.5
    Image.new("RGB", (100, 80), "white").save(tmp_path / "small.png")
    assert max(preprocess.prepare_sketch(tmp_path / "small.png").image.size) >= 512


@pytest.mark.parametrize("name,payload,fragment", [
    ("fake.png", b"this is not an image", "not a valid image"),
    ("doc.txt", b"hello", "Unsupported file type"),
])
def test_invalid_uploads_give_friendly_errors(tmp_path, name, payload, fragment):
    f = tmp_path / name; f.write_bytes(payload)
    with pytest.raises(preprocess.InvalidImageError, match=fragment):
        preprocess.prepare_sketch(f)


def test_missing_and_oversized_files(tmp_path):
    with pytest.raises(preprocess.InvalidImageError):
        preprocess.prepare_sketch(None)
    big = tmp_path / "big.png"; big.write_bytes(b"0" * (2 * 1024 * 1024))
    with pytest.raises(preprocess.InvalidImageError, match="limit"):
        preprocess.prepare_sketch(big, max_upload_mb=1)


# ---------------------------------------------------------------------- prompts
def test_prompt_building_and_validation():
    text = prompts.build_prompt("a modern house.", "architectural")
    assert text.startswith("a modern house.") and "architectural visualization" in text
    with pytest.raises(prompts.PromptError):
        prompts.validate_prompt("  ")
    assert set(s.stability_preset for s in prompts.STYLES.values() if s.stability_preset) <= set(prompts.STABILITY_STYLE_PRESETS)


# -------------------------------------------------------------------- providers
def test_stability_request_format_matches_official_docs():
    prov = providers.StabilitySketchProvider(SETTINGS)
    with mock.patch("providers.requests.post", return_value=fake_response(content=png_bytes())) as post:
        result = prov.generate(make_request())
    args, kw = post.call_args
    assert args[0] == "https://api.stability.ai/v2beta/stable-image/control/sketch"
    assert kw["headers"]["authorization"] == "Bearer " + SETTINGS.stability_api_key
    assert kw["headers"]["accept"] == "image/*"
    assert set(kw["files"]) == {"image"}
    d = kw["data"]
    assert d["prompt"] and d["control_strength"] == "0.70" and d["seed"] == "99"
    assert d["negative_prompt"] == "blurry" and d["style_preset"] == "photographic" and d["output_format"] == "png"
    assert result.seed == 4242 and result.image.size == (256, 256) and result.elapsed_s >= 0


def test_zero_seed_and_empty_options_are_omitted():
    prov = providers.StabilitySketchProvider(SETTINGS)
    req = make_request(); req.seed = 0; req.negative_prompt = ""; req.style_preset = None
    with mock.patch("providers.requests.post", return_value=fake_response(content=png_bytes())) as post:
        prov.generate(req)
    d = post.call_args.kwargs["data"]
    assert "seed" not in d and "negative_prompt" not in d and "style_preset" not in d


@pytest.mark.parametrize("status,body,kind", [
    (401, {"id": "x", "name": "unauthorized", "errors": ["bad key"]}, "auth"),
    (402, {"id": "x", "name": "insufficient_balance", "errors": ["no credits"]}, "credits"),
    (403, {"id": "x", "name": "content_moderation", "errors": ["flagged"]}, "rejected"),
    (403, {"id": "x", "name": "forbidden", "errors": ["nope"]}, "auth"),
    (413, {"id": "x", "name": "payload_too_large", "errors": ["big"]}, "rejected"),
    (422, {"id": "x", "name": "invalid_language", "errors": ["English only"]}, "rejected"),
    (429, {"id": "x", "name": "rate_limit_exceeded", "errors": ["slow down"]}, "rate_limit"),
    (400, {"id": "x", "name": "bad_request", "errors": ["prompt: required"]}, "rejected"),
])
def test_error_mapping(status, body, kind):
    prov = providers.StabilitySketchProvider(SETTINGS)
    with mock.patch("providers.requests.post", return_value=fake_response(status=status, json_body=body)):
        with pytest.raises(providers.ProviderError) as exc:
            prov.generate(make_request())
    assert exc.value.kind == kind


def test_server_error_retries_once_then_reports(monkeypatch):
    monkeypatch.setattr("providers.time.sleep", lambda s: None)
    prov = providers.StabilitySketchProvider(SETTINGS)
    seq = [fake_response(status=500, json_body={"name": "internal_error", "errors": ["oops"]}),
           fake_response(content=png_bytes())]
    with mock.patch("providers.requests.post", side_effect=seq) as post:
        assert prov.generate(make_request()).image.size == (256, 256)
    assert post.call_count == 2
    with mock.patch("providers.requests.post", return_value=fake_response(status=503, json_body={"name": "x", "errors": ["down"]})) as post:
        with pytest.raises(providers.ProviderError) as exc:
            prov.generate(make_request())
    assert exc.value.kind == "server" and post.call_count == 2


def test_timeout_and_connection_errors():
    prov = providers.StabilitySketchProvider(SETTINGS)
    with mock.patch("providers.requests.post", side_effect=requests.exceptions.Timeout()):
        with pytest.raises(providers.ProviderError) as exc:
            prov.generate(make_request())
    assert exc.value.kind == "timeout"
    with mock.patch("providers.requests.post", side_effect=requests.exceptions.ConnectionError()):
        with pytest.raises(providers.ProviderError) as exc:
            prov.generate(make_request())
    assert exc.value.kind == "network"


def test_content_filtered_output_is_reported():
    prov = providers.StabilitySketchProvider(SETTINGS)
    with mock.patch("providers.requests.post", return_value=fake_response(content=png_bytes(), headers={"finish-reason": "CONTENT_FILTERED"})):
        with pytest.raises(providers.ProviderError, match="blurred"):
            prov.generate(make_request())


def test_missing_key_never_calls_network_and_key_never_leaks():
    no_key = replace(SETTINGS, stability_api_key="")
    prov = providers.StabilitySketchProvider(no_key)
    assert not prov.is_configured()
    with mock.patch("providers.requests.post") as post:
        with pytest.raises(providers.ProviderError) as exc:
            prov.generate(make_request())
    assert exc.value.kind == "setup" and not post.called
    prov2 = providers.StabilitySketchProvider(SETTINGS)
    leaky = fake_response(status=400, json_body={"name": "bad", "errors": [f"key {SETTINGS.stability_api_key} is wrong"]})
    with mock.patch("providers.requests.post", return_value=leaky):
        with pytest.raises(providers.ProviderError) as exc:
            prov2.generate(make_request())
    assert SETTINGS.stability_api_key not in str(exc.value)
    assert config.redact("abc sk-secret-value-123 xyz", "sk-secret-value-123") == "abc [hidden] xyz"


def test_placeholder_key_counts_as_unconfigured():
    for k in ("", "your_stability_api_key_here", "PASTE_KEY"):
        assert not replace(SETTINGS, stability_api_key=k).stability_key_configured


# --------------------------------------------------------------- pipeline + log
def test_pipeline_end_to_end_with_mocked_api_and_logging(tmp_path):
    log = evaluation.EvalLog(tmp_path / "log.csv")
    with mock.patch("providers.requests.post", return_value=fake_response(content=png_bytes((512, 384)))):
        out = pipeline.run_generation(str(SAMPLE), "a modern house", "photorealistic", "House / building", "stability",
                                      settings=SETTINGS, eval_log=log, variations=2, seed=10)
    assert out.kind == "success" and len(out.variants) == 2 and out.credits == 10
    assert len(log.records) == 2 and log.records[0].provider == "stability"
    assert log.records[0].human_visual_quality_1to5 == ""                       # never pre-filled
    assert Path(out.variants[0].image_path).is_file()
    log.rate(log.records[0].run_id, 4, 5, clip="0.31", notes="good")
    csv_path = log.export(tmp_path / "export.csv")
    text = csv_path.read_text(encoding="utf-8-sig")
    assert text.splitlines()[0].startswith("run_id,") and ",4,5,0.31," in text
    assert len(evaluation.EvalLog(tmp_path / "log.csv").records) == 2           # persisted


def test_pipeline_partial_failure_keeps_paid_results(tmp_path):
    log = evaluation.EvalLog(tmp_path / "log.csv")
    seq = [fake_response(content=png_bytes()), fake_response(status=429, json_body={"name": "rate_limit_exceeded", "errors": ["x"]})]
    with mock.patch("providers.requests.post", side_effect=seq):
        out = pipeline.run_generation(str(SAMPLE), "a house", "concept", "Other", "stability",
                                      settings=SETTINGS, eval_log=log, variations=3)
    assert out.kind == "partial" and len(out.variants) == 1 and len(log.records) == 1


def test_pipeline_user_errors(tmp_path):
    log = evaluation.EvalLog(tmp_path / "log.csv")
    o = pipeline.run_generation(None, "a house", "photorealistic", "Other", "stability", settings=SETTINGS, eval_log=log)
    assert o.kind == "error" and "No sketch" in o.message
    o = pipeline.run_generation(str(SAMPLE), " ", "photorealistic", "Other", "stability", settings=SETTINGS, eval_log=log)
    assert o.kind == "error"
    nokey = replace(SETTINGS, stability_api_key="")
    o = pipeline.run_generation(str(SAMPLE), "a house", "photorealistic", "Other", "stability", settings=nokey, eval_log=log)
    assert o.kind == "setup" and "API key" in o.title


def test_offline_provider_runs_without_network_and_is_free(tmp_path):
    log = evaluation.EvalLog(tmp_path / "log.csv")
    with mock.patch("providers.requests.post") as post:
        o = pipeline.run_generation(str(SAMPLE), "", "architectural", "House / building", "offline",
                                    settings=replace(SETTINGS, stability_api_key=""), eval_log=log)
    assert o.kind == "success" and o.credits == 0 and not post.called
    assert Image.open(o.variants[0].image_path).size == o.prepared.image.size


def test_structure_proxy_is_above_chance_for_matching_edges_and_low_for_unrelated():
    sketch = Image.open(SAMPLE).convert("RGB")
    matching = Image.open(SAMPLE).convert("RGB").filter(__import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(1))
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (sketch.size[1], sketch.size[0], 3), dtype=np.uint8))
    flat = Image.new("RGB", sketch.size, (128, 128, 128))
    r_match, chance_match = evaluation.edge_structure_proxy(sketch, matching)
    assert r_match > 0.85 and r_match > chance_match + 0.3
    assert evaluation.edge_structure_proxy(sketch, flat) is None               # flat image: no edges, must not crash
    r_noise, chance_noise = evaluation.edge_structure_proxy(sketch, noise)
    assert abs(r_noise - chance_noise) < 0.15                                    # unrelated -> near chance
