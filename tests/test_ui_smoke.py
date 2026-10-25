"""
Smoke test of the real Gradio app: starts the server on a free port and drives the event handlers
through Gradio's own client API, exactly as the browser does.

The Stability HTTP call is SIMULATED here (mock) because tests must not spend credits.
Run:  python -m pytest -q tests/test_ui_smoke.py
"""
import io
import sys
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

gradio_client = pytest.importorskip("gradio_client")
import app as s2r                                       # noqa: E402
import evaluation                                       # noqa: E402
from dataclasses import replace                         # noqa: E402

SAMPLE = str(ROOT / "samples" / "house.png")


def _png(size=(512, 384)):
    buf = io.BytesIO(); Image.new("RGB", size, (90, 140, 200)).save(buf, "PNG"); return buf.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    s2r.EVAL_LOG = evaluation.EvalLog(tmp_path_factory.mktemp("log") / "log.csv")
    s2r.SETTINGS = replace(s2r.SETTINGS, stability_api_key="sk-test-1234567890abcdef")
    demo = s2r.build_demo(); demo.queue(default_concurrency_limit=1)
    _, url, _ = demo.launch(server_name="127.0.0.1", server_port=None, prevent_thread_lock=True, quiet=True,
                            theme=s2r.THEME, css=s2r.CSS, allowed_paths=[str(ROOT / "samples")])
    yield gradio_client.Client(url, verbose=False)
    demo.close()


def _generate(client, provider, variations=1, prompt="a modern house"):
    return client.predict(
        gradio_client.handle_file(SAMPLE), prompt, "photorealistic", "House / building", provider,
        0.7, "blurry", 0, variations, True, "auto", api_name="/on_generate")


def test_offline_generation_through_ui(client):
    out = _generate(client, "offline", prompt="")
    status, compare, gallery, info, download, clean = out   # gr.State is not part of the client API
    assert "Generation complete" in status and "not generative AI" in info
    assert compare is not None and Path(download["value"] if isinstance(download, dict) else download).exists()


def test_stability_generation_through_ui_with_mocked_http(client):
    resp = mock.Mock(); resp.status_code = 200; resp.content = _png()
    resp.headers = {"content-type": "image/png", "seed": "777", "finish-reason": "SUCCESS"}
    with mock.patch("providers.requests.post", return_value=resp) as post:
        status, compare, gallery, info, download, clean = _generate(client, "stability", variations=2)
    assert post.call_count == 2 and "Generation complete" in status
    assert "10 (about $0.10)" in info and "seed 777" in info
    assert len(s2r.EVAL_LOG.records) >= 3


def test_error_paths_show_messages_not_crashes(client):
    status = client.predict(None, "x house", "photorealistic", "Other", "stability", 0.7, "", 0, 1, True, "auto",
                            api_name="/on_generate")[0]
    assert "No sketch uploaded" in status
    resp = mock.Mock(); resp.status_code = 401; resp.headers = {}
    resp.json.return_value = {"name": "unauthorized", "errors": ["bad"]}; resp.text = ""
    with mock.patch("providers.requests.post", return_value=resp):
        status = _generate(client, "stability")[0]
    assert "API key rejected" in status and "sk-test-1234567890abcdef" not in status


def test_evaluation_tab_flow(client):
    table, dd = client.predict(api_name="/refresh_eval")[:2]
    run_id = s2r.EVAL_LOG.records[-1].run_id
    pair, text = client.predict(run_id, api_name="/on_select_run")
    saved = client.predict(run_id, 4, 3, "0.30", "demo note", api_name="/on_save_rating")
    assert "Saved ratings" in saved
    rec = s2r.EVAL_LOG.get(run_id)
    assert rec.human_visual_quality_1to5 == "4" and rec.human_structure_preservation_1to5 == "3"
    bad = client.predict(run_id, 4, 3, "abc", "", api_name="/on_save_rating")
    assert "must be a number" in bad
    exported = client.predict(api_name="/on_export")
    path = exported["value"] if isinstance(exported, dict) else exported
    assert Path(path).read_text(encoding="utf-8-sig").startswith("run_id,")


def test_provider_switch_and_cost_text(client):
    cost = client.predict("stability", 3, api_name="/on_provider_change")   # layout Group is not in the client API
    assert "3 x 5 = 15 credits" in cost and "$0.15" in cost
    cost = client.predict("offline", 1, api_name="/on_provider_change")
    assert "free" in cost


def test_batch_requires_confirmation_then_runs(client):
    files = [gradio_client.handle_file(SAMPLE), gradio_client.handle_file(str(ROOT / "samples" / "chair.png"))]
    args = ["a realistic {name}", "Other", "photorealistic", "stability", 0.7, "", 0, True, "auto"]
    msg = client.predict(files, *args, False, api_name="/on_batch")
    assert "confirm the cost" in msg and "10 credits" in msg
    before = len(s2r.EVAL_LOG.records)
    resp = mock.Mock(); resp.status_code = 200; resp.content = _png()
    resp.headers = {"content-type": "image/png", "seed": "5", "finish-reason": "SUCCESS"}
    with mock.patch("providers.requests.post", return_value=resp) as post:
        msg = client.predict(files, *args, True, api_name="/on_batch")
    assert post.call_count == 2 and "2 of 2 succeeded" in msg and len(s2r.EVAL_LOG.records) == before + 2
