# Sketch2Reality

Turn a rough hand-drawn sketch into a realistic design image, using a **hosted** sketch-conditioned diffusion model.
Mini-project for **Multimodal Generative AI**.

- Runs on a normal Windows laptop: **no GPU, no CUDA, no Docker, no model download**.
- The AI inference runs on Stability AI's servers; your PC only uploads the sketch and receives the image.
- Gradio web interface, Python backend. The pip packages take roughly 350 MB on disk (measured on Linux, including the optional test tool); there are no model weights.

---

## 0. Read this first: cost and what was verified

**Cost.** Real AI generation uses a **paid API**. Nothing here is hidden:

| Item | Value | Source |
|---|---|---|
| Service used | Stability AI *Stable Image Control: Sketch* | official docs |
| Price per successful image | **5 credits** | https://docs.stability.ai/pricing |
| Credit value | **1 credit = $0.01** (so about **$0.05 per image**) | same page |
| Free credits | **25 free credits** advertised for new accounts (about 5 images) | same page |
| Failed generations | Not charged (per the API reference) | https://docs.stability.ai/api-reference/control/sketch |

The 25 free credits are Stability's current offer and can change; they are not guaranteed. A free, non-AI fallback ("Offline stylizer") is included so the app is always usable, but it is simple colouring, **not** generative AI.

**What was verified, and what was not.**

| Verified (by the author, before delivery) | NOT verified (you must check) |
|---|---|
| Endpoint, parameters, limits, error codes and pricing read from Stability's **official** API reference and pricing page | A **live call** to the Stability API with a real key (no key or internet access to it in the build environment) |
| 34 automated tests pass on Linux, Python 3.12, gradio 6.30.0: preprocessing, request format, error handling, logging, CSV, and the real Gradio app driven through its API | Running on **Windows** and rendering in a **real browser** (the visual layout was not seen on screen) |
| In those tests the Stability HTTP response is **simulated** | Image quality and exactly how well your sketches are followed |

So the first thing to do after setup is one real generation to confirm everything works with your key (section 6).

I only verified Stability AI's documentation. I did not verify the current endpoints or prices of other hosted providers, so no comparison with them is claimed. The provider design (section 9) makes adding one a small change.

---

## 1. Python version

Install **Python 3.12 (64-bit)** from https://www.python.org/downloads/windows/
and tick **"Add python.exe to PATH"** in the installer. (Alternative in PowerShell: `winget install Python.Python.3.12`.)

The project was tested with Python 3.12. Gradio 6 needs Python 3.10 or newer. Newer versions probably work but were not tested.

Check:

```powershell
py -3.12 --version
```

## 2. Open the project in VS Code

1. Unzip `Sketch2Reality.zip` (for example to `C:\Projects\Sketch2Reality`).
2. Start VS Code, choose **File > Open Folder...** and select the `Sketch2Reality` folder (the one that contains `app.py`).
3. Install the **Python** extension by Microsoft if VS Code offers it.
4. Open a terminal: **Terminal > New Terminal** (it opens PowerShell in the project folder).

## 3. Create and activate a virtual environment (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` at the start of the prompt.
If PowerShell says *running scripts is disabled*, run this once and activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Get the API key and configure `.env`

1. Go to https://platform.stability.ai and create an account (new accounts are advertised to get 25 free credits).
2. Open the **API keys** page of the dashboard (https://platform.stability.ai/dashboard/api-keys) and create a key. Copy it.
3. Create your private settings file and open it:

```powershell
Copy-Item .env.example .env
code .env
```

4. Replace `your_stability_api_key_here` with your key, with no quotes and no spaces:

```
STABILITY_API_KEY=sk-PASTE_YOUR_REAL_KEY_HERE
```

5. Save. Never share this file; it is excluded by `.gitignore`.

## 6. Launch the app

```powershell
python app.py
```

Your browser opens at http://127.0.0.1:7860 (if not, open it yourself). Stop the app with **Ctrl+C**.

**First test (about 5 credits):** click one of the *Sample sketches* under the Studio tab, press **Generate realistic design**, and wait for the result. If you see a result and a before/after slider, everything works. If you don't have a key yet, the app still starts and shows a setup message; the *Offline stylizer* generator works without a key.

---

## 7. Using the app

**Studio tab**

1. Drop a sketch (PNG/JPG/WEBP/BMP/GIF/TIFF, up to 10 MB). Photos of paper sketches work too (webcam and clipboard are also supported).
2. Choose a category (used in the evaluation log) and write the prompt **in English**. The *Insert an example prompt* button helps.
3. Choose a style: Photorealistic, Architectural visualization, Product render, Interior design, Minimalist, Concept art.
4. *(Optional)* **Advanced settings**: sketch adherence (control strength), negative prompt, style preset, seed, variations, paper clean-up.
5. Press **Generate**. You get a before/after slider, a download button, generation time, provider/model, seed, an estimated credit count, and a view of the cleaned sketch the model received.

**Batch & evaluation tab**: batch-run several sketches, rate results 1-5, view the log, export it as CSV (opens cleanly in Excel).

**How it works tab**: the AI concepts, limitations and evaluation notes for your report.

Generated images are written only to your Windows temp folder (`%TEMP%\sketch2reality`), never to the project folder. Use **Download result (PNG)** or the download icon on the image to keep one. Only the text evaluation log is stored in the project (`evaluation/evaluation_log.csv`, no images, no keys).

---

## 8. How it works (pipeline)

```
upload -> validate + clean (preprocess.py) -> prompt + style (prompts.py)
       -> provider.generate() (providers.py) -> time, log, proxy score (pipeline.py, evaluation.py) -> UI (app.py)
```

| File | Purpose |
|---|---|
| `app.py` | Gradio interface and event handlers. Calls the real backend, no placeholders |
| `pipeline.py` | The generation workflow (UI-independent, unit-tested) |
| `preprocess.py` | Upload validation; flattens paper lighting, strengthens faint lines, inverts chalk-style drawings, handles transparency, resizes, pads extreme aspect ratios |
| `prompts.py` | Style definitions, negative prompt, prompt construction |
| `providers.py` | `BaseProvider` abstraction, **Stability AI Control: Sketch** provider, offline provider |
| `offline_stylizer.py` | Non-generative fallback (flood-fill colouring and shading) |
| `evaluation.py` | CSV evaluation log, edge-based structure proxy |
| `config.py` | Environment settings, secret redaction |
| `samples/` | Three demo sketches (made by `tools/make_samples.py`) |
| `evaluation/evaluation_template.csv` | Empty CSV with the column headers |
| `tests/` | Automated tests (simulated HTTP, no credits used) |

**Request sent to Stability** (from the official reference): `POST https://api.stability.ai/v2beta/stable-image/control/sketch`, multipart form, header `authorization: Bearer <key>`, `accept: image/*`; fields `image`, `prompt`, `control_strength` (0-1, default 0.7), `negative_prompt`, `seed`, `output_format`, `style_preset`. The output has the same size as the input image. Limits: sides at least 64 px, at most about 9.4 megapixels, aspect ratio between 1:2.5 and 2.5:1, request under 10 MiB, English prompts only, 150 requests per 10 seconds.

**Honest limits of the model.** It *follows* the sketch but does not copy it exactly; fine details may be dropped or reinterpreted, and quality varies between runs. Raise *sketch adherence* for stricter outlines and try several seeds. Sketch2Reality does not claim perfect structure preservation, and the "How it works" tab says so.

---

## 9. Add another hosted provider later

1. Open `providers.py`, subclass `BaseProvider`, set `key`, `label`, `model_label`, `capabilities`.
2. Implement `is_configured()` and `generate(req) -> GenerationResult`, raising `ProviderError` with a friendly message.
3. Add it to the `PROVIDERS` dict at the bottom. It appears in the *Generator* dropdown automatically. Check the new provider's **official** docs for the endpoint, parameters and price first.
4. Add its key to `.env.example` and `config.py`.

The UI shows or hides advanced controls according to `capabilities`, so the frontend needs no changes.

---

## 10. Academic evaluation

Every generation adds one row to the log. **Three kinds of values are kept separate:**

| Kind | Columns | Notes |
|---|---|---|
| **Measured** (automatic) | `generation_time_s`, `provider`, `model`, `seed_used`, `control_strength`, `credits_estimated` | Time is measured around the API call. Credits are an estimate from the published price; your dashboard is the truth |
| **Human ratings** (you) | `human_visual_quality_1to5`, `human_structure_preservation_1to5`, `notes` | Blank until you rate. Never pre-filled |
| **Experimental proxy** (automatic) | `structure_proxy_edge_recall`, `structure_proxy_chance_level` | Share of the sketch's line pixels lying near strong edges in the output, next to the share a random line would get by chance. Only values clearly above chance mean anything. Not validated; use it as a hint, not as a result |
| **Optional, manual** | `clip_similarity_optional_manual` | Not computed by the app. Running CLIP locally needs PyTorch and hundreds of MB of weights, which breaks the project's constraints, and I did not verify a hosted CLIP endpoint. If you compute a CLIPScore elsewhere (for example in a free Colab notebook), type it in |

**Suggested protocol:** use 5-10 sketches across several categories; use the same style and settings per comparison; generate; rate each result immediately (1-5 on both scales); export the CSV; summarise the averages per category/style and discuss failures honestly. Keep in mind that a handful of ratings from one person is a small, subjective sample.

**Why not PSNR/SSIM?** They compare an output with an **aligned ground-truth image**, pixel by pixel. A sketch has many equally valid realistic renderings, and there is no single correct photo. A good photo-like render differs hugely from a line drawing at pixel level, so PSNR/SSIM would score good results as bad. They only make sense if you have paired ground-truth images. For creative generation use human ratings, CLIPScore (prompt agreement), and distribution metrics such as FID on large sets.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `py` is not recognised | Reinstall Python and tick *Add python.exe to PATH*, or use `python -m venv .venv` |
| `Activate.ps1 cannot be loaded ... disabled` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then activate again |
| `ModuleNotFoundError: gradio` (or another module) | The environment is not active. Run `.\.venv\Scripts\Activate.ps1` and `pip install -r requirements.txt` |
| Banner says "One-time setup needed" | `.env` is missing, still has the placeholder, or the app was not restarted after editing it |
| "API key rejected" | Re-copy the key into `.env` (no quotes/spaces), restart the app, check the key is active in the dashboard |
| "insufficient credits" | Add credits at https://platform.stability.ai/dashboard/billing, or switch *Generator* to *Offline stylizer*. (HTTP 402 handling is defensive: it is not listed in the current API schema, but the app recognises it and credit/balance wording in error messages) |
| "Rate limit reached" | Wait one minute. The API pauses you for 60 seconds after exceeding its limit |
| Timeout / "Could not reach api.stability.ai" | Check internet, VPN, proxy or firewall; increase `REQUEST_TIMEOUT_SECONDS` in `.env` |
| "Only English prompts" | Write the prompt in English |
| "content moderation" | The request or sketch was flagged; change the prompt or sketch (you are not charged) |
| Result ignores the sketch | Raise *Sketch adherence* to 0.85-1.0, use darker, closed outlines, keep *Clean paper* on, mention the main object in the prompt |
| Result looks too "traced" or stiff | Lower *Sketch adherence* to about 0.5 |
| Port 7860 is in use | Set `SERVER_PORT=7861` in `.env` and restart |
| Page looks unstyled / different font offline | The font comes from Google Fonts; the layout still works with a fallback font |
| Browser did not open | Open http://127.0.0.1:7860 manually |
| Sample sketches missing | `python tools/make_samples.py` recreates them |

Run the tests (optional): `pip install -r requirements-dev.txt` then `python -m pytest -q`.

---

## 12. Managing cost and usage

- One click with 1 variation = **5 credits (about $0.05)**. Each extra variation is another paid generation; the app shows the cost under the settings before you press Generate.
- Batch runs cost 5 credits per sketch and require a confirmation tick. They stop early on key/credit/rate-limit errors.
- The Studio tab reports credits used this session. These are **estimates**; the dashboard shows the true balance.
- Practice with the free Offline stylizer, and spend credits only on final runs.
- Keep variations at 1 while experimenting. The price is a flat rate per generation, so a smaller `MAX_IMAGE_SIDE` only makes runs faster; it does not make them cheaper.
- $10 of credits is about 1,000 credits, which is about 200 generations at the published price.
- Watch the dashboard for your account's own limits and billing controls; I did not verify what spending caps exist.

---

## 13. Likely questions

- *Why not run Stable Diffusion + ControlNet locally?* No GPU: it would be very slow, memory heavy and give poor results on CPU. A hosted service moves the compute to remote GPUs.
- *Is this ControlNet?* ControlNet is the best-known published method for this kind of structural conditioning, and we use it as the conceptual reference. Stability's API page does not document its internals, so I do not claim it is ControlNet.
- *What does control strength do?* It sets how strongly the sketch constrains generation (0-1, default 0.7).
- *How is text and image combined?* The prompt gives semantics, the sketch gives spatial layout; both condition the denoising process.
- *How did you evaluate?* Human ratings (primary), measured time, an experimental edge proxy with a chance baseline, and optional CLIPScore. PSNR/SSIM are not suitable without aligned references.
- *What are the limitations?* Sketch is followed, not copied; English prompts; costs credits; depends on internet and a third-party service; results vary per seed.
- *What does CLIP do here?* It is a vision-language model that aligns images and text. Many diffusion models use it for text conditioning and CLIPScore measures prompt agreement. The app does not run it locally.

---

## 14. Security and privacy

- The API key lives only in `.env` (ignored by git). It is never shown in the UI or printed, and error messages are scrubbed of it.
- Your sketches and prompts are sent to Stability AI for processing, so do not upload confidential material. Read their terms for retention details; I did not verify them.
- The app is bound to `127.0.0.1`.
- Gradio analytics are switched off in `app.py`.

---

*Method note: library versions were pinned to the tested set in `requirements.txt` (gradio 6.30.0 and pandas below 3). If you upgrade Gradio, re-run the tests: Gradio 6 changed where `theme` and `css` are passed.*
