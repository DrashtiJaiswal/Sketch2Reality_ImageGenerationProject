"""ui_text.py - static text, HTML and CSS for the interface (kept out of app.py for readability)."""

HERO_HTML = """
<div class="s2r-hero">
  <div class="s2r-hero-copy">
    <h1>Sketch2Reality</h1>
    <p class="s2r-tag">Turn a rough line drawing into a realistic design image.</p>
    <p class="s2r-sub">Upload a sketch, describe what it should become, and a hosted diffusion model renders it
    while following your lines. No GPU and no model download on your computer.</p>
  </div>
  <svg class="s2r-art" viewBox="0 0 560 220" role="img" aria-label="The same house shown as a pencil outline on the left and as a coloured render on the right">
    <defs>
      <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8fc3e0"/><stop offset="1" stop-color="#dceef6"/></linearGradient>
      <linearGradient id="glass" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#bfe3f2"/><stop offset="1" stop-color="#6fa8c4"/></linearGradient>
    </defs>
    <g fill="none" stroke="#9aa5b1" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="7 5">
      <path d="M72 101 L161 46 L249 100"/><path d="M90 100 L90 191 L231 190 L230 100"/>
      <path d="M146 191 L146 141 L176 142 L175 190"/><path d="M105 116 L131 115 L130 140 L106 141 Z"/>
      <path d="M190 116 L215 117 L214 141 L189 140 Z"/><path d="M52 192 L262 191"/>
    </g>
    <path d="M262 24 L262 200" stroke="#6b7785" stroke-width="1.5" stroke-dasharray="2 6" stroke-linecap="round"/>
    <g transform="translate(290 0)">
      <rect x="0" y="0" width="270" height="192" fill="url(#sky)" opacity=".55" rx="4"/>
      <rect x="0" y="188" width="270" height="32" fill="#5f9a58" opacity=".9"/>
      <polygon points="72,101 161,46 249,100" fill="#3d5f7d"/>
      <rect x="90" y="100" width="141" height="91" fill="#e1e8ea"/>
      <rect x="146" y="141" width="30" height="50" fill="#85553a"/>
      <rect x="105" y="116" width="26" height="25" fill="url(#glass)"/><rect x="190" y="116" width="25" height="25" fill="url(#glass)"/>
      <ellipse cx="170" cy="196" rx="95" ry="6" fill="#000" opacity=".15"/>
    </g>
  </svg>
</div>
"""

ABOUT_MD = """
## How Sketch2Reality works

**Pipeline:** upload -> clean the sketch (Pillow/NumPy) -> combine text prompt + style -> hosted model
(Stability AI *Control: Sketch*) -> image returned -> measured, logged and shown next to your sketch.
Everything heavy happens on the provider's GPUs; your computer only sends and receives images.

### Concepts from the syllabus

| Concept | What it means here |
|---|---|
| **Image-to-image translation** | Mapping an input image (line drawing) to an output image (rendered design). Early deep-learning approach: *pix2pix* (Isola et al., 2017, conditional GAN). Diffusion-based img2img such as *SDEdit* (Meng et al., 2022) adds noise to an input and denoises it toward a prompt. |
| **Diffusion models** | Generative models trained to remove noise step by step (DDPM, Ho et al., 2020; latent diffusion, Rombach et al., 2022). Generation starts from noise and repeatedly denoises it, guided by conditions. |
| **Conditional generation** | The denoising is steered by extra inputs: here a **text prompt** and a **sketch**. |
| **Text-image multimodal conditioning** | Text says *what* (materials, colours, mood); the sketch says *where* (layout, outline). Two modalities jointly define the output. |
| **ControlNet-style structural conditioning** | ControlNet (Zhang, Rao & Agrawala, 2023) adds a trainable branch to a frozen diffusion model so an edge/scribble/sketch map controls spatial layout. Stability's API reference describes Control: Sketch as sketch-guided generation but does not document its internals, so treat ControlNet as the **conceptual reference**, not a claim about what runs behind the API. |
| **CLIP & vision-language models** | CLIP (Radford et al., 2021) embeds images and text in one space; it underlies text conditioning in many diffusion models and the *CLIPScore* metric (Hessel et al., 2021) for prompt-image agreement. Captioning VLMs could be used to check whether a result matches its prompt. This project does **not** run CLIP locally (it would need PyTorch and hundreds of MB of weights). |

### Controls explained

- **Sketch adherence (control strength, 0-1):** how strongly the sketch constrains the result. Higher = closer to your lines, lower = more creative freedom. Documented range 0-1, default 0.7.
- **Negative prompt:** things to avoid (blur, watermark, pencil texture).
- **Seed:** the random starting noise. Same sketch + prompt + settings + seed *should* give a very similar result; 0 means random. Exact repeatability is not guaranteed by the provider.
- **Style:** adds descriptive words to your prompt and, for some styles, selects one of Stability's optional style presets.

### Honest limitations

- The model **follows** your sketch but does **not** reproduce it perfectly; thin details can be dropped or reinterpreted.
- Very abstract scribbles may be misread. Clear closed outlines work best.
- Prompts must be in **English** for this service.
- Output resolution matches the (resized) sketch, up to 1024 px on the long side by default.
- Quality varies from run to run; try a few seeds.
- The offline stylizer is simple colouring, **not** AI.

### Evaluating generated images

- **Human ratings (1-5)** for visual quality and structure preservation are the primary evidence in this project.
- **Automatic edge proxy** (experimental): checks whether strong edges in the output lie near the sketch's lines. Compare it with its *chance level*; it is a rough indicator only.
- **CLIPScore / FID / LPIPS:** established metrics for prompt agreement, distribution realism and perceptual distance. Not computed automatically here; you may type a CLIP score you computed elsewhere.
- **Why not PSNR / SSIM?** They compare an output with an *aligned reference image* pixel by pixel (PSNR via squared error, SSIM via local luminance/contrast/structure). A sketch has many equally valid realistic renderings and no ground-truth reference, and a correct photo-like render looks very different from a line drawing at pixel level. They would score good results as bad. Use them only if you have paired ground-truth images (e.g. a dataset of sketches with their true photos).
"""

EVAL_INTRO_MD = """
Every generation is logged automatically. Three kinds of values are kept **separate**:

- **Measured** (automatic): generation time, provider/model, seed, estimated credits.
- **Human ratings** (yours): quality and structure preservation, 1-5. They stay blank until *you* rate.
- **Experimental proxy** (automatic): edge-based structure score with its chance level. Not validated.

The log is saved to `evaluation/evaluation_log.csv` (text only, no images, no keys) and can be exported below.
"""

CSS = """
:root { --s2r-ink:#22262b; --s2r-paper:#f1f4f6; --s2r-pen:#0f766e; --s2r-warn:#b7791f; --s2r-err:#b42318; }
.gradio-container { max-width: 1280px !important; }
.s2r-hero { display:flex; flex-wrap:wrap; gap:24px; align-items:center; justify-content:space-between;
  padding:28px 32px; border-radius:14px; color:#eef2f5;
  background-color:#22262b;
  background-image:linear-gradient(rgba(255,255,255,.055) 1px, transparent 1px),
                   linear-gradient(90deg, rgba(255,255,255,.055) 1px, transparent 1px);
  background-size:24px 24px; }
.s2r-hero-copy { flex:1 1 320px; max-width:520px; }
.s2r-hero h1 { margin:0 0 6px; font-size:2.4rem; letter-spacing:-0.02em; color:#fff; }
.s2r-hero .s2r-tag { margin:0 0 10px; font-size:1.15rem; color:#d8e6ea; }
.s2r-hero .s2r-sub { margin:0; font-size:.95rem; line-height:1.55; color:#aeb9c2; }
.s2r-art { flex:1 1 300px; max-width:520px; width:100%; height:auto; }
.s2r-status { border-left:5px solid #64748b; padding:10px 14px; border-radius:6px; background:rgba(100,116,139,.12); line-height:1.5; }
.s2r-status strong { display:block; margin-bottom:2px; }
.s2r-status.ok { border-color:var(--s2r-pen); background:rgba(15,118,110,.12); }
.s2r-status.warn { border-color:var(--s2r-warn); background:rgba(183,121,31,.14); }
.s2r-status.error { border-color:var(--s2r-err); background:rgba(180,35,24,.12); }
.s2r-status ul { margin:6px 0 0 18px; padding:0; font-size:.9rem; opacity:.9; }
.s2r-status code { padding:1px 5px; border-radius:4px; background:rgba(127,127,127,.2); }
.s2r-cost { font-size:.9rem; opacity:.85; margin:2px 0 8px; }
@media (max-width: 640px) { .s2r-hero { padding:20px 18px; } .s2r-hero h1 { font-size:1.8rem; } }
"""
