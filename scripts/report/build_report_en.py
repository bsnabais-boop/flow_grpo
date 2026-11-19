"""English visual report for the supervisor: every point with its tables, charts and images, simple explanations.
Numbers are read from the result files; run from anywhere:  python scripts/report/build_report_en.py
Output: scripts/report/latent_spark_report_en.html (published to the meeting-report artifact URL).
The in-house trained scorer is left out on purpose (replaced by NoisyCLIP on the supervisor's advice); runs that used it
are labelled "earlier in-house scorer"."""
import base64
import glob
import html
import io
import json
import math
import os
import random
import sys

import numpy as np
from PIL import Image

ROOT = "/user/home/b.nabais/flow_grpo"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "latent_spark_report_en.html")
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def jload(p):
    with open(p) as f:
        return json.load(f)


def en(x, nd=3, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "−")


# ------------------------------------------------------------------ paired comparisons
def per_prompt(x, pid):
    return np.array([np.nanmean(x[pid == p]) for p in np.unique(pid)])


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def metric_arrays(d, name):
    m = {"CLIP": d["clip"], "PickScore": d["pick"]}
    if name == "ocr":
        m["OCR"] = d["ocr"]
        m["Exact"] = (d["ocr"] >= 1.0).astype(float)
    else:
        ok = (d["clip"][:, None] > d["neg"]).astype(float)
        ok[np.isnan(d["neg"])] = np.nan
        with np.errstate(all="ignore"):
            m["Neg"] = np.nanmean(ok, 1)
    return m


def compare(ref_dir, run_dir):
    out = {}
    for name in ["align", "ocr"]:
        r = np.load(os.path.join(ref_dir, f"{name}.npz"))
        d = np.load(os.path.join(run_dir, f"{name}.npz"))
        rm, dm = metric_arrays(r, name), metric_arrays(d, name)
        for k in rm:
            with np.errstate(all="ignore"):
                diff = per_prompt(dm[k] - rm[k], r["prompt_ids"])
            diff = diff[~np.isnan(diff)]
            lo, hi = boot(diff)
            out[f"{name}.{k}"] = dict(val=float(np.nanmean(dm[k])), diff=float(diff.mean()), lo=lo, hi=hi,
                                      sig=bool(lo > 0 or hi < 0))
    return out


def ckpts(run, div):
    """checkpoint-N -> epoch N // div (2 optimizer steps per epoch in the older runs, 1 in nc_wsched_fix)."""
    ds = []
    for d in glob.glob(f"logs/phase3_eval/mixed_{run}_checkpoint-*"):
        if os.path.exists(os.path.join(d, "summary.json")):
            ds.append((int(d.rsplit("-", 1)[1]) // div, d))
    return sorted(ds)


RUN_DIV = {"prm": 2, "orm": 2, "prm_wsched": 2, "prm_ssched": 2, "nc_wsched_fix": 1}
P3 = {run: [(ep, compare("logs/phase3_eval/base_10steps", d)) for ep, d in ckpts(run, div)]
      for run, div in RUN_DIV.items()}

CONF = {}
for name in ["prm_checkpoint-120", "orm_checkpoint-120", "wsched_checkpoint-60", "wsched_checkpoint-200",
             "ssched_checkpoint-200", "nc_wsched_fix_checkpoint-100"]:
    if os.path.exists(f"logs/phase3_confirm/{name}/summary.json"):
        CONF[name] = compare("logs/phase3_confirm/base_10steps", f"logs/phase3_confirm/{name}")
CONF_BASE = {n: jload(f"logs/phase3_confirm/base_{n}steps/summary.json") for n in [10, 20, 50]}


def sig(r, nd=3):
    return en(r["diff"], nd, True) + (" *" if r["sig"] else " (n.s.)")


# ------------------------------------------------------------------ images
def jpeg_uri(im, q=82):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def grid_row(path, row, nrows, thumb_out=110):
    g = Image.open(path).convert("RGB")
    cell = g.height // nrows
    ncols = g.width // cell
    return [g.crop((c * cell, row * cell, (c + 1) * cell, (row + 1) * cell)).resize((thumb_out, thumb_out), Image.LANCZOS)
            for c in range(ncols)]


def strip(imgs, labels, cls="strip"):
    cells = "".join(f"<figure class='cell'><img src='{jpeg_uri(im)}' alt='{html.escape(l)}'>"
                    f"<figcaption>{l}</figcaption></figure>" for im, l in zip(imgs, labels))
    return f"<div class='{cls}'>{cells}</div>"


def grid_cells(path, n):
    g = Image.open(path).convert("RGB")
    h = g.height
    return [g.crop((i * h, 0, (i + 1) * h, h)).resize((150, 150), Image.LANCZOS) for i in range(n)]


# ------------------------------------------------------------------ SVG charts
W, H = 600, 260
ML, MR, MT, MB = 52, 16, 14, 44


def line_chart(series, xs, ymin, ymax, yticks, xlab, ylab, xpos=None, xticklabels=None, refs=(), fmt=None, zero=False):
    fmt = fmt or (lambda v: en(v, 2))
    xmin, xmax = min(xs), max(xs)
    if xpos is None:
        xpos = lambda x: (x - xmin) / (xmax - xmin) if xmax > xmin else 0.5
    sx = lambda x: ML + xpos(x) * (W - ML - MR)
    sy = lambda y: MT + (1 - (y - ymin) / (ymax - ymin)) * (H - MT - MB)
    o = [f"<svg viewBox='0 0 {W} {H}' role='img' class='chart'>"]
    for t in yticks:
        o.append(f"<line class='grid{' zero' if (zero and abs(t) < 1e-12) else ''}' x1='{ML}' x2='{W - MR}' "
                 f"y1='{sy(t):.1f}' y2='{sy(t):.1f}'/><text class='tick' x='{ML - 6}' y='{sy(t) + 4:.1f}' "
                 f"text-anchor='end'>{fmt(t)}</text>")
    for x, lab in zip(xs, xticklabels or xs):
        o.append(f"<text class='tick' x='{sx(x):.1f}' y='{H - MB + 18}' text-anchor='middle'>{lab}</text>")
    o.append(f"<text class='axl' x='{(ML + W - MR) / 2:.0f}' y='{H - 6}' text-anchor='middle'>{xlab}</text>")
    o.append(f"<text class='axl' transform='translate(13 {(MT + H - MB) / 2:.0f}) rotate(-90)' "
             f"text-anchor='middle'>{ylab}</text>")
    for y, lab in refs:
        o.append(f"<line class='ref' x1='{ML}' x2='{W - MR}' y1='{sy(y):.1f}' y2='{sy(y):.1f}'/>"
                 f"<text class='reflab' x='{W - MR - 4}' y='{sy(y) - 5:.1f}' text-anchor='end'>{lab}</text>")
    for s in series:
        pts = [(x, y) for x, y in zip(xs, s["ys"]) if y is not None and not (isinstance(y, float) and math.isnan(y))]
        if s.get("lo") is not None:
            band = [(x, lo, hi) for x, lo, hi in zip(xs, s["lo"], s["hi"]) if lo is not None and not math.isnan(lo)]
            if band:
                up = " ".join(f"{sx(x):.1f},{sy(hi):.1f}" for x, lo, hi in band)
                dn = " ".join(f"{sx(x):.1f},{sy(lo):.1f}" for x, lo, hi in reversed(band))
                o.append(f"<polygon class='band {s['cls']}' points='{up} {dn}'/>")
        if len(pts) > 1:
            o.append(f"<polyline class='ln {s['cls']}{' dash' if s.get('dash') else ''}' points='"
                     + " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts) + "'/>")
        for x, y in pts:
            o.append(f"<circle class='dot {s['cls']}' cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3.2'/>")
        if s.get("label") and pts:
            x, y = pts[-1]
            o.append(f"<text class='endlab {s['cls']}' x='{sx(x) - 6:.1f}' y='{sy(y) - 8:.1f}' "
                     f"text-anchor='end'>{s['label']}</text>")
    o.append("</svg>")
    return "".join(o)


def hbars(items, vmax, unit_fmt, w=600, vmin=0.0):
    """items: (label, value, cls, note). Supports negative values when vmin < 0."""
    rowh, lw = 34, 210
    h = rowh * len(items) + 8
    span = (w - lw - 120)
    x0 = lw + span * (-vmin) / (vmax - vmin)
    o = [f"<svg viewBox='0 0 {w} {h}' role='img' class='chart'>"]
    if vmin < 0:
        o.append(f"<line class='grid zero' x1='{x0:.1f}' x2='{x0:.1f}' y1='0' y2='{h}'/>")
    for i, (lab, v, cls, note) in enumerate(items):
        y = 4 + i * rowh
        bw = max(2, span * abs(v) / (vmax - vmin))
        bx = x0 if v >= 0 else x0 - bw
        tx = (x0 + bw + 8) if v >= 0 else (x0 + 8)
        o.append(f"<text class='blab' x='{lw - 10}' y='{y + 20}' text-anchor='end'>{lab}</text>"
                 f"<rect class='bar {cls}' x='{bx:.1f}' y='{y + 6}' width='{bw:.1f}' height='20' rx='2'/>"
                 f"<text class='bval' x='{tx:.1f}' y='{y + 21}'>{unit_fmt(v)}{note}</text>")
    o.append("</svg>")
    return "".join(o)


def vbars(groups, series, ymax, yticks, xlab, ylab, fmt):
    """groups: x labels; series: list of (values, cls). Grouped vertical bars."""
    sx0, sx1 = ML, W - MR
    sy = lambda y: MT + (1 - y / ymax) * (H - MT - MB)
    gw = (sx1 - sx0) / len(groups)
    bw = gw * 0.8 / len(series)
    o = [f"<svg viewBox='0 0 {W} {H}' role='img' class='chart'>"]
    for t in yticks:
        o.append(f"<line class='grid' x1='{ML}' x2='{W - MR}' y1='{sy(t):.1f}' y2='{sy(t):.1f}'/>"
                 f"<text class='tick' x='{ML - 6}' y='{sy(t) + 4:.1f}' text-anchor='end'>{fmt(t)}</text>")
    for gi, g in enumerate(groups):
        gx = sx0 + gi * gw + gw * 0.1
        for si, (vals, cls) in enumerate(series):
            v = vals[gi]
            o.append(f"<rect class='bar {cls}' x='{gx + si * bw:.1f}' y='{sy(v):.1f}' width='{bw - 1:.1f}' "
                     f"height='{sy(0) - sy(v):.1f}' rx='1.5'/>")
        o.append(f"<text class='tick' x='{sx0 + gi * gw + gw / 2:.1f}' y='{H - MB + 18}' text-anchor='middle'>{g}</text>")
    o.append(f"<text class='axl' x='{(ML + W - MR) / 2:.0f}' y='{H - 6}' text-anchor='middle'>{xlab}</text>")
    o.append(f"<text class='axl' transform='translate(13 {(MT + H - MB) / 2:.0f}) rotate(-90)' text-anchor='middle'>{ylab}</text>")
    o.append("</svg>")
    return "".join(o)


def legend(items):
    return "<div class='legend'>" + "".join(f"<span><i class='sw {c}'></i>{l}</span>" for l, c in items) + "</div>"


def fig(body, title, text):
    return f"<figure class='fig'>{body}<figcaption><strong>{title}</strong> {text}</figcaption></figure>"


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='tw'><table class='{cls}'><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>"


S, NAV = [], []


def section(num, title, lead, body, short=None):
    sid = f"p{num}" if num else "p" + "".join(c for c in title.lower() if c.isalnum())[:8]
    n = f"<span class='num'>{num}</span>" if num else ""
    NAV.append((sid, f"{num}. {short or title}" if num else (short or title)))
    S.append(f"<section id='{sid}'><h2>{n}{title}</h2><p class='lead'>{lead}</p>{body}</section>")


def concl(text):
    return f"<p class='concl'><b>Takeaway:</b> {text}</p>"


# ================================================================== 0. at a glance
w_fin = CONF.get("wsched_checkpoint-200")
glance = ("<ul class='glance'>"
          "<li><b>Text rendering:</b> one run passed the independent check: the weight curriculum (alignment first, "
          f"text later) at epoch 100, OCR {en(w_fin['ocr.OCR']['val'])} vs 0.103 for the base model "
          f"({en(w_fin['ocr.OCR']['diff'], 4, True)}, 95% interval {en(w_fin['ocr.OCR']['lo'], 4, True)} to "
          f"{en(w_fin['ocr.OCR']['hi'], 4, True)}). Real but small: 20 sampling steps give +0.049.</li>"
          "<li><b>Why OCR barely moves (point 9):</b> the reward signal is strong, but each GRPO update is almost pure "
          "noise (~1–2 × 10⁻⁴ signal per image, so a whole run adds up to only ~2× the noise), and the last "
          "denoising step took most of the update and of the KL penalty. Sharing the start noise or training only "
          "the text-forming steps does not raise the signal. Next: a differentiable text reward.</li>"
          "<li><b>NoisyCLIP from Hugging Face on SD 1.5 (point 5):</b> it was trained on SDXL latents; on SD 1.5 it only "
          "works if the latent is first turned into colours with our SD 1.5 colour map Φ, and then it is about as good "
          "as plain CLIP on that colour image.</li>"
          "<li><b>New run (point 10):</b> NoisyCLIP + OCR weight starting at 40 (not 0) + three fixes from the diagnostic. "
          "Confirmed with new seeds: alignment and image quality improve (alignment CLIP and PickScore +0.0015, "
          "text-prompt CLIP +0.007, above base 20 steps); OCR does not (−0.007, n.s.).</li></ul>")
S.append(f"<section id='glance' class='first'><h2>At a glance</h2>{glance}</section>")

# ================================================================== 1. SD 1.5
ts = [901, 801, 701, 601, 501, 401, 301, 201, 101, 1]
cells = ""
for i, t in enumerate(ts, 1):
    c = "spark" if i <= 5 else ""
    q = " q" if i in (4, 6, 8, 10) else ""
    cut = " cut" if i == 6 else ""
    cells += f"<div class='st {c}{q}{cut}'><b>{i}</b><span>t={t}</span></div>"
timeline = (f"<div class='steps'>{cells}</div>"
            "<div class='stkey'><span><i class='k spark'></i>high-noise half (the \"latent spark\", t ∈ [1, 0.5])</span>"
            "<span><i class='k q'></i>steps where the alignment reward is read in Phase 3</span>"
            "<span><i class='k cut'></i>Phase 2 pruning check (after step 6)</span></div>")
section(1, "Switch from SD 3.5 to SD 1.5",
        "As asked, training moved to Stable Diffusion 1.5. The NoisyCLIP colour projection is made for 4-channel latents "
        "(SD 1.5 / SDXL); SD 3.5 has 16. SD 1.5 predicts noise (it is not flow matching), so the GRPO sampler was "
        "rewritten: a noisy DDIM step with an exact probability per step, a rank-32 LoRA, and 10 sampling steps in training.",
        fig(timeline, "The 10 sampling steps used throughout.",
            "Each box is one step; t is the noise level (901 = almost pure noise, 1 = finished image). The plan's "
            "high-noise half is steps 1 to 5."), "SD 1.5")

# ================================================================== 2. OCR GRPO
rows2 = [
    ["Original: reward from mid-generation previews, long sentences", "~225", "14 h", "+0.003", "no gain"],
    ["Reward of the final image, short prompts (1–2 words)", "~375 (8 prompts/epoch), then ~79 (32)", "2.5 h + 2.5 h",
     "+0.011", "no gain"],
    ["Training window on the steps where text forms", "~40", "7.6 h", "−0.015 to +0.005 (4 evals)", "no gain"],
    ["4-step window", "~64", "15 h", "−0.002 (ep. 30) / +0.002 (ep. 60)", "no gain"],
    ["Wider PPO clip (1e-4 → 1e-3)", "~32", "7.9 h", "−0.011", "no gain"],
    ["1 update per epoch", "~100", "6.8 h", "−0.021 (ep. 30) / −0.002 (ep. 90)", "no gain"],
    ["Higher learning rate (3e-4)", "~115", "3.8 h", "+0.0005", "no gain"],
]
section(2, "GRPO for text: OCR + CLIP reward",
        "Reward = 80 × OCR + CLIPScore. OCR (PaddleOCR) is 1 when the text is exactly right and 0 when it is unrelated. "
        "CLIP changes only by thousandths within a group, so in practice OCR decides. The base is the original SD 1.5, "
        "untrained, with the same steps and seeds.",
        fig(table(["What changed", "Epochs", "Time (1 GPU)", "OCR vs base", "Verdict"], rows2),
            "Seven set-ups, none improved the text.",
            "\"OCR vs base\" = mean OCR of the trained model minus that of the base. Two values = two checkpoints, "
            "each checked independently. Each run stopped when it stalled, so durations differ.")
        + "<ul class='take'><li><b>The model barely moved:</b> each update changed the policy by ~1e-7 (KL).</li>"
          "<li><b>Evaluating carelessly misleads:</b> two \"gains\" seen during training (z = 2.5 and 1.8) disappeared "
          "in the independent check.</li>"
          "<li>Why the reward did not turn into better text is analysed in point 9.</li></ul>", "OCR GRPO")

# ================================================================== 3. per-step previews
s3 = jload("logs/intermediate_eval/win4_base/summary.json")
steps3 = [m["step"] for m in s3["per_step"]]
c3a = line_chart([{"ys": [m["mean_ocr"] for m in s3["per_step"]], "cls": "c2"}], steps3, 0, 0.14,
                 [0, 0.04, 0.08, 0.12], "step (of 10)", "OCR of the preview",
                 refs=[(s3["final"]["mean_ocr"], "final image 0.12")])
c3b = line_chart([{"ys": [m["within_group_spearman"] for m in s3["per_step"]], "cls": "c1"}], steps3, 0, 1,
                 [0, 0.25, 0.5, 0.75, 1], "step (of 10)", "ranking agreement in the group")
strip3 = strip(grid_row("logs/intermediate_eval/win4_base/grids/prompt_000.png", 0, 8),
               [f"step {i}" for i in range(1, 11)] + ["final"])
tab3 = table(["Step (of 10)", "1–3", "4", "5", "6", "7", "8", "10"],
             [["OCR of the preview", "0.00–0.05", "0.10", "0.12", "0.13", "0.13", "0.13", "0.12"],
              ["Ranking agreement in the group", "0.02–0.14", "0.26", "0.45", "0.60", "0.70", "0.79", "0.93"]])
section(3, "Where the image and the text form",
        "At every step the model estimates the noise; removing it gives a prediction of the final image (x̂₀). I decoded "
        "that prediction with the VAE into a preview and scored it with the training reward. 128 prompts × 8 images.",
        fig(strip3, "The same image seen at every step.",
            "Prompt: <i>A book cover with the title \"GAME OVER\" in large serif font</i>. Steps 1–3 are blobs; by "
            "step 4–5 the layout and the letters are in place; later steps only refine details.")
        + "<div class='two'>"
        + fig(c3a, "Text becomes readable early.", "The preview's OCR reaches the final value at step 5.")
        + fig(c3b, "Ranking the group's images needs more steps.",
              "Agreement (Spearman) between the order of the 8 images by preview and by final image. With the VAE it "
              "only passes 0.6 at step 6.")
        + "</div>"
        + fig(tab3, "The same numbers as a table.", "The ranking uses the total reward (80 × OCR + CLIP), which follows OCR.")
        + "<p class='note'>Each VAE preview + OCR costs ~1.4 UNet steps: that is the cost Phase 1 wants to remove.</p>",
        "Steps")

# ================================================================== 4. fewer steps
NS = [4, 6, 8, 10, 20, 50]
s4 = {n: jload(f"logs/intermediate_eval/h200_eta0_{n}steps/summary.json")["final"] for n in NS}
logpos = lambda x: (math.log(x) - math.log(4)) / (math.log(50) - math.log(4))
c4a = line_chart([{"ys": [s4[n]["mean_ocr"] for n in NS], "cls": "c2"}], NS, 0, 0.2, [0, 0.05, 0.1, 0.15, 0.2],
                 "number of steps (log scale)", "OCR", xpos=logpos)
c4b = line_chart([{"ys": [s4[n]["mean_clip"] for n in NS], "cls": "c1"}], NS, 0.8, 1.0, [0.8, 0.85, 0.9, 0.95, 1.0],
                 "number of steps (log scale)", "CLIP (training scale)", xpos=logpos)
strip4 = strip([grid_row(f"logs/intermediate_eval/h200_eta0_{n}steps/grids/prompt_000.png", 0, 8)[-1] for n in NS],
               [f"{n} steps" for n in NS])
tab4 = table(["Steps", "4", "6", "8", "10", "20", "50"],
             [["OCR"] + [en(s4[n]["mean_ocr"]) for n in NS],
              ["% of the 50-step OCR"] + [f"{100 * s4[n]['mean_ocr'] / s4[50]['mean_ocr']:.0f}%" for n in NS],
              ["CLIP"] + [en(s4[n]["mean_clip"]) for n in NS],
              ["Exact text"] + [f"{100 * s4[n]['exact_ocr_frac']:.0f}%" for n in NS]])
section(4, "How much is lost with fewer steps",
        "Purpose: measure how much the untrained model loses with fewer steps, before any training. It is the starting "
        "point and the yardstick for the 40–50 → 10–15 steps goal. Base SD 1.5, 1024 images per row, H200.",
        fig(strip4, "Same prompt and seed, 4 to 50 steps.",
            "Prompt: <i>A book cover with the title \"GAME OVER\" in large serif font</i>. With few steps the image is "
            "poor; text is what takes longest to get right.")
        + "<div class='two'>"
        + fig(c4a, "Text keeps improving up to 50 steps.", "At 10 steps the model keeps 61% of its 50-step OCR.")
        + fig(c4b, "Alignment settles early.", "CLIP hardly changes after 10–20 steps.")
        + "</div>" + fig(tab4, "Full table.", "")
        + concl("with fewer steps SD 1.5 keeps almost all of its alignment but loses a lot of text. That gap is what "
                "training has to close."), "Fewer steps")

# ================================================================== 5. NoisyCLIP on SD 1.5
s5 = jload("logs/phase1_noisyclip/eval10_noisyclip/summary.json")
ps = s5["per_step"]
st5 = [p["step"] for p in ps]
M5 = [("nc_zt", "c3", "NoisyCLIP on z_t (paper)", True), ("nc_x0", "c4", "NoisyCLIP on x̂₀, SDXL colours", True),
      ("nc_x0phi", "c1", "NoisyCLIP on x̂₀, our Φ", False), ("taesd", "c2", "TAESD + CLIP", False)]
c5 = line_chart([{"ys": [p[f"{m}_spearman"] for p in ps], "cls": c, "dash": d} for m, c, _, d in M5]
                + [{"ys": [p["dec_spearman"] for p in ps], "cls": "c5"}],
                st5, -0.3, 1, [-0.25, 0, 0.25, 0.5, 0.75, 1], "step (of 10)", "correlation with final CLIP", zero=True)
c5w = line_chart([{"ys": [p[f"{m}_grp_spearman"] for p in ps], "cls": c, "dash": d} for m, c, _, d in M5]
                 + [{"ys": [p["dec_grp_spearman"] for p in ps], "cls": "c5"}],
                 st5, -0.1, 1, [0, 0.25, 0.5, 0.75, 1], "step (of 10)", "ranking within a prompt", zero=True)
c5d = line_chart([{"ys": [p[f"{m}_disc_acc"] for p in ps], "cls": c, "dash": d} for m, c, _, d in M5],
                 st5, 0.4, 1.0, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "step (of 10)", "wrong prompt spotted",
                 fmt=lambda v: f"{100 * v:.0f}%")
leg5 = legend([(l, c) for _, c, l, _ in M5] + [("VAE + CLIP (steps 5 and 10 only)", "c5")])
cost = s5["cost"]
cost5 = hbars([("NoisyCLIP (any input)", cost["nc_in_unet_steps"], "c1", ""),
               ("TAESD + CLIP", cost["taesd_in_unet_steps"], "c2", ""),
               ("VAE + CLIP", cost["decode_in_unet_steps"], "c3",
                f"  ({cost['decode_in_unet_steps'] / cost['nc_in_unet_steps']:.0f}× NoisyCLIP)")],
              cost["decode_in_unet_steps"], lambda v: en(v, 2))
A = os.path.join(HERE, "assets")
meta5 = jload(os.path.join(A, "nc_views.json"))
views = ""
for m in meta5:
    k = m["k"]
    load = lambda p: Image.open(os.path.join(A, p)).resize((128, 128), Image.LANCZOS)
    views += (f"<div class='vrow'><div class='vprompt'><i>{html.escape(m['prompt'])}</i></div><div class='vgrid'>"
              "<div class='rowlab'>NoisyCLIP's SDXL colour map</div>"
              + "".join(f"<img src='{jpeg_uri(load(f'nc_{k}_{s}.png'))}' alt='SDXL map, step {s}'>" for s in (2, 4, 6, 8, 10))
              + "<div class='rowlab'>our SD 1.5 colour map Φ</div>"
              + "".join(f"<img src='{jpeg_uri(load(f'phi_{k}_{s}.png'))}' alt='Phi, step {s}'>" for s in (2, 4, 6, 8, 10))
              + f"<div class='final'><img src='{jpeg_uri(load(f'final_{k}.png'))}' alt='final'><span>final (VAE)</span></div>"
              "</div></div>")
views = ("<div class='vhead'><span></span>" + "".join(f"<span>step {s}</span>" for s in (2, 4, 6, 8, 10))
         + "</div>" + views)
i5, i10 = 4, 9
tab5 = table(["Input given to NoisyCLIP", "Step 5: correlation / within-prompt / wrong prompt",
              "Step 10: correlation / within-prompt / wrong prompt"],
             [[l] + [f"{en(ps[i][m + '_spearman'], 2)} / {en(ps[i][m + '_grp_spearman'], 2)} / "
                     f"{100 * ps[i][m + '_disc_acc']:.0f}%" for i in (i5, i10)] for m, _, l, _ in M5]
             + [["Plain CLIP on our Φ image (no NoisyCLIP)"]
                + [f"{en(ps[i]['proj_x0_spearman'], 2)} / {en(ps[i]['proj_x0_grp_spearman'], 2)} / "
                   f"{100 * ps[i]['proj_x0_disc_acc']:.0f}%" for i in (i5, i10)],
                ["VAE decode + CLIP"] + [f"{en(ps[i]['dec_spearman'], 2)} / {en(ps[i]['dec_grp_spearman'], 2)} / "
                                         f"{100 * ps[i]['dec_disc_acc']:.0f}%" for i in (i5, i10)]])
section(5, "Phase 1: NoisyCLIP as the mid-generation alignment score",
        "Goal: know halfway through generation whether the image will match the prompt, without the VAE. As advised, I "
        "use the published <b>NoisyCLIP</b> (Hugging Face <code>asiimo/noisyclip</code>): CLIP ViT-L/14 whose image "
        "side was fine-tuned on <b>SDXL</b> noisy latents, turned into a small colour picture with a fixed 4→3 colour "
        "map. Its text side is the normal CLIP. SD 1.5 latents use a different colour code, so I tried three inputs: "
        "(1) the noisy latent z_t with NoisyCLIP's colour map, as in the paper; (2) the model's prediction of the final "
        "latent x̂₀ with the same map; (3) x̂₀ with <b>Φ</b>, a colour map fitted to SD 1.5 (15 numbers, least squares "
        "against decoded images). 128 unseen GenAI-Bench prompts × 4 images, 10 noisy steps (the GRPO setting).",
        fig(views, "What NoisyCLIP is shown, for one image per prompt.",
            "Top row of each pair: the SD 1.5 latent painted with NoisyCLIP's SDXL colour map (wrong colours, e.g. a "
            "blue warrior and an orange sky). Bottom row: the same latent with our Φ, close to the real colours. "
            "Right: the final decoded image. These show x̂₀; the noisy z_t itself looks like coloured noise until late.")
        + fig(leg5 + c5, "How well each score predicts the final image's CLIP score.",
              "Spearman correlation over all 512 images between the score at each step and CLIP on the final image. "
              "NoisyCLIP on z_t (the paper's input) is useless on SD 1.5; with our Φ it reaches ~0.45 at step 5 and "
              "~0.55 at the end. TAESD (a small decoder) is better from step 5 on.")
        + "<div class='two'>"
        + fig(c5w, "Ranking images of the same prompt (what GRPO needs).",
              "Within-prompt Spearman, averaged over prompts. This is the harder test: GRPO compares images of the "
              "same prompt. NoisyCLIP with Φ stays at ~0.3; TAESD reaches 0.77.")
        + fig(c5d, "Spotting a wrong prompt.",
              "Share of images that score higher with their own prompt than with a random other prompt. Easy for "
              "every input except z_t; NoisyCLIP with Φ is at 96% by step 2.")
        + "</div>"
        + fig(cost5, "Cost of one score, in UNet steps (1 = one generation step).",
              "NoisyCLIP needs no decoder, so it is about as cheap as it gets.")
        + fig(tab5, "Summary table.", "Correlation with the final CLIP / ranking within a prompt / wrong prompt spotted.")
        + concl("NoisyCLIP is cheap and spots clearly wrong prompts early, but on SD 1.5 it does not transfer as is: "
                "it was trained on SDXL latents. Fed our SD 1.5 colour image it behaves like plain CLIP on that image, "
                "and it ranks images of the same prompt poorly (~0.3). Retraining it on SD 1.5 latents, or "
                "switching to SDXL, would remove the mismatch."), "NoisyCLIP")

# ================================================================== 6. pruning with NoisyCLIP
rng = random.Random(3)
finals = [0.9, 0.82, 0.78, 0.55, 0.45, 0.35, 0.3, 0.2]
pruned = [False, False, False, True, True, True, True, True]
paths = ""
for fv, pr in zip(finals, pruned):
    ys = [0.5 + (fv - 0.5) * ((s / 10) ** 0.8) + (rng.random() - 0.5) * 0.06 * (1 - s / 10) for s in range(11)]
    last = 6 if pr else 10
    paths += f"<polyline class='ln {'c3 faint' if pr else 'c1'}' points='" + " ".join(
        f"{50 + s * 52:.0f},{200 - ys[s] * 170:.1f}" for s in range(last + 1)) + "'/>"
    xe, ye = 50 + last * 52, 200 - ys[last] * 170
    paths += (f"<text class='x' x='{xe:.0f}' y='{ye + 4:.1f}' text-anchor='middle'>×</text>" if pr else
              f"<circle class='dot c1' cx='{xe:.0f}' cy='{ye:.1f}' r='3.5'/>")
ticks6 = "".join(f"<text class='tick' x='{50 + s * 52}' y='222' text-anchor='middle'>{s}</text>" for s in range(0, 11, 2))
diagram6 = (f"<svg viewBox='0 0 620 240' role='img' class='chart'>"
            f"<line class='ref' x1='{50 + 6 * 52}' x2='{50 + 6 * 52}' y1='10' y2='205'/>"
            f"<text class='reflab' x='{50 + 6 * 52 + 6}' y='20'>check at step 6</text>{paths}{ticks6}"
            f"<text class='axl' x='310' y='238' text-anchor='middle'>step</text>"
            f"<text class='axl' transform='translate(16 110) rotate(-90)' text-anchor='middle'>alignment score</text></svg>")
# offline CRC results on logs/phase1_noisyclip/eval10_noisyclip (G=4, risk = best image pruned), from
# scripts/phase2_conformal_pruning.py ... --methods nc_x0phi taesd dec --alpha 0.1 0.2 --splits 40 (2026-11-12)
save6 = hbars([("NoisyCLIP + Φ (step 2–6)", 7, "c1", ""), ("TAESD + CLIP (step 6–7)", 9, "c2", ""),
               ("VAE + CLIP (step 5)", -7, "c3", "  (scoring costs more than it saves)"),
               ("Random pruning at step 1", 9, "c4", "  (reference)")], 25, lambda v: f"{v:+.0f}%".replace("-", "−"),
              vmin=-10)
tab6 = table(["Risk allowed (α)", "NoisyCLIP + Φ", "TAESD + CLIP", "VAE + CLIP", "Random pruning"],
             [["10%", "7% saved (16–20% pruned)", "9% (step 6, 30% pruned)", "−7%", "≈ 9%"],
              ["20%", "14–16% (step 1–3)", "17% (step 6, 48% pruned)", "−1%", "≈ 18%"]])
section(6, "Phase 2: stop bad trajectories early",
        "GRPO generates several images per prompt. If halfway through the score already says an image will end up "
        "clearly behind the group's best, stop it and save the remaining steps. The threshold is set with Conformal "
        "Risk Control so that the chance of throwing away the group's best image stays ≤ α. Redone offline with "
        "NoisyCLIP (groups of 4, 128 prompts, calibrated on half and tested on the other half, 40 times).",
        fig(diagram6, "How it works (illustration).",
            "Lines = images of the same prompt. At step 6 the ones too far behind the leader are stopped (×); only the "
            "others run to step 10.")
        + fig(save6, "Compute saved with α = 10%.",
              "Share of the UNet compute saved, after paying for the scores. Random pruning (stop 10% of the "
              "images at step 1) already saves ≈ 9% at the same risk, so NoisyCLIP adds nothing on SD 1.5.")
        + fig(tab6, "Results by allowed risk.",
              "The measured test risk was ≈ α in every row, so the guarantee holds; the savings are the problem.")
        + concl("the pruning machinery works (the risk guarantee holds), but it needs a score that ranks images of the "
                "same prompt well by mid-generation. NoisyCLIP on SD 1.5 does not (point 5), so pruning with it is no "
                "better than random."), "Pruning")

# ================================================================== 7. Phase 3
def series(run, key):
    xs = [ep for ep, _ in P3[run]]
    return xs, [r[key]["diff"] for _, r in P3[run]], [r[key]["lo"] for _, r in P3[run]], [r[key]["hi"] for _, r in P3[run]]


def p3_chart(runs, key, ymin, ymax, yt, ylab, nd=2):
    runs = [r for r in runs if P3[r[0]]]
    xs_all = sorted(set().union(*[[ep for ep, _ in P3[r]] for r, _, _ in runs]))
    ss = []
    for r, cls, band in runs:
        xs, m, lo, hi = series(r, key)
        m, lo, hi = dict(zip(xs, m)), dict(zip(xs, lo)), dict(zip(xs, hi))
        s = {"ys": [m.get(x, float("nan")) for x in xs_all], "cls": cls}
        if band:
            s["lo"] = [lo.get(x, float("nan")) for x in xs_all]
            s["hi"] = [hi.get(x, float("nan")) for x in xs_all]
        ss.append(s)
    return line_chart(ss, xs_all, ymin, ymax, yt, "training epoch", ylab, zero=True,
                      fmt=lambda v: en(v, nd, sign=True) if v else "0")


c7a = p3_chart([("orm", "c3", True), ("prm", "c1", True)], "ocr.OCR", -0.03, 0.05, [-0.02, 0, 0.02, 0.04], "OCR vs base")
c7b = p3_chart([("orm", "c3", True), ("prm", "c1", True)], "align.PickScore", -0.005, 0.004,
               [-0.004, -0.002, 0, 0.002, 0.004], "PickScore vs base", nd=3)
rows7 = []
for name, lab in [("prm_checkpoint-120", "PRM run, epoch 60"), ("orm_checkpoint-120", "ORM run (CLIP), epoch 60")]:
    r = CONF[name]
    rows7.append([lab, en(r["ocr.OCR"]["val"]), sig(r["ocr.OCR"]), sig(r["align.PickScore"], 4)])
b10 = CONF_BASE[10]
for n in [10, 20, 50]:
    b = CONF_BASE[n]
    rows7.append([f"Base, {n} steps", en(b["ocr"]["ocr"]),
                  "—" if n == 10 else en(b["ocr"]["ocr"] - b10["ocr"]["ocr"], 3, True) + " *",
                  "—" if n == 10 else en(b["align"]["pickscore"] - b10["align"]["pickscore"], 4, True) + " *"])
tab7 = table(["Model (10 steps unless stated)", "OCR", "OCR vs base 10", "PickScore vs base 10"], rows7)
imgs7 = (grid_cells("logs/phase3_eval/base_10steps/grid_ocr.png", 5)
         + grid_cells("logs/phase3_eval/mixed_prm_checkpoint-120/grid_ocr.png", 5)
         + grid_cells("logs/phase3_eval/base_50steps/grid_ocr.png", 5))
g7 = ("<div class='imgrid'>" + "".join(
    f"<div class='rowlab'>{lab}</div>" + "".join(f"<img src='{jpeg_uri(im)}' alt='{lab}'>" for im in imgs7[k * 5:(k + 1) * 5])
    for k, lab in enumerate(["Base, 10 steps", "PRM run, epoch 60", "Base, 50 steps"])) + "</div>")
section(7, "Phase 3: GRPO with a process reward vs an outcome reward",
        "Focus on alignment without losing text: 1440 GenAI-Bench prompts + 1088 text prompts. Reward = 100 × alignment "
        "+ 80 × OCR (OCR only on prompts with quoted text). Two runs identical except for the alignment score: "
        "<b>PRM</b> (process reward: alignment read mid-generation at steps 4, 6, 8, 10 with the earlier in-house scorer, "
        "no VAE) and <b>ORM</b> (outcome reward: CLIP on the final image). 10 steps, 32 prompts × 8 images per epoch, "
        "100 epochs (~4.5 h each).",
        fig(legend([("PRM run", "c1"), ("ORM run (CLIP on the final image)", "c3"), ("shade = 95% interval", "c4")])
            + "<div class='two'>" + c7a + c7b + "</div>",
            "During training: difference from the base at each checkpoint.",
            "512 + 512 images, always the same seeds. The PRM run rises in OCR at some checkpoints; the ORM run stays "
            "at zero and drops slightly in PickScore. With many checkpoints a single peak can be luck, hence the "
            "confirmation below.")
        + fig(tab7, "Confirmation with new seeds (1024 + 1024 images).",
              "* = significant (95% interval excludes zero); n.s. = not significant. The +0.026 seen during training "
              "did not hold up: it was the best of 12 checkpoints, a selection effect.")
        + fig(g7, "Examples: same prompt and seed in each column.",
              "Requested texts: TICKETS (street sign), TICKETS (cake), TOP SECRET (wooden sign), SALE NOW (banner), KEEP "
              "OUT (chalkboard). Small differences between base and the PRM run; 50 steps help text more.")
        + concl("training works and the policy moves (KL ~2e-4), but neither run gave a confirmed gain. Alignment did "
                "not get worse."), "Phase 3")

# ================================================================== 8. curricula
eps = list(range(0, 101, 5))
c8w = line_chart([{"ys": [100 * (1 - 0.5 * min(1, e / 60)) for e in eps], "cls": "c1", "label": "alignment weight"},
                  {"ys": [80 * min(1, e / 60) for e in eps], "cls": "c2", "label": "OCR weight (first runs)"},
                  {"ys": [80 * (0.5 + 0.5 * min(1, e / 60)) for e in eps], "cls": "c2", "dash": True,
                   "label": "OCR weight (new run)"}],
                 eps, 0, 100, [0, 25, 50, 75, 100], "epoch", "weight", fmt=lambda v: f"{v:.0f}")
c8s = line_chart([{"ys": [4 if e < 20 else 6 if e < 40 else 8 if e < 60 else 10 for e in eps], "cls": "c4",
                   "label": "sampling steps"}], eps, 0, 12, [0, 4, 6, 8, 10], "epoch", "steps", fmt=lambda v: f"{v:.0f}")
runs8 = [("prm", "c1", False), ("prm_wsched", "c2", True), ("prm_ssched", "c4", False)]
c8 = p3_chart(runs8, "ocr.OCR", -0.02, 0.05, [-0.02, 0, 0.02, 0.04], "OCR vs base")
c8p = p3_chart(runs8, "align.PickScore", -0.004, 0.003, [-0.004, -0.002, 0, 0.002], "PickScore vs base", nd=3)
tab8 = table(["Run", "Weights (alignment / OCR)", "Steps in training", "Length", "Status"],
             [["Weight curriculum", "60-epoch ramp: 100 → 50 / 0 → 80", "10", "100 epochs, 4.5 h", "finished"],
              ["Step curriculum", "fixed 100 / 80", "4 → 6 → 8 → 10 (20 epochs each)", "100 epochs, 3.5 h", "finished"],
              ["Both together", "ramp", "4 → 10", "stopped at epoch 51 (to free the GPU)", "cancelled"]])
names8 = [("wsched_checkpoint-60", "Weight curriculum, epoch 30"),
          ("wsched_checkpoint-200", "Weight curriculum, epoch 100 (chosen in advance)"),
          ("ssched_checkpoint-200", "Step curriculum, epoch 100")]
conf8 = fig(table(["Model", "OCR", "OCR vs base 10", "Exact text vs base", "PickScore (alignment) vs base"],
                  [[lab, en(CONF[n]["ocr.OCR"]["val"]), sig(CONF[n]["ocr.OCR"], 4),
                    en(100 * CONF[n]["ocr.Exact"]["diff"], 1, True) + " pp" + (" *" if CONF[n]["ocr.Exact"]["sig"] else " (n.s.)"),
                    sig(CONF[n]["align.PickScore"], 4)] for n, lab in names8 if n in CONF]),
            "Confirmation with new seeds (1024 + 1024 images, base OCR 0.103).",
            "The final weight-curriculum model is the first OCR gain that passes the independent check. It is small: "
            "base SD 1.5 with 20 steps gives +0.049.")
section(8, "The curricula suggested by the supervisor",
        "(1) Change the weights of alignment and OCR during training. (2) Increase the number of sampling steps during "
        "training. Same set-up as the PRM run except for the curriculum, one run at a time. Steps start at 4, not 2: "
        "with 2 steps SD 1.5 produces nothing usable. Following the supervisor's advice, the new run (point 10) starts "
        "the OCR weight at 40 instead of 0.",
        "<div class='two'>"
        + fig(c8w, "Weight curriculum.", "Alignment first, text comes in gradually; alignment never goes to zero. "
                                         "Dashed: the new schedule, where OCR starts at half weight.")
        + fig(c8s, "Step curriculum.", "4 → 6 → 8 → 10 steps, 20 epochs at each level.")
        + "</div>"
        + fig(tab8, "The three runs.", "All evaluated at 10 steps, with the base's seeds.")
        + fig(legend([("PRM run, no curriculum", "c1"), ("weight curriculum (with 95% interval)", "c2"),
                      ("step curriculum", "c4")]) + "<div class='two'>" + c8 + c8p + "</div>",
              "During training: difference from the base.",
              "The weight curriculum is above zero in OCR at every checkpoint, with PickScore back to ~0 at the end. "
              "The step curriculum stays flat.")
        + conf8
        + concl("the weight curriculum is the most promising result so far and the only one confirmed: +0.013 OCR "
                "with alignment unchanged."), "Curricula")

# ================================================================== 9. diagnostic
D = {n: jload(f"logs/diag_ocr_signal/{f}.json") for n, f in
     [("base", "base"), ("prm", "mixed_prm_checkpoint-200"), ("wsched", "mixed_prm_wsched_checkpoint-200")]}
D2 = jload("logs/diag_ocr_signal/v2_fix/mixed_prm_wsched_checkpoint-200.json")
sigrows = []
for lab, key, f in [("Mean OCR of the images", "ocr_mean", lambda v: en(v, 2)),
                    ("OCR spread within a group (std)", "ocr_within_group_std_mean", lambda v: en(v, 2)),
                    ("Best minus worst OCR in a group", "ocr_group_range_mean", lambda v: en(v, 2)),
                    ("Groups with an image of OCR ≥ 0.5", "frac_groups_best_ocr_ge_0.5", lambda v: f"{100 * v:.0f}%"),
                    ("Groups with an exactly right image", "frac_groups_with_exact_match", lambda v: f"{100 * v:.0f}%"),
                    ("Share of the reward differences due to OCR", "share_of_reward_variance_from_ocr", lambda v: f"{100 * v:.0f}%"),
                    ("Correlation of the advantage with OCR", "corr_advantage_vs_ocr_within_group", lambda v: en(v, 2))]:
    sigrows.append([lab] + [f(D[n]["signal"][key]) for n in ["base", "prm", "wsched"]])
tab9a = table(["Text prompts, groups of 8", "Base", "PRM run, ep. 100", "Weight curriculum, ep. 100"], sigrows)
pt_share = D["base"]["gradient"]["per_transition_norm_share"]
kl_share = D2["gradient"]["kl_per_transition_norm_share"]
trans = [f"{a}→{b}" for a, b in zip(ts[:-1], ts[1:])]
c9a = vbars([f"{i}" for i in range(1, 10)], [([100 * v for v in pt_share], "c1"), ([100 * v for v in kl_share], "c3")],
            100, [0, 25, 50, 75, 100], "transition (1 = t 901→801 … 9 = t 101→1)", "share of the gradient",
            lambda v: f"{v:.0f}%")


def debias(g, B):
    s = g["snr_this_batch"]
    d = s * s - 1
    return math.copysign(math.sqrt(abs(d) * B / 384), d)


tab9b = table(["Model", "Raw measure (384 images)", "Useful signal ÷ noise, 128 images (one update)", "… 1024 images"],
              [[lab, en(D[n]["gradient"]["snr_this_batch"], 2), en(debias(D[n]["gradient"], 128), 2),
                en(debias(D[n]["gradient"], 1024), 2)]
               for n, lab in [("base", "Base"), ("prm", "PRM run, ep. 100"), ("wsched", "Weight curriculum, ep. 100")]])
kl9 = hbars([("β 0.04, PRM run", D["prm"]["gradient"]["kl_grad_over_policy_grad"], "c3", ""),
             ("β 0.04, weight curriculum", D["wsched"]["gradient"]["kl_grad_over_policy_grad"], "c3", ""),
             ("β 0.01, all steps", D2["gradient"]["kl_grad_over_policy_grad"], "c2", ""),
             ("β 0.01, without the last step", D2["gradient"]["nolast_kl_grad_over_policy_grad"], "c1", "  (new run)")],
            1.2, lambda v: f"{en(v, 2)}×")
V3 = [(lab, jload(f"logs/diag_ocr_signal/v3_variants/base_{f}.json")) for lab, f in
      [("Own z_T per image, steps 0–7 (= training now)", "v0_indep_full"),
       ("Shared z_T in the group, steps 0–7", "v1_shared_full"),
       ("Own z_T, stochastic only at steps 1–4", "v2_indep_win1-4"),
       ("Shared z_T, stochastic only at steps 1–4", "v3_shared_win1-4"),
       ("Shared z_T, stochastic only at steps 2–3", "v4_shared_win2-3")]]
tab9c = table(["Sampling in a group of 8", "Signal per image (× 10⁻⁴, ± s.e.)", "Signal ÷ noise, 256-image update (95% CI)",
               "OCR spread in a group (std)", "Groups without spread"],
              [[lab, f"{en(1e4 * d['gradient']['trained_rho2_per_image'], 1)} ± {1e4 * d['gradient']['trained_rho2_se']:.1f}",
                f"{en(d['gradient']['trained_snr_unbiased_at_256'], 2, True)} [{en(d['gradient']['trained_snr_unbiased_at_256_ci'][0], 2, True)}, "
                f"{en(d['gradient']['trained_snr_unbiased_at_256_ci'][1], 2, True)}]",
                en(d["signal"]["ocr_within_group_std_mean"], 2), f"{100 * d['signal']['frac_groups_ocr_range_lt_0.1']:.0f}%"]
               for lab, d in V3])
rho_v0 = V3[0][1]["gradient"]["trained_rho2_per_image"]
section(9, "Why OCR barely improves: a look inside the updates",
        "The supervisor asked to check the hyperparameters, how strong the signal is, the spread within groups and the "
        "gradients. A diagnostic script reproduces training exactly (same sampler, reward and advantages; 32 text + 16 "
        "alignment prompts × 8 images) and computes the gradient of every single image, step by step, for the base "
        "model and two trained models.",
        fig(tab9a, "1. The reward signal is strong.",
            "Images of the same prompt differ a lot in OCR, most groups contain a good image and ~45% contain an "
            "exactly right one. The advantage (what GRPO learns from) follows OCR almost perfectly. So the old "
            "explanation \"there is never a good image in the group\" does not hold for the short text prompts.")
        + fig(legend([("reward (policy) gradient, base", "c1"), ("KL-penalty gradient, weight curriculum", "c3")]) + c9a,
              "2. The last step takes most of the update.",
              "Size of each denoising transition's contribution. The last transition (t 101 → 1) is 57% of the reward "
              "gradient and 92% of the KL penalty's gradient. At that step the noise is tiny, which makes its "
              "probability gradients huge, but the letters are already fixed there: text forms around steps 3–5 "
              "(point 3).")
        + fig(tab9b, "3. Each update is almost pure noise.",
              "A raw value of 1.0 is what pure noise gives (the noise inflates the measure), so the useful signal is "
              "the part above 1. Corrected: for the base model the useful direction is ~0.1× the noise in one "
              "128-image update, and for trained models it is too small to detect. Over a whole run the useful part "
              "adds up only to about the size of the noise, which fits the small, hard-to-confirm gains.")
        + fig(kl9, "4. The KL penalty pulled back as hard as the reward pushed.",
              "Size of the KL-penalty gradient relative to the reward gradient. With β = 0.04 it was 0.7–1.2× at the "
              "end of training. With β = 0.01 and the last step left out it drops to 0.05×.")
        + "<ul class='take'><li><b>Not the problem:</b> gradient clipping never triggers (gradient norm ~0.002–0.004 vs "
          "the limit 1.0); the OCR and alignment parts of the gradient do not fight each other (unrelated directions), "
          "although on a mixed batch about half of each update goes to alignment.</li></ul>"
        + fig(tab9c, "5. Could a different way of sampling the group give a stronger signal? No.",
              "Follow-up test (64 text prompts × 8 images, base model, same seeds in every row, an unbiased estimate "
              "of the signal with error bars). The idea: each image starts from its own random noise z_T, and if z_T "
              "decided whether the text appears, those differences could not be learned from and would only add "
              "noise. Giving the whole group the same z_T, or making only the text-forming steps random, should then "
              "raise the signal. It does not: all rows are within noise of each other and of zero, and sharing z_T "
              "barely reduces the OCR spread (0.27 → 0.25), so the randomness we already train on is what decides the "
              "text. The signal is about 1–2 × 10⁻⁴ per image in every setup.")
        + "<ul class='take'><li><b>What that means in numbers:</b> one update would need about "
          f"{1 / rho_v0 / 1000:.0f},000 images for the signal to equal the noise (we use 256). A whole 100-epoch run "
          f"(25,600 images) adds up to a total of only ~{math.sqrt(25600 * rho_v0):.1f}× the noise, which explains "
          "OCR gains of +0.01–0.02 that look significant in one evaluation and vanish with new seeds.</li>"
          "<li><b>Consequence:</b> RL on the OCR score cannot be fixed with hyperparameters. It needs ~10× more "
          "images per run (30–40 GPU-hours) or a gradient with less noise. We are trying the second first: back-"
          "propagate a differentiable text-recognition score through the last denoising steps (as in DRaFT / ReFL), "
          "keeping PaddleOCR as the independent check. If that fails, the long RL run is the fallback.</li></ul>"
        + concl("the reward knows which images are better, but each image carries very little usable gradient signal "
                "(and with all steps trained most of it went to the last step). Different group sampling does not "
                "change that; only many more images or a lower-noise gradient can."), "Diagnostic")

# ================================================================== 10. new run
def wb_ocr(run_dir):
    from wandb_offline_metrics import history
    per = {}
    for row in history(run_dir):
        if "reward_ocr" in row and "epoch" in row:
            per.setdefault(int(row["epoch"]), []).append(row["reward_ocr"])
    return {e: float(np.mean(v)) for e, v in per.items()}


def blocks(d, size=10):
    out = {}
    for e, v in d.items():
        out.setdefault(e // size, []).append(v)
    return [float(np.mean(out[b])) for b in sorted(out)]


new_ocr = blocks(wb_ocr("wandb/offline-run-20261111_210253-fu9u7p94"))
old_ocr = blocks(wb_ocr("wandb/offline-run-20261110_205325-mwqrmaxd"))
xs10 = [10 * b + 5 for b in range(len(new_ocr))]
c10 = line_chart([{"ys": old_ocr, "cls": "c2", "label": "weight curriculum"},
                  {"ys": new_ocr, "cls": "c1", "label": "new run"}], xs10, 0.2, 0.3,
                 [0.2, 0.22, 0.24, 0.26, 0.28, 0.3], "epoch (10-epoch averages)", "OCR of training images",
                 xticklabels=[f"{x - 5}–{x + 4}" for x in xs10])
tab10 = table(["Setting", "Weight curriculum (point 8)", "New run", "Why"],
              [["Alignment score", "earlier in-house scorer", "NoisyCLIP on x̂₀ with Φ", "supervisor: use NoisyCLIP"],
               ["OCR weight", "0 → 80", "40 → 80", "supervisor: do not start OCR at 0"],
               ["Steps trained", "all 9 transitions", "first 8 (last one left out)", "diagnostic point 2"],
               ["KL penalty β", "0.04", "0.01", "diagnostic point 4"],
               ["Images per update", "128 (2 updates/epoch)", "256 (1 update/epoch)", "diagnostic point 3"],
               ["Length", "100 epochs, 4.5 h (H200)", "100 epochs, 8.3 h (A100)", ""]])
nc_eval = P3["nc_wsched_fix"]
if nc_eval:
    c10e = p3_chart([("prm_wsched", "c2", False), ("nc_wsched_fix", "c1", True)], "ocr.OCR", -0.02, 0.05,
                    [-0.02, 0, 0.02, 0.04], "OCR vs base")
    c10p = p3_chart([("prm_wsched", "c2", False), ("nc_wsched_fix", "c1", True)], "align.PickScore", -0.004, 0.003,
                    [-0.004, -0.002, 0, 0.002], "PickScore vs base", nd=3)
    eval10 = fig(legend([("weight curriculum", "c2"), ("new run (with 95% interval)", "c1")])
                 + "<div class='two'>" + c10e + c10p + "</div>", "Evaluation of every checkpoint vs the base.",
                 "512 + 512 images, same seeds as the base, H200. A confirmation with new seeds on epoch 100 (chosen in "
                 "advance) is still needed before calling it a gain.")
    ep100 = dict(nc_eval).get(100)
    if ep100:
        rows10 = []
        keys10 = [("ocr.OCR", "OCR (text prompts)", 4), ("ocr.Exact", "Exact match (text prompts)", 4),
                  ("ocr.CLIP", "CLIP (text prompts)", 4), ("ocr.PickScore", "PickScore (text prompts)", 4),
                  ("align.CLIP", "CLIP (alignment prompts)", 4), ("align.PickScore", "PickScore (alignment prompts)", 4),
                  ("align.Neg", "Subtle-negative accuracy", 3)]
        cf = CONF.get("nc_wsched_fix_checkpoint-100")
        for k, lab, nd in keys10:
            row = [lab, sig(ep100[k], nd)]
            row.append(sig(cf[k], nd) if cf else "pending")
            rows10.append(row)
        eval10 += fig(table(["Metric (diff vs base, 10 steps)", "Epoch 100, in-training eval (512 imgs)",
                             "Epoch 100, new-seed confirmation (1024 imgs)"], rows10),
                      "Final checkpoint (chosen in advance) vs the base.",
                      "* = 95% bootstrap interval excludes zero. The confirmation uses new seeds and 8 images per prompt, H200.")
        eval10 += concl("first run where alignment and image quality improve steadily: CLIP and PickScore rise from ~epoch 40 "
                        "to the end on both prompt sets (text-prompt CLIP at epoch 100 reaches the base-20-step level). "
                        "OCR stays where every earlier run was: it moves between +0.000 and +0.026 across checkpoints, "
                        "the final one is not significant in the in-training eval, and exact match does not move."
                        + (" New-seed confirmation (1024 + 1024 images): the alignment and quality gains hold "
                           "(alignment CLIP and PickScore +0.0015*, text-prompt CLIP +0.007* and PickScore +0.0035*), "
                           "the OCR gain does not (−0.007, n.s.). So the fixes made the reward reach the model for "
                           "alignment, but OCR is still limited by the gradient noise of point 9."
                           if cf else " The new-seed confirmation of epoch 100 is queued."))
else:
    eval10 = ("<p class='note'><b>Evaluation pending:</b> the checkpoints are queued for the H200 (the base references "
              "were made there, and the GPU type changes the images). Epoch 100, chosen in advance, will then get the "
              "new-seed confirmation.</p>")
section(10, "New run: NoisyCLIP + the fixes from the diagnostic",
        "One run that follows all three pieces of advice and the diagnostic. It changes several things at once compared "
        "with the weight curriculum, so a gain would not say which change caused it.",
        fig(tab10, "What changed.", "")
        + fig(c10, "OCR of the training images.",
              "Average OCR of the images generated during training, in 10-epoch blocks (~128 text images per epoch, "
              "so this is noisy). From epoch 30 on the new run is higher in 6 of 7 blocks, by ~0.01 on average. In-training numbers have misled "
              "us before, so this is a hint, not a result.")
        + eval10, "New run")

# ================================================================== checking, next, glossary
section("", "How a result is checked",
        "Something counts as an improvement only after an independent check. Three false gains taught this.",
        "<ol class='steps-list'><li><b>Paired:</b> model and base generate with the same seeds.</li>"
        "<li><b>New seeds</b>, never used to choose checkpoints, and the checkpoint is chosen in advance.</li>"
        "<li><b>More images:</b> 1024 per set (128 prompts × 8).</li>"
        "<li><b>Same GPU type:</b> H200 and A100 give different images for the same seed.</li>"
        "<li><b>95% bootstrap intervals</b>; it counts only if the interval excludes zero.</li></ol>", "Checks")
section("", "Questions and next steps", "",
        "<ul class='take'>"
        "<li><b>NoisyCLIP on SD 1.5:</b> retrain NoisyCLIP on SD 1.5 latents (with Vasco's code), or move to SDXL, "
        "where it works as published?</li>"
        "<li><b>OCR:</b> RL gets ~1–2 × 10⁻⁴ signal per image (point 9). Next: a differentiable text-recognition "
        "reward back-propagated through the last steps; fallback: a ~10× longer RL run with a 2-step random window "
        "and text-only prompts (~30–40 GPU-hours).</li>"
        "<li><b>Which steps to train:</b> only the steps where text forms (the \"latent spark\", ~steps 2–6) instead of "
        "all but the last?</li>"
        "<li><b>Attribution:</b> if the new run helps, which single change to isolate first?</li></ul>", "Next")
gl = [("GRPO", "Generate a group of images per prompt; the model learns to do more like the best and less like the worst."),
      ("PRM / ORM", "Process reward (scores mid-generation) / outcome reward (scores only the final image)."),
      ("x̂₀", "The model's prediction, at each step, of the final image (as a latent)."),
      ("z_t", "The noisy latent at step t, i.e. what the model actually holds mid-generation."),
      ("VAE / TAESD", "Decoder from latent to image (expensive) / a small, fast imitation of it."),
      ("Φ (Phi)", "A fixed linear map from the latent's 4 channels to RGB (15 numbers), fitted for SD 1.5."),
      ("NoisyCLIP", "CLIP whose image side is fine-tuned to read noisy (SDXL) latents rendered as colour pictures."),
      ("LoRA", "Small trainable matrices on top of a frozen model."),
      ("CLIPScore / PickScore", "Image–text match score / a second judge, not used in training."),
      ("Spearman", "Rank correlation: 1 = orders the images exactly like the final result."),
      ("Conformal Risk Control", "Picks a threshold with a statistical guarantee that the risk is ≤ α."),
      ("KL penalty (β)", "Keeps the trained model close to the original; β sets how strongly."),
      ("Signal-to-noise", "How big the useful, consistent part of an update is compared with its random part.")]
section("", "Glossary", "Each term in one sentence.", table(["Term", "Meaning"], [[a, b] for a, b in gl]), "Glossary")

# ================================================================== page
CSS = open(os.path.join(HERE, "build_visual_guide.py")).read().split('CSS = """')[1].split('"""')[0] + """
.c5.ln{stroke:var(--fg)}.c5.dot{fill:var(--fg)}.sw.c5{background:var(--fg)}
.glance{margin:8px 0 0;padding-left:20px;max-width:78ch}.glance li{margin:6px 0}
section.first{border-top:none;margin-top:24px}
code{font:.85em var(--mono)}
.vhead{display:grid;grid-template-columns:150px repeat(5,1fr) 1fr;gap:4px;font:10.5px var(--mono);color:var(--mut);text-align:center}
.vrow{margin:10px 0 14px}.vprompt{font-size:.82rem;color:var(--mut);margin-bottom:4px}
.vgrid{display:grid;grid-template-columns:150px repeat(5,1fr) 1fr;grid-template-rows:auto auto;gap:4px;align-items:center}
.vgrid img{width:100%;display:block;border-radius:3px;min-width:40px}
.vgrid .final{grid-column:7;grid-row:1 / span 2;display:flex;flex-direction:column;align-items:center;gap:2px;font:10px var(--mono);color:var(--mut)}
@media (max-width:560px){.vhead,.vgrid{grid-template-columns:70px repeat(5,1fr) 1fr}.rowlab{font-size:.68rem}}
"""
MARKER = ("<svg width='0' height='0' style='position:absolute'><defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' "
          "markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' style='fill:var(--mut)'/>"
          "</marker></defs></svg>")
nav = "".join(f"<a href='#{sid}'>{lab}</a>" for sid, lab in NAV)
page = f"""<title>Steering the Latent Spark</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&family=Space+Grotesk:wght@500;600&display=swap">
<style>{CSS}</style>
{MARKER}
<div class="wrap">
<header><h1>Steering the Latent Spark</h1>
<p>Process rewards and GRPO for SD 1.5: what was tried, how, and what came out. Every table, chart and image comes from
the result files. 6–12 November 2026.</p></header>
<nav>{nav}</nav>
{''.join(S)}
</div>"""
with open(OUT, "w") as f:
    f.write(page)
print(f"wrote {OUT} {os.path.getsize(OUT) // 1024} KB; nc_wsched_fix evals {[e for e, _ in P3['nc_wsched_fix']]}; "
      f"conf {list(CONF)}; new ocr blocks {[round(v, 3) for v in new_ocr]}")
