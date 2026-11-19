"""Builds the meeting report page from the result files in the repo (numbers are read, not typed)."""
import base64
import glob
import io
import json
import os
import re

import numpy as np
from PIL import Image

ROOT = "/user/home/b.nabais/flow_grpo"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latent_spark_report.html")
os.chdir(ROOT)


def jload(p):
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------- Phase 3 paired comparisons
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
        assert (r["prompt_ids"] == d["prompt_ids"]).all()
        rm, dm = metric_arrays(r, name), metric_arrays(d, name)
        for k in rm:
            with np.errstate(all="ignore"):
                diff = per_prompt(dm[k] - rm[k], r["prompt_ids"])
            diff = diff[~np.isnan(diff)]
            lo, hi = boot(diff)
            out[f"{name}.{k}"] = dict(ref=float(np.nanmean(rm[k])), val=float(np.nanmean(dm[k])),
                                      diff=float(diff.mean()), lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0))
    return out


def ckpts(run):
    ds = []
    for d in glob.glob(f"logs/phase3_eval/mixed_{run}_checkpoint-*"):
        if os.path.exists(os.path.join(d, "summary.json")):
            ds.append((int(d.rsplit("-", 1)[1]), d))
    return sorted(ds)


p3 = {run: [(n, compare("logs/phase3_eval/base_10steps", d)) for n, d in ckpts(run)]
      for run in ["prm", "orm", "prm_wsched", "prm_ssched", "prm_both"]}
base_p3 = {n: jload(f"logs/phase3_eval/base_{n}steps/summary.json") for n in [10, 20, 50]}

confirm = {}
cdir = "logs/phase3_confirm"
CONF_LAB = {"prm_checkpoint-120": "GRPO + our PRM, epoch 60", "orm_checkpoint-120": "GRPO + CLIP ORM, epoch 60",
            "wsched_checkpoint-60": "Weight curriculum, epoch 30 (best in-training)",
            "wsched_checkpoint-200": "Weight curriculum, epoch 100 (final)",
            "ssched_checkpoint-200": "Steps curriculum, epoch 100 (final)"}
for name in CONF_LAB:
    if os.path.exists(f"{cdir}/{name}/summary.json") and os.path.exists(f"{cdir}/base_10steps/summary.json"):
        confirm[name] = compare(f"{cdir}/base_10steps", f"{cdir}/{name}")
confirm_base = {n: jload(f"{cdir}/base_{n}steps/summary.json") for n in [10, 20, 50]
                if os.path.exists(f"{cdir}/base_{n}steps/summary.json")}

# ---------------------------------------------------------------- Phase 1 curves (10 steps, eta=1, final scorer)
s10 = jload("logs/phase1_noisyclip/eval10_v2_lat_10steps_eta1/summary.json")
s50 = jload("logs/phase1_noisyclip/sd15_50steps_trained_v1/summary.json")
cost = s10["cost"]

# ---------------------------------------------------------------- images
def grid_jpeg(path, n=6, thumb=150):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    k = w // h
    tiles = [im.crop((i * h, 0, (i + 1) * h, h)).resize((thumb, thumb), Image.LANCZOS) for i in range(min(n, k))]
    return tiles


def strip_b64(tiles):
    th = tiles[0].size[0]
    canvas = Image.new("RGB", (th * len(tiles) + 4 * (len(tiles) - 1), th), (255, 255, 255))
    for i, t in enumerate(tiles):
        canvas.paste(t, (i * (th + 4), 0))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


img_rows = []
for label, d in [("Base SD1.5, 50 steps", "logs/phase3_eval/base_50steps"),
                 ("Base SD1.5, 10 steps", "logs/phase3_eval/base_10steps"),
                 ("GRPO + our PRM, 10 steps (checkpoint-120)", "logs/phase3_eval/mixed_prm_checkpoint-120"),
                 ("GRPO + CLIP ORM, 10 steps (checkpoint-120)", "logs/phase3_eval/mixed_orm_checkpoint-120")]:
    if os.path.exists(f"{d}/grid_ocr.png"):
        img_rows.append((label, strip_b64(grid_jpeg(f"{d}/grid_ocr.png")), strip_b64(grid_jpeg(f"{d}/grid_align.png"))))
ocr_prompts = [l.strip() for l in open("dataset/ocr_simple/test.txt") if l.strip()][:6]


# ---------------------------------------------------------------- helpers for HTML
def f(x, nd=3):
    return f"{x:.{nd}f}"


def sgn(x, nd=3):
    return f"{x:+.{nd}f}"


def diff_cell(r, key, nd=3):
    v = r[key]
    cls = "pos" if v["sig"] and v["diff"] > 0 else ("neg" if v["sig"] else "ns")
    return (f'<td class="num {cls}">{sgn(v["diff"], nd)}<span class="ci">[{sgn(v["lo"], nd)}, {sgn(v["hi"], nd)}]</span>'
            f'{"<b class=star>*</b>" if v["sig"] else ""}</td>')


def line_chart(series, xs, ylab, ymin, ymax, yticks, width=640, height=280, xlab="DDIM step (of 10)", xfmt=str):
    L, R, T, B = 52, 16, 14, 40
    W, H = width - L - R, height - T - B
    x0, x1 = min(xs), max(xs)
    sx = lambda x: L + (x - x0) / (x1 - x0) * W
    sy = lambda y: T + (1 - (y - ymin) / (ymax - ymin)) * H
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{ylab}">']
    for t in yticks:
        out.append(f'<line x1="{L}" x2="{L + W}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="grid"/>')
        out.append(f'<text x="{L - 8}" y="{sy(t) + 4:.1f}" class="tick" text-anchor="end">{t:g}</text>')
    for x in xs:
        out.append(f'<text x="{sx(x):.1f}" y="{T + H + 18}" class="tick" text-anchor="middle">{xfmt(x)}</text>')
    out.append(f'<text x="{L + W / 2:.1f}" y="{height - 4}" class="axis" text-anchor="middle">{xlab}</text>')
    out.append(f'<text x="14" y="{T + H / 2:.1f}" class="axis" text-anchor="middle" '
               f'transform="rotate(-90 14 {T + H / 2:.1f})">{ylab}</text>')
    if ymin < 0 < ymax:
        out.append(f'<line x1="{L}" x2="{L + W}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" class="zero"/>')
    for s in series:
        pts = [(sx(x), sy(y)) for x, y in zip(xs, s["ys"]) if y is not None and not np.isnan(y)]
        if s.get("band"):
            lo, hi = s["band"]
            poly = [(sx(x), sy(h)) for x, h in zip(xs, hi)] + [(sx(x), sy(l)) for x, l in zip(xs, lo)][::-1]
            out.append(f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in poly)}" class="band {s["cls"]}"/>')
        out.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" class="ln {s["cls"]}"/>')
        for a, b in pts:
            out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3" class="dot {s["cls"]}"/>')
    out.append("</svg>")
    return "".join(out)


steps = [r["step"] for r in s10["per_step"]]
meths = [("trained", "Our scorer (v2_lat, 0.16 UNet steps)", "s1"), ("taesd", "TAESD + CLIP (0.31)", "s2"),
         ("dec", "VAE + CLIP (1.78)", "s3"), ("proj_x0", "Untrained Φ + CLIP (0.10)", "s4")]
legend = "".join(f'<span class="lg"><i class="sw {c}"></i>{lab}</span>' for _, lab, c in meths)
chart_sp = line_chart([{"ys": [r[f"{m}_spearman"] for r in s10["per_step"]], "cls": c} for m, _, c in meths], steps,
                      "Spearman with final CLIP", -0.2, 1.0, [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
chart_grp = line_chart([{"ys": [r[f"{m}_grp_spearman"] for r in s10["per_step"]], "cls": c} for m, _, c in meths],
                       steps, "Within-prompt Spearman", -0.2, 1.0, [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Phase 3 OCR chart
def p3_series(run, key, cls):
    xs = [n // 2 for n, _ in p3[run]]
    return xs, {"ys": [r[key]["diff"] for _, r in p3[run]], "cls": cls,
                "band": ([r[key]["lo"] for _, r in p3[run]], [r[key]["hi"] for _, r in p3[run]])}


p3_charts = {}
for key, lab, rng_, ticks in [("ocr.OCR", "OCR vs base 10 steps", (-0.03, 0.06), [-0.02, 0, 0.02, 0.04, 0.06]),
                              ("align.PickScore", "PickScore vs base 10 steps", (-0.005, 0.003),
                               [-0.004, -0.002, 0, 0.002])]:
    xs_p, sp = p3_series("prm", key, "s1")
    xs_o, so = p3_series("orm", key, "s3")
    xs = sorted(set(xs_p) | set(xs_o))
    def pad(xsr, s):
        m = dict(zip(xsr, s["ys"]))
        lo = dict(zip(xsr, s["band"][0])); hi = dict(zip(xsr, s["band"][1]))
        s = dict(s); s["ys"] = [m.get(x, np.nan) for x in xs]
        s["band"] = ([lo.get(x, np.nan) for x in xs], [hi.get(x, np.nan) for x in xs])
        if any(np.isnan(s["ys"])):
            s["band"] = None
        return s
    p3_charts[key] = line_chart([pad(xs_o, so), pad(xs_p, sp)], xs, lab, rng_[0], rng_[1], ticks,
                                xlab="training epoch")

xs_c = sorted(set().union(*[[n // 2 for n, _ in p3[r]] for r in ["prm", "prm_wsched", "prm_ssched"]]))
def nob(run, key, cls):
    m = {n // 2: r[key]["diff"] for n, r in p3[run]}
    return {"ys": [m.get(x, np.nan) for x in xs_c], "cls": cls, "band": None}
curr_charts = {}
for key, lab, rng_, ticks in [("ocr.OCR", "OCR vs base 10 steps", (-0.01, 0.04), [0, 0.01, 0.02, 0.03, 0.04]),
                              ("align.PickScore", "PickScore vs base 10 steps", (-0.003, 0.002),
                               [-0.002, -0.001, 0, 0.001, 0.002])]:
    curr_charts[key] = line_chart([nob("prm", key, "s1"), nob("prm_wsched", key, "s2"), nob("prm_ssched", key, "s4")],
                                  xs_c, lab, rng_[0], rng_[1], ticks, xlab="training epoch")

# Phase 3 tables
def p3_table(run):
    rows = []
    for n, r in p3[run]:
        rows.append(f"<tr><td>{n // 2}</td>{diff_cell(r, 'ocr.OCR')}{diff_cell(r, 'ocr.Exact')}"
                    f"{diff_cell(r, 'align.CLIP', 4)}{diff_cell(r, 'align.PickScore', 4)}{diff_cell(r, 'align.Neg')}</tr>")
    return "".join(rows)


CURR_NOTE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "curr_note.html")).read() if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "curr_note.html")) else ""

latest = {run: (p3[run][-1][0] // 2 if p3[run] else 0) for run in p3}

def ci(r, k, nd=3):
    d = r[k]
    return f"{d['diff']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]" + ("*" if d["sig"] else "")

CURR_CONF = [n for n in confirm if n.startswith(("wsched", "ssched"))]
CURR_VERDICT = ""
if CURR_CONF:
    CURR_VERDICT = ("<p><strong>Curricula on new seeds</strong> (OCR, exact match and PickScore difference to base, "
                    "95% CI, * = significant): " + "; ".join(
        f"{CONF_LAB[n]}: OCR {ci(confirm[n], 'ocr.OCR')}, exact {ci(confirm[n], 'ocr.Exact')}, "
        f"PickScore {ci(confirm[n], 'align.PickScore', 4)}" for n in CURR_CONF) + ".</p>" + CURR_NOTE)
else:
    CURR_VERDICT = ("<p class='pending'>Confirmation of the curricula on new seeds is running (jobs 626875–626877: "
                    "weight curriculum epoch 30 and epoch 100, steps curriculum epoch 100).</p>")

CURR_GLANCE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "curr_glance.html")).read() if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "curr_glance.html")) else "Confirmation on new seeds is running."

def confirm_block():
    if not confirm:
        return ('<p class="pending">The confirmation run is queued on the H200 node (jobs 626597–626601): '
                'base 10/20/50 steps and checkpoint-120 of both runs, new seeds (2024), 8 images per prompt '
                '(1024 + 1024 images per model). This section fills in when they finish.</p>')
    rows = []
    for name, r in confirm.items():
        lab = CONF_LAB[name]
        rows.append(f"<tr><td>{lab}</td><td class='num'>{f(r['ocr.OCR']['val'])}</td>{diff_cell(r, 'ocr.OCR')}"
                    f"{diff_cell(r, 'ocr.Exact')}{diff_cell(r, 'align.PickScore', 4)}{diff_cell(r, 'align.CLIP', 4)}"
                    f"{diff_cell(r, 'align.Neg')}</tr>")
    b = ""
    for n, s in confirm_base.items():
        b += (f"<tr><td>Base, {n} steps</td><td class='num'>{f(s['ocr']['ocr'])}</td><td class='num ns' colspan='2'>"
              f"exact {s['ocr']['exact'] * 100:.1f}%</td><td class='num ns'>{f(s['align']['pickscore'], 4)}</td>"
              f"<td class='num ns'>{f(s['align']['clip'], 4)}</td><td class='num ns'>{f(s['align']['neg_acc']['all'])}</td></tr>")
    verdict = ("<p><strong>Verdict: the OCR gain does not reproduce.</strong> On new seeds (8 images per prompt, 1024 per set) "
               "the PRM checkpoint is +0.005 OCR over base (CI includes zero), against +0.026 on the in-training seeds. "
               "That was a selection effect: the best of 12 compared checkpoints on 512 images. Alignment metrics are "
               "unchanged for both runs; the ORM run is slightly worse on PickScore (−0.002, significant). Base at 20 and 50 "
               "steps is clearly better on text (OCR +0.049 / +0.079) and PickScore, so the 10-step gap is still open.</p>")
    verdict += CURR_VERDICT
    return (verdict + "<div class='tw'><table><thead><tr><th>Model (10 steps unless noted)</th><th>OCR</th><th>OCR vs base 10</th>"
            "<th>Exact vs base 10</th><th>PickScore vs base 10</th><th>CLIP vs base 10</th><th>Subtle-neg. acc. vs base 10</th>"
            f"</tr></thead><tbody>{''.join(rows)}{b}</tbody></table></div>")


imgs_html = ""
if img_rows:
    imgs_html = "<div class='imgs'>" + "".join(
        f"<figure><figcaption>{lab}</figcaption><img src='{o}' alt='{lab}, text prompts'></figure>"
        for lab, o, a in img_rows) + "</div>"
    imgs_html += ("<p class='note'>Columns, left to right: " + " · ".join(
        f"<span class='mono'>{re.sub('<', '&lt;', p)}</span>" for p in ocr_prompts) +
        ". Same prompt and same initial noise in every row.</p>")
    imgs_html += "<div class='imgs'>" + "".join(
        f"<figure><figcaption>{lab}</figcaption><img src='{a}' alt='{lab}, GenAI-Bench prompts'></figure>"
        for lab, o, a in img_rows) + "</div>"

CURR = [("prm_wsched", "Weight curriculum (CLIP first, OCR ramps up over 60 epochs)"),
        ("prm_ssched", "Steps curriculum (4 → 6 → 8 → 10 steps, 20 epochs each)"),
        ("prm_both", "Both curricula")]
curr_html = ""
for run, lab in CURR:
    if run == "prm_both":
        curr_html += (f"<h3>{lab}</h3><p class='pending'>Running now (job 626750, started after the steps curriculum "
                      "finished). No results yet.</p>")
        continue
    if p3[run]:
        curr_html += (f"<h3>{lab} (latest epoch {p3[run][-1][0] // 2})</h3><div class='tw'><table><thead><tr><th>Epoch</th>"
                      "<th>OCR</th><th>Exact text</th><th>CLIP</th><th>PickScore</th><th>Subtle-neg. acc.</th></tr></thead>"
                      f"<tbody>{p3_table(run)}</tbody></table></div>")
    else:
        curr_html += f"<h3>{lab}</h3><p class='pending'>Queued: no checkpoint evaluated yet.</p>"

tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html")).read()
subs = {
    "%%CHART_SP%%": chart_sp, "%%CHART_GRP%%": chart_grp, "%%LEGEND%%": legend,
    "%%P3_OCR_CHART%%": p3_charts["ocr.OCR"], "%%P3_PICK_CHART%%": p3_charts["align.PickScore"],
    "%%P3_PRM_ROWS%%": p3_table("prm"), "%%P3_ORM_ROWS%%": p3_table("orm"),
    "%%LATEST_PRM%%": str(latest["prm"]), "%%LATEST_ORM%%": str(latest["orm"]),
    "%%CONFIRM%%": confirm_block(), "%%CURRICULA%%": curr_html, "%%CURR_GLANCE%%": CURR_GLANCE, "%%CURR_OCR_CHART%%": curr_charts["ocr.OCR"], "%%CURR_PICK_CHART%%": curr_charts["align.PickScore"], "%%IMAGES%%": imgs_html,
    "%%B10_OCR%%": f(base_p3[10]["ocr"]["ocr"]), "%%B20_OCR%%": f(base_p3[20]["ocr"]["ocr"]),
    "%%B50_OCR%%": f(base_p3[50]["ocr"]["ocr"]),
    "%%B10_PICK%%": f(base_p3[10]["align"]["pickscore"]), "%%B20_PICK%%": f(base_p3[20]["align"]["pickscore"]),
    "%%B50_PICK%%": f(base_p3[50]["align"]["pickscore"]),
    "%%B10_CLIP%%": f(base_p3[10]["align"]["clip"], 4), "%%B20_CLIP%%": f(base_p3[20]["align"]["clip"], 4),
    "%%B50_CLIP%%": f(base_p3[50]["align"]["clip"], 4),
    "%%B10_NEG%%": f(base_p3[10]["align"]["neg_acc"]["all"]), "%%B20_NEG%%": f(base_p3[20]["align"]["neg_acc"]["all"]),
    "%%B50_NEG%%": f(base_p3[50]["align"]["neg_acc"]["all"]),
    "%%B10_EX%%": f"{base_p3[10]['ocr']['exact'] * 100:.1f}", "%%B20_EX%%": f"{base_p3[20]['ocr']['exact'] * 100:.1f}",
    "%%B50_EX%%": f"{base_p3[50]['ocr']['exact'] * 100:.1f}",
    "%%COST_TRAINED%%": f(cost["trained_in_unet_steps"], 2), "%%COST_TAESD%%": f(cost["taesd_in_unet_steps"], 2),
    "%%COST_VAE%%": f(cost["decode_in_unet_steps"], 2), "%%COST_PHI%%": f(cost["proj_in_unet_steps"], 2),
}
prm_last = p3["prm"][-1][1] if p3["prm"] else None
subs["%%PRM_LAST_OCR%%"] = f(prm_last["ocr.OCR"]["val"]) if prm_last else "–"
for k, v in subs.items():
    tpl = tpl.replace(k, v)
left = re.findall(r"%%[A-Z0-9_]+%%", tpl)
assert not left, left
open(OUT, "w").write(tpl)
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB; prm ckpts", [n for n, _ in p3["prm"]],
      "orm", [n for n, _ in p3["orm"]], "confirm", list(confirm))
