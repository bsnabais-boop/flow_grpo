"""Meeting report for 2026-11-17 (covers work from 2026-11-10 to 2026-11-16).
Builds scripts/report/meeting1117/{index.html, img/*} from the result files. Rebuild + republish the same path."""
import glob, json, os, shutil, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "meeting1117"); IMG = os.path.join(OUT, "img")
os.makedirs(IMG, exist_ok=True)
L = lambda *p: os.path.join(ROOT, "logs", *p)
BLUE, ORANGE, AQUA, YELLOW, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8a85", "#1f1f1e", "#6b6b66"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "lines.linewidth": 1.8, "legend.frameon": False})


def save(fig, name):
    fig.savefig(os.path.join(IMG, name), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def boot(pp, n=2000):
    rng = np.random.default_rng(0)
    b = pp[rng.integers(0, len(pp), (n, len(pp)))].mean(1)
    return np.percentile(b, [2.5, 97.5])


def per_prompt(run, kind, key, base="phase3_confirm"):
    z = np.load(L(base, run, f"{kind}.npz"))
    v = (z["ocr"] >= 1.0).astype(float) if key == "exact" else z[key]
    pid = z["prompt_ids"]
    return np.array([v[pid == p].mean() for p in np.unique(pid)])


def diff(run, kind, key, ref="base_10steps"):
    d = per_prompt(run, kind, key) - per_prompt(ref, kind, key)
    lo, hi = boot(d)
    return d.mean(), lo, hi


# ------------------------------------------------------------------ charts
def chart_steps():
    s = [4, 6, 8, 10, 20, 50]; ocr = [0.047, 0.065, 0.097, 0.118, 0.160, 0.193]; clip = [0.819, 0.923, 0.953, 0.967, 0.977, 0.975]
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.6))
    axs[0].plot(s, ocr, color=BLUE, marker="o"); axs[0].set_title("OCR score (text legibility)", fontsize=9)
    axs[1].plot(s, clip, color=AQUA, marker="o"); axs[1].set_title("CLIP score (relative to its max)", fontsize=9)
    for ax in axs:
        ax.set_xscale("log"); ax.set_xticks(s); ax.set_xticklabels(s); ax.minorticks_off(); ax.set_xlabel("sampling steps")
        ax.axvline(10, color=ORANGE, lw=0.8, ls="--")
    axs[0].annotate("10 steps keep only\n61% of the 50-step OCR", (10, 0.118), (12, 0.06), fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    save(fig, "steps.png")


RUNS = [("base_20steps", "Base model, 20 steps", GRAY), ("base_50steps", "Base model, 50 steps", GRAY),
        ("orm_checkpoint-120", "ORM (CLIP on decoded image)", MUTED),
        ("wsched_checkpoint-200", "Weight schedule run", MUTED), ("ssched_checkpoint-200", "Step schedule run", MUTED),
        ("nc_wsched_fix_checkpoint-100", "NoisyCLIP + weight schedule + fixes", AQUA),
        ("opt2_pilot_v1_checkpoint-600", "Differentiable OCR (fix 2)", ORANGE),
        ("ocr_long_win23_checkpoint-500", "Long OCR run, seed 1 (fix 1)", BLUE),
        ("ocr_long_win23_seed2_checkpoint-500", "Long OCR run, seed 2 (fix 1)", BLUE)]


def chart_confirm(kind, key, title, fname, xlim=None, skip=()):
    rows = [(lab, c, *diff(r, kind, key)) for r, lab, c in RUNS
            if os.path.exists(L("phase3_confirm", r, kind + ".npz")) and r not in skip]
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(rows) + 0.8))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, c, m, lo, hi) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=c, lw=2.2, solid_capstyle="round")
        ax.plot(m, yi, "o", color=c, ms=6)
        sig = lo > 0 or hi < 0
        ax.text(hi if m >= 0 else lo, yi + 0.28, f"{m:+.4f}" + ("*" if sig else ""), fontsize=7.5, color=INK,
                ha="right" if m < 0 else "left")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"difference vs base model at 10 steps (95% CI) - {title}")
    if xlim: ax.set_xlim(*xlim)
    save(fig, fname)
    return rows


def chart_snr():
    v = [("Own z_T, all steps (normal runs)", 1.3, 1.0), ("Shared z_T, all steps", -1.0, 1.2), ("Own z_T, window 1-4", 1.0, 1.2),
         ("Shared z_T, window 1-4", 0.0, 1.7), ("Shared z_T, window 2-3 (fix 1)", 1.6, 1.0)]
    fig, ax = plt.subplots(figsize=(8, 2.4))
    y = np.arange(len(v))[::-1]
    for yi, (lab, m, se) in zip(y, v):
        ax.plot([m - 1.96 * se, m + 1.96 * se], [yi, yi], color=BLUE, lw=2.2, solid_capstyle="round"); ax.plot(m, yi, "o", color=BLUE)
    ax.axvline(0, color=INK, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([x[0] for x in v]); ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"per-image gradient signal $\rho^2$ ($\times 10^{-4}$, 95% CI)")
    save(fig, "snr.png")


def chart_pilot():
    rows = [json.loads(l) for l in open(L("opt2/pilot_v1/log.jsonl"))]
    st = np.array([r["step"] for r in rows]); pad = np.array([r["paddle"] for r in rows]); ct = np.array([r["clip_text"] for r in rows])
    k = 50; sm = lambda x: np.convolve(x, np.ones(k) / k, mode="valid")
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.4))
    axs[0].plot(st[k - 1:], sm(pad), color=ORANGE); axs[0].set_title("PaddleOCR on training images (real legibility)", fontsize=9)
    axs[1].plot(st[k - 1:], sm(ct), color=GRAY); axs[1].set_title("CLIP text-match term (part of the proxy)", fontsize=9)
    for ax in axs: ax.set_xlabel("update")
    save(fig, "pilot.png")


def chart_long_train():
    # training-image OCR per 50 epochs, both seeds, from the slurm logs' reward prints is noisy; use W&B-free summary from eval
    pass


# ------------------------------------------------------------------ image strips
def font(sz):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(f):
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def labelled_rows(rows, fname, tile=200, labw=170, head=None, wrap=None):
    """rows: list of (label, PIL image grid of 8x256 tiles or list of tiles)"""
    import textwrap
    n = max(len(t) for _, t in rows)
    if head and wrap:
        head = ["\n".join(textwrap.wrap(h, wrap)) for h in head]
    hh = (18 * max(h.count("\n") + 1 for h in head) + 20) if head else 0
    W, H = labw + n * tile, hh + len(rows) * tile
    can = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(can); f = font(15); fh = font(13)
    if head:
        for j, h in enumerate(head):
            d.multiline_text((labw + j * tile + tile / 2, hh / 2), h, fill=INK, font=fh, anchor="mm", align="center")
    for i, (lab, tiles) in enumerate(rows):
        d.multiline_text((10, hh + i * tile + tile / 2), lab, fill=INK, font=f, anchor="lm")
        for j, t in enumerate(tiles):
            can.paste(t.resize((tile - 4, tile - 4)), (labw + j * tile + 2, hh + i * tile + 2))
    can.save(os.path.join(IMG, fname), quality=88)


def split_grid(path, k=8):
    """first row of k square tiles (ocr_robustness grids have 2 rows: prompts 0-7 on top, 8-15 below)"""
    im = Image.open(path).convert("RGB"); w = im.width // k
    return [im.crop((i * w, 0, (i + 1) * w, w)) for i in range(k)]


def targets(n=8):
    """the first n OCR test prompts = the prompts of the grid tiles (phase3_eval / ocr_robustness order)"""
    return [l.strip() for l in open(os.path.join(ROOT, "dataset/ocr_simple/test.txt")) if l.strip()][:n]


def align_prompts(n=8):
    """the first n alignment eval prompts, in phase3_eval.py's order"""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import random
    from phase3_eval import load_genai_bench
    P, _ = load_genai_bench(os.path.join(ROOT, "dataset/genai_bench"))
    ids = sorted(P); random.Random(0).shuffle(ids)
    return [" ".join(P[i].split()) for i in ids[32:32 + n]]


def strips():
    head = targets()
    rows = [("Base, 10 steps", "base_10steps"), ("Base, 50 steps", "base_50steps"),
            ("Differentiable\nOCR (failed)", "opt2_pilot_v1_checkpoint-600"),
            ("Ours seed 1,\n10 steps", "ocr_long_win23_checkpoint-500"), ("Ours seed 2,\n10 steps", "ocr_long_win23_seed2_checkpoint-500")]
    labelled_rows([(lab, split_grid(L("phase3_confirm", r, "grid_ocr.png"))) for lab, r in rows], "grid_ocr.jpg", head=head, wrap=24)
    labelled_rows([("Base, 20 steps", split_grid(L("ocr_robustness/v1/grid_base_20.png"))[:8]),
                   ("Ours seed 1,\n20 steps", split_grid(L("ocr_robustness/v1/grid_long500_20.png"))[:8])], "grid_20.jpg",
                  head=head, wrap=24)
    pa = [l.strip() for l in open(os.path.join(ROOT, "dataset/genai_bench/test.txt"))][:8] if os.path.exists(
        os.path.join(ROOT, "dataset/genai_bench/test.txt")) else None
    labelled_rows([("Base, 10 steps", split_grid(L("phase3_confirm/base_10steps/grid_align.png"))),
                   ("NoisyCLIP run,\n10 steps", split_grid(L("phase3_confirm/nc_wsched_fix_checkpoint-100/grid_align.png")))],
                  "grid_align.jpg", head=align_prompts(), wrap=24)
    A = os.path.join(HERE, "assets")
    st = [2, 4, 6, 8, 10]
    labelled_rows([("NoisyCLIP view\n(SDXL colour map,\non z_t)", [Image.open(f"{A}/nc_0_{s}.png").convert("RGB") for s in st]),
                   ("Our SD1.5 map\non predicted x0", [Image.open(f"{A}/phi_0_{s}.png").convert("RGB") for s in st]),
                   ("Final image", [Image.open(f"{A}/final_0.png").convert("RGB")])], "views.jpg", tile=150, labw=190,
                  head=[f"step {s}/10" for s in st])


def copy_paper_figs():
    import paper_figures as pf
    import matplotlib.figure as mf
    orig = mf.Figure.savefig
    def sv(self, path, **k):
        orig(self, path, **k); orig(self, os.path.join(IMG, os.path.basename(path).replace(".pdf", ".png")), dpi=200,
                                    facecolor="white")
    mf.Figure.savefig = sv
    pf.overview(); pf.nc_steps(); pf.grad_share(); pf.ocr_curves()
    mf.Figure.savefig = orig
    q = os.path.join(ROOT, "paper/figures/qualitative.pdf")
    qp = "/tmp/claude-1065000101/-user-home-b-nabais-flow-grpo/72711de6-fc5e-4592-83b9-4ac70e5dfda0/scratchpad/qual.png"
    if os.path.exists(qp):
        shutil.copy(qp, os.path.join(IMG, "qualitative.png"))


if __name__ == "__main__":
    chart_steps(); chart_snr(); chart_pilot(); strips(); copy_paper_figs()
    ocr_rows = chart_confirm("ocr", "ocr", "OCR score", "conf_ocr.png")
    pick_rows = chart_confirm("align", "pick", "PickScore on alignment prompts", "conf_pick.png", skip=("opt2_pilot_v1_checkpoint-600",))
    clip_rows = chart_confirm("align", "clip", "CLIP on alignment prompts", "conf_clip.png", skip=("opt2_pilot_v1_checkpoint-600",))
    json.dump({"ocr": ocr_rows, "pick": pick_rows}, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=float)
    print(sorted(os.listdir(IMG)))
