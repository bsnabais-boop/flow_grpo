"""Figures for paper/ (2026-11-16). Reads result files, writes PDFs to paper/figures/.
  overview.pdf      method schematic (10-step trajectory, SDE window, noise judge, OCR on final image)
  nc_steps.pdf      Spearman of each judge with the final CLIP score along the trajectory
  grad_share.pdf    share of the policy-gradient norm per denoising transition (base model)
  ocr_curves.pdf    (a) held-out OCR vs training epoch, two seeds; (b) OCR vs number of sampling steps
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper/figures")
L = lambda *p: os.path.join(ROOT, "logs", *p)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, INK, MUTED = "#8a8a85", "#1f1f1e", "#6b6b66"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "lines.linewidth": 1.6, "pdf.fonttype": 42,
                     "legend.frameon": False})


def boot(per_prompt, n=2000):
    rng = np.random.default_rng(0)
    b = per_prompt[rng.integers(0, len(per_prompt), (n, len(per_prompt)))].mean(1)
    return np.percentile(b, [2.5, 97.5])


def ocr_eval(run):
    z = np.load(L("phase3_eval", run, "ocr.npz"))
    pid = z["prompt_ids"]
    return np.array([z["ocr"][pid == p].mean() for p in np.unique(pid)])


# ---------------------------------------------------------------- overview
def overview():
    fig, ax = plt.subplots(figsize=(5.5, 1.9))
    ax.set_xlim(0, 11.2); ax.set_ylim(0, 3.4); ax.axis("off"); ax.grid(False)
    xs = np.linspace(0.5, 9.5, 11)
    ax.add_patch(FancyBboxPatch((xs[2] - 0.25, 1.55), xs[4] - xs[2] + 0.5, 0.9, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#cde2fb", ec=BLUE, lw=0.8))
    ax.text((xs[2] + xs[4]) / 2, 2.62, "SDE window (trained)", ha="center", color=BLUE, fontsize=7)
    ax.plot([xs[0], xs[5]], [3.0, 3.0], color=MUTED, lw=0.8)
    ax.text(xs[0], 3.1, "latent spark: layout and glyphs form", ha="left", color=MUTED, fontsize=6.5)
    for i, x in enumerate(xs):
        ax.add_patch(plt.Circle((x, 2.0), 0.2, fc="white", ec=INK, lw=0.8, zorder=3))
        ax.text(x, 2.0, "$z_T$" if i == 0 else ("$x_0$" if i == 10 else str(i)), ha="center", va="center",
                fontsize=6.5, zorder=4)
        if i < 10:
            stoch = 2 <= i < 4
            ax.annotate("", (xs[i + 1] - 0.21, 2.0), (x + 0.21, 2.0),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE if stoch else MUTED, lw=1.4 if stoch else 0.8,
                                        ls="-" if stoch else (0, (2, 1.5))))
    ax.text(xs[0] - 0.35, 2.0, "", ha="right")
    # noise judge
    for k in (4, 6, 8, 10):
        ax.annotate("", (xs[k], 1.05), (xs[k], 1.78), arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=0.8))
    ax.add_patch(FancyBboxPatch((xs[4] - 0.5, 0.35), xs[10] - xs[4] + 0.8, 0.68, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#d4f2e6", ec=AQUA, lw=0.8))
    ax.text((xs[4] + xs[10]) / 2, 0.69, r"noise judge: NoisyCLIP$(\Phi(\hat x_0))$, 0.10 UNet steps", ha="center",
            va="center", fontsize=6.5, color=INK)
    # pruning
    ax.text(xs[7] + 0.1, 2.75, "conformal\npruning", ha="left", va="center", fontsize=6.5, color=ORANGE)
    ax.annotate("", (xs[6] + 0.12, 2.18), (xs[7], 2.7), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.8))
    # final reward
    ax.add_patch(FancyBboxPatch((10.05, 1.55), 1.1, 0.9, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#fde3d6",
                                ec=ORANGE, lw=0.8))
    ax.text(10.6, 2.0, "VAE\n+ OCR", ha="center", va="center", fontsize=6.5)
    ax.annotate("", (10.03, 2.0), (xs[10] + 0.21, 2.0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
    ax.annotate("", (10.6, 1.05), (10.6, 1.53), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.8))
    ax.text(10.6, 0.69, "reward:\nalign + OCR", ha="center", va="center", fontsize=6.5)
    ax.text(10.6, 0.12, "(final image)", ha="center", va="center", fontsize=6, color=MUTED)
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(OUT, "overview.pdf"))


# ---------------------------------------------------------------- NoisyCLIP per step
def nc_steps():
    ps = json.load(open(L("phase1_noisyclip/eval10_noisyclip/summary.json")))["per_step"]
    steps = [r["step"] for r in ps]
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.9), sharey=True)
    series = [("nc_zt", r"NoisyCLIP ($z_t$, SDXL map)", GRAY, "--"), ("nc_x0phi", r"NoisyCLIP ($\Phi(\hat x_0)$)", BLUE, "-"),
              ("proj_x0", r"CLIP ($\Phi(\hat x_0)$)", AQUA, "-"), ("taesd", "TAESD + CLIP", ORANGE, "-")]
    for ax, key, title in [(axs[0], "spearman", "all images"), (axs[1], "grp_spearman", "within prompt")]:
        for m, lab, c, ls in series:
            ax.plot(steps, [r[f"{m}_{key}"] for r in ps], color=c, ls=ls, marker="o", ms=2.5, label=lab)
        dec = [(r["step"], r[f"dec_{key}"]) for r in ps if not np.isnan(r[f"dec_{key}"])]
        ax.plot(*zip(*dec), ls="none", marker="D", ms=4, color=INK, label="VAE + CLIP")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(title, fontsize=8, color=INK); ax.set_xlabel("denoising step (of 10)"); ax.set_xticks(range(1, 11))
    axs[0].set_ylabel("Spearman with final CLIP")
    axs[1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "nc_steps.pdf"))


# ---------------------------------------------------------------- gradient share
def grad_share():
    g = json.load(open(L("diag_ocr_signal/v2_fix/base.json")))["gradient"]
    share = np.array(g["per_transition_norm_share"]); cos = np.array(g["per_transition_cos_with_total"])
    n = len(share); x = np.arange(1, n + 1)
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.7))
    axs[0].bar(x, share * 100, color=[BLUE] * (n - 1) + [ORANGE], width=0.7)
    axs[0].set_ylabel("% of gradient norm"); axs[0].set_xlabel(r"transition $t\to t-1$ (1 = first, 9 = last)")
    axs[0].text(n, share[-1] * 100 + 2, f"{share[-1]*100:.0f}%", ha="center", fontsize=7)
    axs[1].bar(x, cos, color=[BLUE] * (n - 1) + [ORANGE], width=0.7)
    axs[1].set_ylabel("cosine with total"); axs[1].set_xlabel("transition"); axs[1].set_ylim(0, 1.05)
    for ax in axs:
        ax.set_xticks(x); ax.grid(axis="x", visible=False)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "grad_share.pdf"))


# ---------------------------------------------------------------- OCR curves
def ocr_curves():
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.0), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axs[0]
    base = ocr_eval("base_10steps")
    ref = {s: ocr_eval(f"base_{s}steps").mean() for s in (10, 20, 50)}
    for s, ls, dy in [(10, ":", 0), (20, "--", 0.005), (50, "-.", -0.005)]:
        ax.axhline(ref[s], color=GRAY, ls=ls, lw=0.9)
        ax.text(508, ref[s] + dy, f"base {s}", va="center", fontsize=6.5, color=MUTED)
    for run, lab, c in [("ocr_long_win23", "seed 1", BLUE), ("ocr_long_win23_seed2", "seed 2", ORANGE)]:
        ep, m, lo, hi = [0], [base.mean()], [base.mean()], [base.mean()]
        for e in range(50, 501, 50):
            p = ocr_eval(f"{run}_checkpoint-{e}")
            ci = boot(p - base)
            ep.append(e); m.append(p.mean()); lo.append(base.mean() + ci[0]); hi.append(base.mean() + ci[1])
        ax.fill_between(ep, lo, hi, color=c, alpha=0.15, lw=0)
        ax.plot(ep, m, color=c, marker="o", ms=2.5, label=lab)
    ax.set_xlabel("training epoch (512 images each)"); ax.set_ylabel("held-out OCR (10 steps)")
    ax.set_xlim(0, 560); ax.legend(loc="upper left", fontsize=7)
    ax.set_title("(a) during training", fontsize=8, color=INK)
    # (b) steps
    ax = axs[1]
    d = L("ocr_robustness/v1")
    def mean_ci(name, key="strict"):
        return np.load(os.path.join(d, name + ".npz"))[key].mean()
    sb = [10, 15, 20, 50]; so = [10, 15, 20]
    for key, ls, lab in [("paddle", "-", "reward"), ("strict", "--", "strict")]:
        ax.plot(sb, [mean_ci(f"base_{s}", key) for s in sb], color=GRAY, ls=ls, marker="o", ms=2.5,
                label="base" if key == "paddle" else None)
        yo = [mean_ci(f"long500_{s}", key) for s in so]
        ax.plot(so, yo, color=BLUE, ls=ls, marker="o", ms=2.5, label="ours (seed 1)" if key == "paddle" else None)
        ax.text(22, yo[-1], lab, va="center", fontsize=6.5, color=MUTED)
    ax.set_xscale("log"); ax.set_xticks(sb); ax.set_xticklabels([str(s) for s in sb]); ax.minorticks_off()
    ax.set_xlabel("sampling steps"); ax.set_ylabel("OCR")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("(b) at inference", fontsize=8, color=INK)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "ocr_curves.pdf"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    overview(); nc_steps(); grad_share(); ocr_curves()
    for f in ["overview", "nc_steps", "grad_share", "ocr_curves"]:
        print(os.path.join(OUT, f + ".pdf"))
