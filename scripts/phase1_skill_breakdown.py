"""Phase 1 analysis: break down the per-step PRM metrics by GenAI-Bench skill (basic/advanced, counting, spatial...).

Reads <run>/scores.npz from phase1_noisyclip.py (which stores a ';'-joined skill string per image) and reports, per
skill and per method at a few trajectory fractions: Spearman with the final CLIP score, within-prompt Spearman
(with a bootstrap CI over prompts) and wrong-prompt accuracy. No GPU needed.

Usage: python scripts/phase1_skill_breakdown.py logs/phase1_noisyclip/sd15_50steps_trained_v1 [--fracs 0.2 0.4 0.6 0.8]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_noisyclip import metrics, safe  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def grp_ci(own, final, groups, n_boot=1000, seed=0):
    per = np.array([safe(spearmanr, own[groups == k], final[groups == k]) for k in np.unique(groups)])
    per = per[~np.isnan(per)]
    if len(per) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [rng.choice(per, len(per)).mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--fracs", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--methods", nargs="+", default=["trained", "taesd", "dec", "proj_x0"])
    args = p.parse_args()
    d = np.load(os.path.join(args.run, "scores.npz"))
    final, groups, skills = d["final_own"], d["groups"], d["skills"]
    S = d[f"{args.methods[0]}_own"].shape[1]
    names = sorted({s for row in skills for s in row.split(";") if s})
    names = ["all"] + [n for n in ["basic", "advanced"] if n in names] + [n for n in names if n not in ("basic", "advanced")]
    out = {}
    print(f"run {args.run}: {len(final)} images, {len(np.unique(groups))} prompts, {S} steps")
    for name in names:
        mask = np.ones(len(final), bool) if name == "all" else np.array([name in s.split(";") for s in skills])
        n_p = len(np.unique(groups[mask]))
        print(f"\n=== {name}: {n_p} prompts, final CLIPScore {final[mask].mean():.3f} "
              f"(wrong-prompt final acc {(final[mask] > d['final_mis'][mask]).mean():.2f})")
        print(f"{'method':>8} {'frac':>5} | {'spearman':>8} {'grp_sp':>7} {'grp 95% CI':>15} {'disc':>5}")
        out[name] = {"prompts": n_p, "final_clip": float(final[mask].mean()), "rows": []}
        for meth in args.methods:
            own, mis = d[f"{meth}_own"], d[f"{meth}_mis"]
            for fr in args.fracs:
                i = min(S - 1, int(round(fr * S)) - 1)
                # decoded methods are only scored every k steps: use the nearest scored step at or before i
                while i > 0 and np.isnan(own[mask, i]).all():
                    i -= 1
                m = metrics(own[mask, i], mis[mask, i], final[mask], groups[mask])
                lo, hi = grp_ci(own[mask, i], final[mask], groups[mask])
                out[name]["rows"].append({"method": meth, "frac": fr, "step": i + 1, **m, "grp_ci": [lo, hi]})
                print(f"{meth:>8} {(i + 1) / S:>5.2f} | {m['spearman']:>8.2f} {m['grp_spearman']:>7.2f} "
                      f"[{lo:>5.2f},{hi:>5.2f}] {m['disc_acc']:>5.2f}")
    with open(os.path.join(args.run, "skill_breakdown.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
