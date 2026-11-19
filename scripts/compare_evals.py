"""Paired comparison of several eval_intermediate_latents.py runs that used the same prompts and seeds.

The first directory is the reference (e.g. base SD1.5). For every other run it reports the final-image reward
difference vs the reference with a standard error over prompts, then a per-step table: mean OCR of the x0 preview
at each denoising step and how well that preview ranks samples within a prompt group (Spearman vs final reward).

Usage: python scripts/compare_evals.py logs/intermediate_eval/win4_base logs/intermediate_eval/win4_ckpt60 ...
"""
import argparse
import json
import os

import numpy as np


def load(d):
    r = np.load(os.path.join(d, "rewards.npz"))
    with open(os.path.join(d, "summary.json")) as f:
        s = json.load(f)
    return r, s


def paired(ref, run, key, groups):
    d = run[key] - ref[key]
    dp = np.array([d[groups == g].mean() for g in np.unique(groups)])
    se = dp.std(ddof=1) / np.sqrt(len(dp))
    return d.mean(), se, (dp > 0).sum(), (dp < 0).sum(), (dp == 0).sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    args = p.parse_args()

    runs = [(os.path.basename(os.path.normpath(d)), *load(d)) for d in args.dirs]
    ref_name, ref, _ = runs[0]
    groups = ref["group_ids"]
    for name, r, _ in runs[1:]:
        assert (r["prompts"] == ref["prompts"]).all(), f"{name}: prompts differ from {ref_name}, not paired"

    print(f"Final image, {len(groups)} samples ({len(np.unique(groups))} prompts), paired vs {ref_name}")
    print(f"{'run':24s} {'ocr':>6} {'exact':>6} {'clip':>6} {'d_ocr':>8} {'SE':>6} {'z':>5} {'prompts +/-/=':>14}")
    for name, r, _ in runs:
        line = (f"{name:24s} {r['final_ocr'].mean():>6.3f} {(r['final_ocr'] >= 1).mean():>6.3f} "
                f"{r['final_clipscore'].mean():>6.3f}")
        if name != ref_name:
            m, se, up, down, same = paired(ref, r, "final_ocr", groups)
            line += f" {m:>+8.4f} {se:>6.4f} {m / se:>+5.1f} {up:>5}/{down}/{same}"
        print(line)

    n_steps = ref["step_ocr"].shape[1]
    print("\nPer step: preview OCR | within-group Spearman of preview vs final reward")
    print(f"{'step':>4} " + " ".join(f"{n[:14]:>14s} {'':>6}" for n, _, _ in runs))
    for i in range(n_steps):
        cells = []
        for _, r, s in runs:
            cells.append(f"{r['step_ocr'][:, i].mean():>14.3f} {s['per_step'][i]['within_group_spearman']:>6.2f}")
        print(f"{i + 1:>4} " + " ".join(cells))


if __name__ == "__main__":
    main()
