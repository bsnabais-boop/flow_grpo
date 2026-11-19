"""Offline simulation of early pruning for best-of-N generation, from eval_intermediate_latents.py outputs.

Strategy prune(k, m): start N candidates per prompt, run all of them for k denoising steps, score their x0 previews,
keep the m best, finish only those, then output the best of the m by final reward (if m == 1 no final scoring).
Baseline best-of-n: run n full trajectories, score all n finals, output the best (n == 1: plain sampling, no scoring).

Because every trajectory in the saved data was run to the end, pruning can be simulated exactly: a pruned
trajectory's continuation is the same as in the full run.

Cost is in UNet-step units per prompt (one step of one sample, CFG included):
    prune(k, m):  N*k + m*(T-k) + N*s + (m > 1) * m*s
    best-of-n:    n*T + (n > 1) * n*s
where s = cost of scoring one image (VAE decode + reward) in UNet steps, measured from the run's own timings
unless --score-cost is given.

Usage: python scripts/early_selection.py logs/intermediate_eval/win4_base [--score-cost 1.5] [--metric ocr]
"""
import argparse
import json
import os
from itertools import product
from math import comb

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--score-cost", type=float, default=None, help="Cost of scoring one image in UNet-step units")
    p.add_argument("--select-by", default="avg", choices=["avg", "ocr", "clipscore"],
                   help="Which reward the previews / finals are ranked by")
    p.add_argument("--metric", default="ocr", choices=["avg", "ocr", "clipscore"],
                   help="Quality measure reported for the output image")
    args = p.parse_args()

    r = np.load(os.path.join(args.run_dir, "rewards.npz"))
    with open(os.path.join(args.run_dir, "summary.json")) as f:
        timing = json.load(f)["timing_per_batch_s"]
    T = r["step_avg"].shape[1]
    s = args.score_cost
    if s is None:
        unet_step = timing["unet_full_trajectory"] / T
        s = (timing["vae_decode_per_step"] + timing["reward_per_step"]) / unet_step
    groups = r["group_ids"]
    gids = np.unique(groups)
    N = int((groups == gids[0]).sum())

    sel_step, sel_final = r[f"step_{args.select_by}"], r[f"final_{args.select_by}"]
    out_final = r[f"final_{args.metric}"]

    rows = []
    # baselines: best-of-n with full trajectories, selected by final reward
    for n in range(1, N + 1):
        q = []
        for g in gids:
            idx = np.where(groups == g)[0]
            # expected metric of the sample with the best final selection score in a random n-subset
            q.append(_expected_metric_of_best(sel_final[idx], out_final[idx], n))
        cost = n * T + (n > 1) * n * s
        rows.append(("best-of-%d (full)" % n, cost, float(np.mean(q))))

    # pruning: N candidates to step k, keep m by preview score, finish, pick best by final score
    for k, m in product(range(1, T), [1, 2, 4]):
        if m >= N:
            continue
        q = []
        for g in gids:
            idx = np.where(groups == g)[0]
            keep = idx[np.argsort(-sel_step[idx, k - 1], kind="stable")[:m]]
            best = keep[np.argmax(sel_final[keep])]
            q.append(out_final[best])
        cost = N * k + m * (T - k) + N * s + (m > 1) * m * s
        rows.append((f"prune@{k} keep {m}", cost, float(np.mean(q))))

    rows.sort(key=lambda x: x[1])
    full_bestN = next(q for name, c, q in rows if name == f"best-of-{N} (full)")
    plain = next(q for name, c, q in rows if name == "best-of-1 (full)")
    print(f"{args.run_dir}: {len(gids)} prompts x {N} candidates, T={T} steps, scoring cost s={s:.2f} UNet steps, "
          f"select by {args.select_by}, report final {args.metric}")
    print(f"plain sampling {args.metric}={plain:.4f} (cost {T}); full best-of-{N} {args.metric}={full_bestN:.4f}\n")
    print(f"{'strategy':22s} {'cost':>7} {args.metric:>8} {'gain_vs_plain':>13} {'pareto':>7}")
    best_so_far = -1e9
    for name, c, q in rows:
        pareto = q > best_so_far + 1e-12
        best_so_far = max(best_so_far, q)
        print(f"{name:22s} {c:>7.1f} {q:>8.4f} {q - plain:>+13.4f} {'*' if pareto else '':>7}")


def _expected_metric_of_best(sel, out, n):
    """E[out of the argmax-by-sel element] over uniformly random n-subsets (exact, ties broken by order)."""
    order = np.argsort(sel, kind="stable")
    v = out[order]
    N = len(v)
    w = np.array([comb(i, n - 1) for i in range(N)], dtype=float) / comb(N, n)
    return float((w * v).sum())


if __name__ == "__main__":
    main()
