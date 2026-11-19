"""Phase 2 (offline): calibrated early pruning of trajectories with the noise-aware PRM (Conformal Risk Control).

Setting: G trajectories per prompt (a GRPO group / best-of-G). At step k the PRM scores every trajectory; those far
below the group leader are stopped, the rest finish. Score used: gap_i = s_i(k) - max_j s_j(k) (<= 0), i.e. relative to
the group, which removes the per-prompt offset of the score (a simple drift / difficulty correction).
Keep trajectory i iff gap_i >= -lambda (the step leader is always kept).

Risk (what we must not lose), per group, bounded in [0,1]:
  best   1 if the trajectory with the best FINAL score is pruned
  top    fraction of the top-half (by final score) trajectories that are pruned
Conformal Risk Control (Angelopoulos et al.): with n calibration groups, pick the smallest lambda such that
  n/(n+1) * mean_calib_loss(lambda) + 1/(n+1) <= alpha,
which guarantees E[loss on a new group] <= alpha. Loss is non-increasing in lambda, as CRC requires.

Reported per step k and method (averaged over random calibration/test splits of the prompts):
  test risk (should be <= alpha), fraction of trajectories pruned, UNet compute saved (pruned x remaining steps,
  minus the scorer's cost), and regret = final score of the best trajectory minus that of the best kept one.

Multi-point pruning (--multi 3,5,7): prune at several steps. At checkpoint k a live trajectory stops if
  (s_i(k) - max over live j of s_j(k)) / sigma_k < -lambda,
with sigma_k the median within-group std of the score at step k on the calibration groups (puts all steps on one
scale, a simple form of the slides' drift correction). One lambda for all checkpoints, calibrated with CRC.
--group-size g uses only the first g images of each prompt (e.g. compare G=4 and G=8 on the same run).

Usage: python scripts/phase2_conformal_pruning.py <phase1_noisyclip run dir> [--alpha 0.1] [--risk best]
       [--methods trained taesd dec] [--cost trained=0.16 taesd=0.31 dec=1.78]
"""
import argparse
import json
import os

import numpy as np


def group_view(x, groups):
    """(N, ...) -> (P, G, ...) assuming equal group sizes, ordered by group id."""
    order = np.argsort(groups, kind="stable")
    P = len(np.unique(groups))
    return x[order].reshape(P, -1, *x.shape[1:])


def losses(gap, final, lam, risk):
    """gap, final: (P, G). Returns per-group loss and pruned mask for threshold lam."""
    keep = gap >= -lam
    if risk == "best":
        best = final.argmax(1)
        loss = (~keep[np.arange(len(final)), best]).astype(float)
    else:
        top = final >= np.median(final, axis=1, keepdims=True)
        loss = ((~keep) & top).sum(1) / top.sum(1)
    return loss, ~keep


def crc_lambda(gap, final, alpha, risk, grid):
    n = len(gap)
    for lam in grid:  # ascending: most pruning first
        if n / (n + 1) * losses(gap, final, lam, risk)[0].mean() + 1 / (n + 1) <= alpha:
            return lam
    return np.inf


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--alpha", type=float, nargs="+", default=[0.1, 0.2])
    p.add_argument("--risk", choices=["best", "top", "regret"], default="best")
    p.add_argument("--delta", type=float, default=0.005, help="risk 'regret': loss = 1 if the best kept trajectory's "
                   "final score is more than delta below the group's best (multi-point mode only)")
    p.add_argument("--methods", nargs="+", default=["trained", "taesd", "dec", "proj_x0"])
    p.add_argument("--cost", nargs="+", default=["trained=0.16", "taesd=0.31", "dec=1.78", "proj_x0=0.11"],
                   help="scorer cost per call in UNet-step units (from phase1_noisyclip summary)")
    p.add_argument("--splits", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--multi", nargs="+", default=[], help="checkpoint sets, e.g. 4 3,5 3,5,7 (1-indexed steps)")
    p.add_argument("--group-size", type=int, default=0)
    args = p.parse_args()
    cost = {k: float(v) for k, v in (c.split("=") for c in args.cost)}

    d = np.load(os.path.join(args.run, "scores.npz"))
    groups = d["groups"]
    final = group_view(d["final_own"], groups)                     # (P, G)
    g = args.group_size or final.shape[1]
    final = final[:, :g]
    P, G = final.shape
    if args.multi:
        return run_multi(args, d, groups, final, cost, g)
    rng = np.random.default_rng(args.seed)
    splits = [rng.permutation(P) for _ in range(args.splits)]
    out = {"run": args.run, "risk": args.risk, "P": P, "G": G, "rows": []}
    print(f"{args.run}: {P} prompts x G={G}; risk='{args.risk}' "
          f"({'best final trajectory pruned' if args.risk == 'best' else 'fraction of top-half pruned'})")
    for alpha in args.alpha:
        print(f"\nalpha = {alpha}  (CRC guarantee: expected risk <= alpha)")
        print(f"{'method':>8} {'step':>5} {'frac':>5} | {'lambda':>7} {'test risk':>9} {'pruned':>7} "
              f"{'saved':>6} {'regret':>7}")
        for m in args.methods:
            if f"{m}_own" not in d:
                continue
            sc = group_view(d[f"{m}_own"], groups)[:, :g]           # (P, G, S)
            S = sc.shape[-1]
            for k in range(S):
                if np.isnan(sc[..., k]).all():
                    continue
                gap = sc[..., k] - sc[..., k].max(1, keepdims=True)
                grid = np.unique(np.concatenate([[0.0], np.sort(-gap.ravel())]))
                res = []
                for perm in splits:
                    cal, te = perm[:P // 2], perm[P // 2:]
                    lam = crc_lambda(gap[cal], final[cal], alpha, args.risk, grid)
                    loss, pruned = losses(gap[te], final[te], lam, args.risk)
                    keep_best = np.where(pruned, -np.inf, final[te]).max(1)
                    frac_pruned = pruned.mean()
                    # compute: each pruned trajectory skips the remaining S-k-1 steps; every trajectory pays one
                    # scorer call at step k
                    saved = (frac_pruned * (S - k - 1) - cost.get(m, 0.0)) / S
                    res.append((lam, loss.mean(), frac_pruned, saved, (final[te].max(1) - keep_best).mean()))
                lam, r, pr, sv, rg = np.array(res).mean(0)
                row = {"alpha": alpha, "method": m, "step": k + 1, "frac": (k + 1) / S, "lambda": lam,
                       "test_risk": r, "pruned": pr, "saved": sv, "regret": rg}
                out["rows"].append(row)
                if S <= 12 or (k + 1) % 5 == 0:
                    print(f"{m:>8} {k + 1:>5} {(k + 1) / S:>5.2f} | {lam:>7.3f} {r:>9.3f} {pr:>7.2f} {sv:>6.2f} "
                          f"{rg:>7.4f}")
        best = max((r for r in out["rows"] if r["alpha"] == alpha), key=lambda r: r["saved"])
        print(f"best: {best['method']} at step {best['step']} saves {best['saved']:.0%} of UNet compute "
              f"(pruned {best['pruned']:.0%}, test risk {best['test_risk']:.3f})")
    with open(os.path.join(args.run, f"pruning_{args.risk}.json"), "w") as f:
        json.dump(out, f, indent=1)


def multi_prune(sc, final, ks, lam, sigma, risk):
    """sc: (P,G,S) scores; ks: 0-indexed checkpoints. Returns per-group loss, pruned mask, steps run per trajectory,
    scorer calls per trajectory."""
    P, G, S = sc.shape
    alive = np.ones((P, G), bool)
    steps_run = np.full((P, G), S, float)
    calls = np.zeros((P, G))
    for k in ks:
        calls += alive
        s = np.where(alive, sc[..., k], -np.inf)
        gap = (s - s.max(1, keepdims=True)) / sigma[k]
        stop = alive & (gap < -lam)
        steps_run[stop] = k + 1
        alive &= ~stop
    pruned = ~alive
    if risk == "best":
        loss = pruned[np.arange(P), final.argmax(1)].astype(float)
    elif risk.startswith("regret"):
        kept = np.where(alive, final, -np.inf).max(1)
        loss = (final.max(1) - kept > float(risk.split(":")[1])).astype(float)
    else:
        top = final >= np.median(final, axis=1, keepdims=True)
        loss = (pruned & top).sum(1) / top.sum(1)
    return loss, pruned, steps_run, calls


def run_multi(args, d, groups, final, cost, g):
    P, G = final.shape
    rk = f"regret:{args.delta}" if args.risk == "regret" else args.risk
    sd = final.std(1)
    print(f"final CLIP score within a group: std {sd.mean():.4f}, best - median {np.mean(final.max(1) - np.median(final, 1)):.4f}")
    rng = np.random.default_rng(args.seed)
    splits = [rng.permutation(P) for _ in range(args.splits)]
    grid = np.linspace(0, 8, 161)
    out = {"run": args.run, "risk": args.risk, "G": G, "rows": []}
    print(f"{args.run}: {P} prompts x G={G}; multi-point pruning; risk='{args.risk}'")
    for alpha in args.alpha:
        print(f"\nalpha = {alpha}")
        print(f"{'method':>8} {'checkpoints':>12} | {'lambda':>6} {'test risk':>9} {'pruned':>7} {'saved':>6} {'regret':>7}")
        for m in args.methods:
            if f"{m}_own" not in d:
                continue
            sc = group_view(d[f"{m}_own"], groups)[:, :g]
            S = sc.shape[-1]
            for spec in args.multi:
                ks = [int(x) - 1 for x in spec.split(",")]
                if any(np.isnan(sc[..., k]).all() for k in ks):
                    continue
                res = []
                for perm in splits:
                    cal, te = perm[:P // 2], perm[P // 2:]
                    sigma = {k: max(np.median(sc[cal, :, k].std(1)), 1e-6) for k in ks}
                    lam = np.inf
                    n = len(cal)
                    for l in grid:
                        if n / (n + 1) * multi_prune(sc[cal], final[cal], ks, l, sigma, rk)[0].mean() \
                                + 1 / (n + 1) <= alpha:
                            lam = l
                            break
                    loss, pruned, steps_run, calls = multi_prune(sc[te], final[te], ks, lam, sigma, rk)
                    saved = 1 - (steps_run.mean() + calls.mean() * cost.get(m, 0.0)) / S
                    kept_best = np.where(pruned, -np.inf, final[te]).max(1)
                    res.append((lam, loss.mean(), pruned.mean(), saved, (final[te].max(1) - kept_best).mean()))
                lam, r, pr, sv, rg = np.array(res).mean(0)
                out["rows"].append({"alpha": alpha, "method": m, "checkpoints": spec, "lambda": lam,
                                    "test_risk": r, "pruned": pr, "saved": sv, "regret": rg})
                print(f"{m:>8} {spec:>12} | {lam:>6.2f} {r:>9.3f} {pr:>7.2f} {sv:>6.2f} {rg:>7.4f}")
        rows = [r for r in out["rows"] if r["alpha"] == alpha]
        if rows:
            b = max(rows, key=lambda r: r["saved"])
            print(f"best: {b['method']} at steps {b['checkpoints']} saves {b['saved']:.0%} of UNet compute "
                  f"(pruned {b['pruned']:.0%}, test risk {b['test_risk']:.3f})")
    tag = f"regret{args.delta}" if args.risk == "regret" else args.risk
    with open(os.path.join(args.run, f"pruning_multi_{tag}_G{G}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
