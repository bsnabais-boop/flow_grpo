"""Paired comparison of phase3_eval.py runs (same prompts and seeds) against the first run (the reference).

For every metric: mean of each run, difference to the reference and a 95% bootstrap CI over prompts (paired: the
same prompt/seed images are compared). A difference is called significant when the CI excludes 0.

Usage: python scripts/phase3_compare.py logs/phase3_eval/base_10steps logs/phase3_eval/prm_ckpt-XX ...
"""
import argparse
import json
import os

import numpy as np


def boot_ci(per_prompt, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = [rng.choice(per_prompt, len(per_prompt)).mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def per_prompt(x, pid):
    return np.array([np.nanmean(x[pid == p]) for p in np.unique(pid)])


def metrics(d, name):
    m = {"CLIP": d["clip"], "PickScore": d["pick"]}
    if name == "ocr":
        m["OCR"] = d["ocr"]
        m["exact"] = (d["ocr"] >= 1.0).astype(float)
    else:
        ok = (d["clip"][:, None] > d["neg"]).astype(float)
        ok[np.isnan(d["neg"])] = np.nan
        m["neg_acc"] = np.nanmean(ok, 1)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    args = p.parse_args()
    out = {}
    for name in ["align", "ocr"]:
        paths = [os.path.join(r, f"{name}.npz") for r in args.runs]
        if not all(os.path.exists(x) for x in paths):
            continue
        ds = [np.load(x) for x in paths]
        ref = ds[0]
        print(f"\n=== {name} ({len(ref['clip'])} images) reference: {args.runs[0]}")
        ref_m = metrics(ref, name)
        for key in ref_m:
            print(f"  {key}")
            for r, d in zip(args.runs, ds):
                assert (d["prompt_ids"] == ref["prompt_ids"]).all()
                v = metrics(d, name)[key]
                line = f"    {os.path.basename(r.rstrip('/')):>36}: {np.nanmean(v):.4f}"
                if d is not ref:
                    diff = per_prompt(v - ref_m[key], ref["prompt_ids"])
                    diff = diff[~np.isnan(diff)]
                    lo, hi = boot_ci(diff)
                    sig = "*" if lo > 0 or hi < 0 else " "
                    line += f"  diff {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] {sig}"
                    out.setdefault(name, {}).setdefault(key, {})[r] = [float(diff.mean()), float(lo), float(hi)]
                print(line)
    with open(os.path.join(args.runs[-1], "compare_vs_ref.json"), "w") as f:
        json.dump({"ref": args.runs[0], **out}, f, indent=1)


if __name__ == "__main__":
    main()
