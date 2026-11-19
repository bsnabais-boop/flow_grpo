"""Paired comparison of the fixed-seed evals inside one training log (train_sd15.py with eval_seed set).

Every eval uses the same prompts, seeds and SDE windows, so eval k vs eval 0 differs only by the trained LoRA.
Reports per eval: mean OCR, exact-match rate, and the paired difference vs eval 0 with its standard error
(computed over prompts, each prompt averaged over its eval_images_per_prompt images).

Usage: python scripts/eval_paired.py logs/slurm_<jobid>.out [--images-per-prompt 4] [--test-batch-size 16]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_progress import DEBUG_RE, ocr_score  # noqa: E402


def eval_segments(path):
    """Return a list of evals, each a list of (target, read) in generation order."""
    with open(path, errors="replace") as f:
        lines = f.read().replace("\r", "\n").split("\n")
    evals, cur, buf = [], [], []
    for line in lines:
        m = DEBUG_RE.search(line)
        if m:
            buf.append(m.groups())
        elif "AVALIA" in line:  # end of one eval batch
            cur += buf
            buf = []
        elif "PONTUA" in line:  # end of a training epoch's sampling
            if cur:
                evals.append(cur)
                cur = []
            buf = []
    if cur:
        evals.append(cur)
    return evals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--images-per-prompt", type=int, default=4)
    p.add_argument("--test-batch-size", type=int, default=16)
    p.add_argument("--eval-freq", type=int, default=10, help="Only used to label evals with their epoch")
    args = p.parse_args()

    evals = eval_segments(args.log)
    if not evals:
        print("no evals found")
        return
    base = evals[0]
    k, bs = args.images_per_prompt, args.test_batch_size
    n = len(base)
    idx = np.arange(n)
    # eval images per batch are ordered [repeat 0: bs prompts, repeat 1: bs prompts, ...]
    prompt_id = (idx // (k * bs)) * bs + (idx % bs)
    s0 = np.array([ocr_score(*g)[0] for g in base])

    print(f"{'epoch':>5} {'n':>4} {'ocr':>6} {'exact':>6} {'diff_vs_ep0':>11} {'SE':>6} {'z':>5} {'prompts +/-/=':>14}")
    for i, e in enumerate(evals):
        sc = [ocr_score(*g) for g in e]
        s = np.array([x[0] for x in sc])
        exact = np.mean([x[1] for x in sc])
        line = f"{i * args.eval_freq:>5} {len(s):>4} {s.mean():>6.3f} {exact:>6.3f}"
        if i > 0 and len(e) == n and all(a[0] == b[0] for a, b in zip(base, e)):
            d = s - s0
            dp = np.array([d[prompt_id == q].mean() for q in np.unique(prompt_id)])
            se = dp.std(ddof=1) / np.sqrt(len(dp))
            line += (f" {d.mean():>+11.4f} {se:>6.4f} {d.mean() / se:>+5.1f} "
                     f"{(dp > 0).sum():>4}/{(dp < 0).sum()}/{(dp == 0).sum()}")
        elif i > 0:
            line += "  (incomplete or not paired with eval 0)"
        print(line)


if __name__ == "__main__":
    main()
