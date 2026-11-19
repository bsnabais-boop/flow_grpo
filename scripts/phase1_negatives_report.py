"""Phase 1 analysis: subtle-negative detection per category and per step (needs a phase1_noisyclip.py run with
--negatives). For each method and step it reports, per category (count/color/background/subject/relation):

  acc     fraction of (image, negative) pairs where the image scores higher with its own prompt than with the negative
  agree   fraction where the method's verdict (own > negative) equals CLIP's verdict on the FINAL image, i.e. how
          early the PRM predicts what the outcome reward will say. A CLIP-distilled scorer cannot exceed CLIP's own
          final accuracy (row "final"), only reach it earlier.

Usage: python scripts/phase1_negatives_report.py <run dir> [<run dir 2> ...] [--fracs 0.2 0.4 0.6 1.0]
"""
import argparse
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--methods", nargs="+", default=["trained", "taesd", "dec", "proj_x0"])
    args = p.parse_args()
    for run in args.runs:
        d = np.load(os.path.join(run, "scores.npz"))
        cats = list(d["neg_cats"])
        final_own, final_neg = d["final_own"], d["final_neg"]                 # (N,), (N,C)
        valid = ~np.isnan(final_neg)
        fin_ok = final_own[:, None] > final_neg                               # CLIP verdict on the final image
        print(f"\n{run}: {len(final_own)} images; negatives per category "
              f"{dict(zip(cats, valid.sum(0).tolist()))}")
        hdr = f"{'method':>8} {'frac':>5} | " + " ".join(f"{c[:9]:>9}" for c in cats) + " |       all"
        print("accuracy own > negative   (agreement with the final CLIP verdict in brackets)")
        print(hdr); print("-" * len(hdr))
        row = [f"{fin_ok[valid[:, c], c].mean():9.2f}" for c in range(len(cats))]
        print(f"{'final':>8} {'':>5} | " + " ".join(row) + f" | {fin_ok[valid].mean():9.2f}")
        report = {"final": {c: float(fin_ok[valid[:, k], k].mean()) for k, c in enumerate(cats)}, "methods": {}}
        for m in args.methods:
            if f"{m}_neg" not in d:
                continue
            own, neg = d[f"{m}_own"], d[f"{m}_neg"]                             # (N,S), (N,S,C)
            S = own.shape[1]
            report["methods"][m] = []
            for fr in args.fracs:
                i = min(S - 1, int(round(fr * S)) - 1)
                while i > 0 and np.isnan(own[:, i]).all():
                    i -= 1
                ok = own[:, i, None] > neg[:, i]
                cells, rec = [], {"step": i + 1, "frac": (i + 1) / S}
                for c, cat in enumerate(cats):
                    v = valid[:, c]
                    acc, agr = ok[v, c].mean(), (ok[v, c] == fin_ok[v, c]).mean()
                    cells.append(f"{acc:.2f}({agr:.2f})")
                    rec[cat] = {"acc": float(acc), "agree": float(agr)}
                acc, agr = ok[valid].mean(), (ok[valid] == fin_ok[valid]).mean()
                rec["all"] = {"acc": float(acc), "agree": float(agr)}
                report["methods"][m].append(rec)
                print(f"{m:>8} {(i + 1) / S:>5.2f} | " + " ".join(f"{x:>9}" for x in cells) + f" | {acc:.2f}({agr:.2f})")
        with open(os.path.join(run, "negatives_report.json"), "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
