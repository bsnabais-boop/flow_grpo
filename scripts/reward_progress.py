"""Summarise reward progress from a train_sd15.py / train_sd3_fast.py slurm stdout log.

Usage: python scripts/reward_progress.py logs/slurm_<jobid>.out [--every N] [--last K]

Per training epoch it reports:
  total   mean final-step reward (the weighted sum, e.g. 80*ocr + clipscore) from the "PONTUAÇÕES" block
  ocr     mean OCR score (1 - edit_distance/len, same formula as flow_grpo/ocr.py) over the [DEBUG OCR] lines
  exact   fraction of OCR reads that contain the target text exactly
Eval segments (lines before an "AVALIAÇÃO" marker) are reported separately as EVAL rows.
"""
import argparse
import re

from Levenshtein import distance

DEBUG_RE = re.compile(r"\[DEBUG OCR\] Alvo: '(.*)' \| Lido pelo OCR: '(.*)'$")
NUM_RE = re.compile(r"-?\d+\.\d+(?:e[-+]?\d+)?")


def ocr_score(target, read):
    t = target.replace(" ", "").lower()
    r = read.replace(" ", "").lower()
    if not t:
        return None, False
    if t in r:
        return 1.0, True
    d = min(distance(r, t), len(t))
    return 1 - d / len(t), False


def parse(path):
    with open(path, errors="replace") as f:
        lines = f.read().replace("\r", "\n").split("\n")

    rows, scores, exact, eval_scores = [], [], [], []
    in_block, block_vals, epoch = False, [], 0
    for line in lines:
        m = DEBUG_RE.search(line)
        if m:
            s, e = ocr_score(*m.groups())
            if s is not None:
                scores.append(s)
                exact.append(e)
            continue
        if line.startswith("AVALIAÇÃO"):
            # everything accumulated since the last marker was eval
            eval_scores.extend(scores)
            scores, exact = [], []
            continue
        if eval_scores and not line.startswith("AVALIAÇÃO") and ("PONTUAÇÕES" in line):
            rows.append(("EVAL", None, sum(eval_scores) / len(eval_scores), None, len(eval_scores)))
            eval_scores = []
        if "PONTUAÇÕES" in line:
            in_block, block_vals = True, []
            continue
        if in_block:
            if line.startswith("====="):
                in_block = False
                # each tensor row is [step_1, ..., step_k]; take the last step
                finals = []
                for row in "\n".join(block_vals).split("],"):
                    nums = NUM_RE.findall(row)
                    if nums:
                        finals.append(float(nums[-1]))
                total = sum(finals) / len(finals) if finals else float("nan")
                ocr = sum(scores) / len(scores) if scores else float("nan")
                ex = sum(exact) / len(exact) if exact else float("nan")
                rows.append((epoch, total, ocr, ex, len(scores)))
                epoch += 1
                scores, exact = [], []
            else:
                block_vals.append(line)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--every", type=int, default=1, help="average training epochs in windows of N")
    ap.add_argument("--last", type=int, default=0, help="only print the last K rows")
    args = ap.parse_args()

    rows = parse(args.log)
    out, buf = [], []
    for r in rows:
        if r[0] == "EVAL":
            out.append(f"EVAL          ocr={r[2]:.3f}  (n={r[4]})")
            continue
        buf.append(r)
        if len(buf) == args.every:
            e0, e1 = buf[0][0], buf[-1][0]
            tot = sum(b[1] for b in buf) / len(buf)
            ocr = sum(b[2] for b in buf) / len(buf)
            ex = sum(b[3] for b in buf) / len(buf)
            label = f"ep {e0}" if e0 == e1 else f"ep {e0}-{e1}"
            out.append(f"{label:<12}  total={tot:7.3f}  ocr={ocr:.3f}  exact={ex:.2f}")
            buf = []
    if args.last:
        out = out[-args.last:]
    print("\n".join(out) if out else "no completed epochs yet")


if __name__ == "__main__":
    main()
