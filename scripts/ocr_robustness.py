"""Robustness check of the option-1 OCR gain (2026-11-15).

The training/eval OCR reward gives full credit when the target is a substring of the recognised text, and the trained model
often writes the word several times. This script regenerates the OCR test images (same per-image seeds as phase3_eval.py,
default seed 2024 x 8 images = the confirmation images) for several (model, steps) settings and scores them with:

  paddle         the training/eval reward (PaddleOCR det+rec, substring = full credit)      -> should reproduce phase3_confirm
  strict         1 - Levenshtein(all recognised text, target)/len, NO substring credit (extra text is penalised)
  strict_exact   all recognised text == target (nothing else written)
  line_exact     some single detected line == target (the word is written cleanly at least once)
  line_score     best single line, 1 - Levenshtein/len, no substring credit
  repeat         target occurs >= 2 times in the recognised text;  n_occ = number of occurrences
  trocr_exact    a second recogniser (TrOCR-large scene-text, greedy decoding) reads the target exactly on some PaddleOCR line
                 crop or on the whole image (detector shared, recogniser independent)
  trocr_score    best 1 - Levenshtein/len of the TrOCR readings
  easy_*         fully independent engine (EasyOCR, own detector + recogniser; --easyocr): easy_paddle_rule (substring credit),
                 easy_strict (no substring credit), easy_line_exact (some detected line == target), easy_repeat

All comparisons are paired with the first setting (same seeds); 95% bootstrap CIs over prompts.
Usage: python scripts/ocr_robustness.py --out logs/ocr_robustness/v1 \
          --setting base_10:none:10 base_50:none:50 long500_10:<lora>:10 ...
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--setting", nargs="+", required=True, help="name:lora_or_none:num_steps")
    p.add_argument("--prompts", default="dataset/ocr_simple/test.txt")
    p.add_argument("--images-per-prompt", type=int, default=8)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--trocr", default="microsoft/trocr-large-str")
    p.add_argument("--easyocr", default=None, help="EasyOCR model dir (enables the easy_* metrics)")
    p.add_argument("--out", required=True)
    return p.parse_args()


def target_of(prompt):
    parts = prompt.split('"')
    return parts[1].strip() if len(parts) > 2 else ""


def norm(s):
    return s.replace(" ", "").lower()


def score_no_substring(rec, tgt):
    from Levenshtein import distance
    if not tgt:
        return 0.0
    return 1 - min(distance(rec, tgt), len(tgt)) / len(tgt)


def score_paddle(rec, tgt):
    if not tgt:
        return 0.0
    return 1.0 if tgt in rec else score_no_substring(rec, tgt)


def count_occ(rec, tgt):
    if not tgt:
        return 0
    n, i = 0, rec.find(tgt)
    while i >= 0:
        n += 1
        i = rec.find(tgt, i + len(tgt))
    return n


def line_box(line, H, W, pad=0.15):
    pts = np.array(line[0], dtype=float)
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    px, py = pad * (x1 - x0) + 2, pad * (y1 - y0) + 2
    x0, x1 = int(max(0, x0 - px)), int(min(W, x1 + px))
    y0, y1 = int(max(0, y0 - py)), int(min(H, y1 + py))
    return None if (x1 - x0 < 4 or y1 - y0 < 4) else (x0, y0, x1, y1)


def boot_ci(diff_per_prompt, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = len(diff_per_prompt)
    bs = diff_per_prompt[rng.integers(0, m, (n, m))].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    from flow_grpo.ocr import OcrScorer

    prompts = [l.strip() for l in open(args.prompts) if l.strip()]
    K = args.images_per_prompt
    flat = [(pid, p, r) for pid, p in enumerate(prompts) for r in range(K)]
    pid_arr = np.array([x[0] for x in flat])
    # identical to phase3_eval.py for the "ocr" set
    seeds = [args.seed * 100000 + 50000 + pid * 16 + r for pid, _, r in flat]
    targets = [target_of(p) for _, p, _ in flat]

    ocr = OcrScorer().ocr
    proc = TrOCRProcessor.from_pretrained(args.trocr)
    trocr = VisionEncoderDecoderModel.from_pretrained(args.trocr).to(device).eval()
    trocr.config.decoder_start_token_id = trocr.generation_config.decoder_start_token_id
    trocr.config.pad_token_id = trocr.generation_config.pad_token_id
    size = proc.image_processor.size
    S = size["height"] if isinstance(size, dict) else size
    t_mean = torch.tensor(proc.image_processor.image_mean).view(1, 3, 1, 1)
    t_std = torch.tensor(proc.image_processor.image_std).view(1, 3, 1, 1)

    reader = None
    if args.easyocr:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=True, model_storage_directory=args.easyocr, download_enabled=False,
                                verbose=False)

    results = {}
    for setting in args.setting:
        name, lora, steps = setting.split(":")
        steps = int(steps)
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
                                                       safety_checker=None, requires_safety_checker=False)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        if lora != "none":
            from peft import PeftModel
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora).merge_and_unload()
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        M = {k: np.zeros(len(flat), np.float32) for k in
             ["paddle", "strict", "strict_exact", "line_exact", "line_score", "repeat", "n_occ", "n_lines",
              "trocr_exact", "trocr_score", "easy_paddle_rule", "easy_strict", "easy_line_exact", "easy_repeat"]}
        recs, trecs, grid = [], [], []
        for b0 in range(0, len(flat), args.batch_size):
            idx = list(range(b0, min(b0 + args.batch_size, len(flat))))
            gens = [torch.Generator(device=device).manual_seed(int(seeds[i])) for i in idx]
            with torch.no_grad():
                imgs = pipe([flat[i][1] for i in idx], num_inference_steps=steps, guidance_scale=7.5, eta=0.0,
                            generator=gens, output_type="pt").images.float().clamp(0, 1)
            for j, i in enumerate(idx):
                tgt = norm(targets[i])
                arr = (imgs[j].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                res = ocr.ocr(arr, cls=False)
                lines = [l for l in (res[0] if res and res[0] else []) if l and len(l) > 1 and l[1] and l[1][1] > 0]
                line_txt = [norm(l[1][0]) for l in lines]
                rec = "".join(line_txt)
                recs.append(rec)
                M["paddle"][i] = score_paddle(rec, tgt)
                M["strict"][i] = score_no_substring(rec, tgt)
                M["strict_exact"][i] = float(rec == tgt and tgt != "")
                M["line_exact"][i] = float(any(t == tgt for t in line_txt) and tgt != "")
                M["line_score"][i] = max([score_no_substring(t, tgt) for t in line_txt], default=0.0)
                M["n_occ"][i] = count_occ(rec, tgt)
                M["repeat"][i] = float(M["n_occ"][i] >= 2)
                M["n_lines"][i] = len(lines)
                # second recogniser on each PaddleOCR line crop + the whole image (greedy decoding)
                x = imgs[j:j + 1]
                crops = [x[:, :, b[1]:b[3], b[0]:b[2]] for b in
                         (line_box(l, arr.shape[0], arr.shape[1]) for l in lines) if b is not None] + [x]
                pix = torch.cat([F.interpolate(c.cpu(), size=(S, S), mode="bilinear", align_corners=False) for c in crops])
                pix = (pix - t_mean) / t_std
                with torch.no_grad():
                    out = trocr.generate(pix.to(device), max_new_tokens=24)
                reads = [norm(t) for t in proc.batch_decode(out, skip_special_tokens=True)]
                trecs.append("|".join(reads))
                if reader is not None:
                    elines = [norm(t) for _, t, c in reader.readtext(arr) if c > 0]
                    erec = "".join(elines)
                    M["easy_paddle_rule"][i] = score_paddle(erec, tgt)
                    M["easy_strict"][i] = score_no_substring(erec, tgt)
                    M["easy_line_exact"][i] = float(any(t == tgt for t in elines) and tgt != "")
                    M["easy_repeat"][i] = float(count_occ(erec, tgt) >= 2)
                M["trocr_exact"][i] = float(any(t == tgt for t in reads) and tgt != "")
                M["trocr_score"][i] = max([score_no_substring(t, tgt) for t in reads], default=0.0)
                if flat[i][2] == 0 and len(grid) < 16:
                    grid.append(Image.fromarray(arr).resize((256, 256)))
            print(f"[{name} {idx[-1] + 1}/{len(flat)}] paddle {M['paddle'][:idx[-1] + 1].mean():.3f} "
                  f"strict {M['strict'][:idx[-1] + 1].mean():.3f} trocr_exact {M['trocr_exact'][:idx[-1] + 1].mean():.3f}",
                  flush=True)
        del pipe; torch.cuda.empty_cache()
        np.savez(os.path.join(args.out, f"{name}.npz"), prompt_ids=pid_arr, rec=np.array(recs), trocr=np.array(trecs),
                 targets=np.array(targets), **M)
        canvas = Image.new("RGB", (256 * 8, 256 * 2), "white")
        for k, im in enumerate(grid):
            canvas.paste(im, (256 * (k % 8), 256 * (k // 8)))
        canvas.save(os.path.join(args.out, f"grid_{name}.png"))
        results[name] = M

    # paired comparison with the first setting
    ref = args.setting[0].split(":")[0]
    summary = {"settings": args.setting, "reference": ref, "N": len(flat), "metrics": {}}
    for name, M in results.items():
        summary["metrics"][name] = {}
        for k, v in M.items():
            d = {"mean": float(v.mean())}
            if name != ref:
                per_prompt = np.array([(v - results[ref][k])[pid_arr == p].mean() for p in np.unique(pid_arr)])
                d["diff"] = float(per_prompt.mean())
                d["ci95"] = boot_ci(per_prompt)
            summary["metrics"][name][k] = d
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    keys = ["paddle", "strict", "strict_exact", "line_exact", "line_score", "repeat", "n_occ", "trocr_exact", "trocr_score"]
    if reader is not None:
        keys += ["easy_paddle_rule", "easy_strict", "easy_line_exact", "easy_repeat"]
    print("setting".ljust(16) + "".join(k.rjust(13) for k in keys))
    for name in results:
        row = summary["metrics"][name]
        print(name.ljust(16) + "".join(f"{row[k]['mean']:13.3f}" for k in keys))
        if name != ref:
            print("  diff".ljust(16) + "".join(
                (f"{row[k]['diff']:+.3f}" + ("*" if row[k]["ci95"][0] > 0 or row[k]["ci95"][1] < 0 else " ")).rjust(13)
                for k in keys))


if __name__ == "__main__":
    main()
