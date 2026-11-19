"""Option 2 (2026-11-13), step 1: is there a differentiable text reward that agrees with the PaddleOCR reward?

RL on the PaddleOCR score carries ~1-2e-4 gradient signal per image (logs/diag_ocr_signal/v3_variants). The plan is to
back-propagate a differentiable text-recognition score through the denoising steps (DRaFT / AlignProp style). This
script checks candidate proxies on base SD1.5 images (10-step DDIM, eta 0, OCR test prompts) before any training:

  paddle         current reward: 1 - Levenshtein / len (PaddleOCR PP-OCRv3 det + rec, CPU)
  trocr-*-box    -NLL of the target text under TrOCR (scene-text 'str' models), on the crop given by the union of
                 the PaddleOCR detection boxes (crop coordinates are constants -> differentiable w.r.t. pixels)
  trocr-*-full   same on the whole image (no detector at all)
  clip           CLIP-L cos(image, 'a photo with the text "X"')

Reports Spearman with the PaddleOCR score (all images / within prompt), AUROC for exact matches and for OCR >= 0.5.
Output: <out>/proxy_check.json and per-image arrays in <out>/proxy_check.npz.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="dataset/ocr_simple/test.txt")
    p.add_argument("--num-prompts", type=int, default=128)
    p.add_argument("--images-per-prompt", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--trocr", default="microsoft/trocr-base-str,microsoft/trocr-large-str")
    p.add_argument("--out", default="logs/opt2/proxy_check")
    return p.parse_args()


def target_of(prompt):
    parts = prompt.split('"')
    return parts[1].strip() if len(parts) > 2 else ""


def paddle_score(rec_text, target):
    from Levenshtein import distance
    r = rec_text.replace(" ", "").lower()
    t = target.replace(" ", "").lower()
    if not t:
        return 0.0
    d = 0 if t in r else min(distance(r, t), len(t))
    return 1 - d / len(t)


def union_box(lines, H, W, pad=0.15):
    """Bounding rectangle of all detected boxes, padded; None if nothing was detected."""
    if not lines:
        return None
    pts = np.concatenate([np.array(l[0], dtype=float) for l in lines], 0)
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    px, py = pad * (x1 - x0) + 2, pad * (y1 - y0) + 2
    x0, x1 = int(max(0, x0 - px)), int(min(W, x1 + px))
    y0, y1 = int(max(0, y0 - py)), int(min(H, y1 + py))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1


def auroc(score, label):
    label = np.asarray(label, bool)
    if label.all() or (~label).all():
        return None
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = label.sum(); n0 = (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_prompt_spearman(x, y, pid):
    rs = []
    for p in np.unique(pid):
        m = pid == p
        if np.std(y[m]) > 0 and np.std(x[m]) > 0:
            rs.append(spearmanr(x[m], y[m]).correlation)
    return float(np.mean(rs)) if rs else None


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from transformers import CLIPModel, CLIPTokenizer, TrOCRProcessor, VisionEncoderDecoderModel
    from flow_grpo.ocr import OcrScorer

    prompts = [l.strip() for l in open(args.prompts) if l.strip()][:args.num_prompts]
    K = args.images_per_prompt

    # ---------------- 1. images ----------------
    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None,
                                                   requires_safety_checker=False, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    images, pid = [], []
    for i, p in enumerate(prompts):
        gens = [torch.Generator(device).manual_seed(args.seed * 1000 + i * K + k) for k in range(K)]
        out = pipe([p] * K, num_inference_steps=args.num_steps, guidance_scale=7.5, eta=0.0, generator=gens,
                   output_type="pt").images  # [K,3,512,512] in [0,1]
        images.append(out.float().cpu()); pid += [i] * K
    images = torch.cat(images); pid = np.array(pid)
    del pipe; torch.cuda.empty_cache()
    print(f"sampled {len(images)} images in {time.time() - t0:.0f}s", flush=True)
    targets = [target_of(prompts[i]) for i in pid]

    # ---------------- 2. PaddleOCR: score + boxes ----------------
    t0 = time.time()
    ocr = OcrScorer().ocr
    paddle, boxes, rec, line_boxes = [], [], [], []
    for img, tgt in zip(images, targets):
        arr = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        res = ocr.ocr(arr, cls=False)
        lines = res[0] if res and res[0] else []
        txt = "".join(l[1][0] for l in lines if l and len(l) > 1 and l[1] and l[1][1] > 0)
        rec.append(txt); paddle.append(paddle_score(txt, tgt))
        boxes.append(union_box(lines, arr.shape[0], arr.shape[1]))
        line_boxes.append([b for b in (union_box([l], arr.shape[0], arr.shape[1]) for l in lines) if b is not None])
    paddle = np.array(paddle)
    t_paddle = (time.time() - t0) / len(images)
    print(f"paddle done ({t_paddle:.2f}s/img): mean {paddle.mean():.3f}, exact {np.mean(paddle >= 1):.3f}, "
          f"detected {np.mean([b is not None for b in boxes]):.2f}", flush=True)

    scores = {"paddle": paddle}

    # ---------------- 3. TrOCR NLL of the target (crop / full) ----------------
    for name in args.trocr.split(","):
        proc = TrOCRProcessor.from_pretrained(name)
        model = VisionEncoderDecoderModel.from_pretrained(name).to(device).eval()
        # teacher forcing needs the start/pad tokens in the model config (generation config: start 2 = </s>, pad 1)
        model.config.decoder_start_token_id = model.generation_config.decoder_start_token_id
        model.config.pad_token_id = model.generation_config.pad_token_id
        size = proc.image_processor.size
        S = size["height"] if isinstance(size, dict) else size
        mean = torch.tensor(proc.image_processor.image_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(proc.image_processor.image_std, device=device).view(1, 3, 1, 1)
        short = name.split("/")[-1]
        t0 = time.time()
        for mode in ["box", "full", "line"]:
            vals = []
            for img, tgt, box, lbs in zip(images, targets, boxes, line_boxes):
                x = img.unsqueeze(0).to(device)
                if mode == "line":
                    # each detected text line separately (+ the whole image), keep the best (lowest-NLL) one
                    crops = [x[:, :, b[1]:b[3], b[0]:b[2]] for b in lbs] + [x]
                    best = 1e9
                    for c in crops:
                        c = (F.interpolate(c, size=(S, S), mode="bilinear", align_corners=False) - mean) / std
                        for t in {tgt.upper(), tgt.upper().replace(" ", "")}:
                            lab = torch.tensor([proc.tokenizer(t).input_ids[1:]], device=device)
                            with torch.no_grad():
                                best = min(best, float(model(pixel_values=c, labels=lab).loss))
                    vals.append(-best); continue
                if mode == "box":
                    if box is None:
                        vals.append(np.nan); continue
                    x0, y0, x1, y1 = box
                    x = x[:, :, y0:y1, x0:x1]
                x = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)  # differentiable
                x = (x - mean) / std
                # the 'str' models read UPPERCASE without spaces; score the target with and without spaces, keep the best
                losses = []
                for t in {tgt.upper(), tgt.upper().replace(" ", "")}:
                    lab = torch.tensor([proc.tokenizer(t).input_ids[1:]], device=device)  # drop <s>
                    with torch.no_grad():
                        losses.append(float(model(pixel_values=x, labels=lab).loss))  # mean token NLL
                vals.append(-min(losses))
            scores[f"{short}-{mode}"] = np.array(vals)
        print(f"{short} done ({(time.time() - t0) / (3 * len(images)):.3f}s/img)", flush=True)
        del model; torch.cuda.empty_cache()

    # box variant with a fallback for images without a detection: the full-image value
    for name in args.trocr.split(","):
        short = name.split("/")[-1]
        b, f = scores[f"{short}-box"], scores[f"{short}-full"]
        scores[f"{short}-box|full"] = np.where(np.isnan(b), f, b)

    # ---------------- 4. CLIP text-match ----------------
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    cm = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    cs = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    vals = []
    with torch.no_grad():
        for s in range(0, len(images), 32):
            x = F.interpolate(images[s:s + 32].to(device), size=(224, 224), mode="bicubic", align_corners=False)
            ie = F.normalize(clip.get_image_features(pixel_values=(x - cm) / cs), dim=-1)
            tt = tok([f'a photo with the text "{t}"' for t in targets[s:s + 32]], padding=True, return_tensors="pt").to(device)
            te = F.normalize(clip.get_text_features(**tt), dim=-1)
            vals += (ie * te).sum(-1).tolist()
    scores["clip"] = np.array(vals)

    # combinations: z-scored TrOCR (best line) + z-scored CLIP
    z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    for name in args.trocr.split(","):
        short = name.split("/")[-1]
        scores[f"{short}-line+clip"] = z(scores[f"{short}-line"]) + z(scores["clip"])

    # ---------------- 5. agreement with PaddleOCR ----------------
    res = {"n_images": len(images), "paddle_mean": float(paddle.mean()), "paddle_exact": float(np.mean(paddle >= 1)),
           "detected_frac": float(np.mean([b is not None for b in boxes])), "paddle_sec_per_image": t_paddle, "proxies": {}}
    for k, v in scores.items():
        if k == "paddle":
            continue
        m = ~np.isnan(v)
        res["proxies"][k] = {
            "coverage": float(m.mean()),
            "spearman_all": float(spearmanr(v[m], paddle[m]).correlation),
            "spearman_within_prompt": within_prompt_spearman(v[m], paddle[m], pid[m]),
            "auroc_exact": auroc(v[m], paddle[m] >= 1),
            "auroc_ocr_ge_0.5": auroc(v[m], paddle[m] >= 0.5),
            "auroc_ocr_gt_0": auroc(v[m], paddle[m] > 0),
            "spearman_where_paddle_gt_0": float(spearmanr(v[m & (paddle > 0)], paddle[m & (paddle > 0)]).correlation),
        }
        print(k, res["proxies"][k], flush=True)
    with open(os.path.join(args.out, "proxy_check.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez(os.path.join(args.out, "proxy_check.npz"), pid=pid, rec=np.array(rec), targets=np.array(targets),
             **{k.replace("|", "_or_"): v for k, v in scores.items()})


if __name__ == "__main__":
    main()
