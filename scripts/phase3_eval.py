"""Phase 3 evaluation: prompt alignment AND text rendering of SD1.5 (base or a GRPO LoRA), fixed seeds -> paired.

Two held-out prompt sets (never used for training):
  align : the 128 GenAI-Bench eval prompts of the Phase 1 split (ids[32:160] after shuffle seed 0)
          metrics: CLIP ViT-L/14 score (= training reward of the ORM run), PickScore (independent judge, catches
          reward hacking of CLIP), subtle-negative accuracy (own prompt > LLM-edited negative, per category, CLIP-L)
  ocr   : dataset/ocr_simple/test.txt (128 prompts, words never seen in training)
          metrics: OCR (1 - normalised edit distance, PaddleOCR), exact match, CLIP score
Sampling: deterministic DDIM (eta=0) with --num-steps, CFG 7.5, the same initial noise for every model (per-image
seed), so differences between models are paired.

Usage: sbatch eval_phase3.slurm --lora logs/phase3/mixed_prm/checkpoints/checkpoint-XX/lora --out logs/phase3_eval/...
       python scripts/phase3_compare.py <base dir> <dir 2> ...
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1_noisyclip import Clip, load_genai_bench  # noqa: E402

CATS = ["count", "color", "background", "subject", "relation"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--lora", default=None)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--images-per-prompt", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--sets", nargs="+", default=["align", "ocr"])
    p.add_argument("--genai-bench", default="dataset/genai_bench")
    p.add_argument("--negatives", default="dataset/genai_bench/negatives.json")
    p.add_argument("--ocr-prompts", default="dataset/ocr_simple/test.txt")
    p.add_argument("--no-pickscore", action="store_true")
    p.add_argument("--grid-prompts", type=int, default=8)
    p.add_argument("--out", required=True)
    return p.parse_args()


def prompt_sets(args):
    sets = {}
    if "align" in args.sets:
        P, tags = load_genai_bench(args.genai_bench)
        ids = sorted(P)
        random.Random(0).shuffle(ids)
        eval_ids = ids[32:160]
        sets["align"] = [(i, " ".join(P[i].split()), ";".join(tags[i])) for i in eval_ids]
    if "ocr" in args.sets:
        lines = [l.strip() for l in open(args.ocr_prompts) if l.strip()]
        sets["ocr"] = [(k, l, "ocr") for k, l in enumerate(lines)]
    return sets


@torch.no_grad()
def generate(pipe, prompts, seeds, args, device):
    gens = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
    out = pipe(prompts, num_inference_steps=args.num_steps, guidance_scale=args.guidance_scale, eta=0.0,
               generator=gens, output_type="pt")
    return out.images.float().clamp(0, 1)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    device = torch.device("cuda")
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16, safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if args.lora:
        from peft import PeftModel
        pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora).merge_and_unload()
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    clip = Clip("openai/clip-vit-large-patch14", device)
    import flow_grpo.rewards
    ocr_fn = flow_grpo.rewards.ocr_score(device) if "ocr" in args.sets else None
    pick_fn = None if args.no_pickscore else flow_grpo.rewards.pickscore_score(device)
    negs = {int(k): v for k, v in json.load(open(args.negatives)).items()} if "align" in args.sets else {}

    summary = {}
    for name, items in prompt_sets(args).items():
        flat = [(pid, p, tag, r) for pid, p, tag in items for r in range(args.images_per_prompt)]
        N = len(flat)
        res = {"clip": np.zeros(N, np.float32), "pick": np.full(N, np.nan, np.float32)}
        if name == "align":
            res["neg"] = np.full((N, len(CATS)), np.nan, np.float32)
        else:
            res["ocr"] = np.zeros(N, np.float32)
        grid = []
        for b0 in range(0, N, args.batch_size):
            batch = flat[b0:b0 + args.batch_size]
            prompts = [x[1] for x in batch]
            # per-image seed: depends only on (set, prompt index, replica), identical for every model
            seeds = [args.seed * 100000 + (0 if name == "align" else 50000) + int(x[0]) * 16 + x[3] for x in batch]
            imgs = generate(pipe, prompts, seeds, args, device)
            te = clip.text(prompts)
            ie = clip.image(imgs)
            res["clip"][b0:b0 + len(batch)] = (ie * te).sum(-1).cpu().numpy()
            if pick_fn is not None:
                s, _ = pick_fn(imgs, prompts, None)
                res["pick"][b0:b0 + len(batch)] = torch.as_tensor(s).float().cpu().numpy()
            if name == "align":
                for c, cat in enumerate(CATS):
                    rows = [k for k, x in enumerate(batch) if negs.get(x[0], {}).get(cat)]
                    if rows:
                        tn = clip.text([negs[batch[k][0]][cat] for k in rows])
                        res["neg"][[b0 + k for k in rows], c] = (ie[rows] * tn).sum(-1).cpu().numpy()
            else:
                s, _ = ocr_fn(imgs, prompts, [{"target_text": p} for p in prompts])
                res["ocr"][b0:b0 + len(batch)] = np.array([float(v) for v in s])
            for k, x in enumerate(batch):
                if x[3] == 0 and len(grid) < args.grid_prompts:
                    grid.append(imgs[k].cpu())
            print(f"[{name} {b0 + len(batch)}/{N}] clip {res['clip'][:b0 + len(batch)].mean():.4f}" +
                  (f" ocr {res['ocr'][:b0 + len(batch)].mean():.3f}" if name == "ocr" else ""), flush=True)
        np.savez(os.path.join(args.out, f"{name}.npz"), prompt_ids=np.array([x[0] for x in flat]),
                 replica=np.array([x[3] for x in flat]), tags=np.array([x[2] for x in flat]), **res)
        s = {"N": N, "clip": float(res["clip"].mean()), "pickscore": float(np.nanmean(res["pick"]))}
        if name == "align":
            ok = res["clip"][:, None] > res["neg"]
            valid = ~np.isnan(res["neg"])
            s["neg_acc"] = {c: float(ok[valid[:, k], k].mean()) for k, c in enumerate(CATS)}
            s["neg_acc"]["all"] = float(ok[valid].mean())
        else:
            s["ocr"] = float(res["ocr"].mean())
            s["exact"] = float((res["ocr"] >= 1.0).mean())
        summary[name] = s
        if grid:
            ims = [Image.fromarray((g.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).resize((256, 256))
                   for g in grid]
            canvas = Image.new("RGB", (256 * len(ims), 256), "white")
            for k, im in enumerate(ims):
                canvas.paste(im, (256 * k, 0))
            canvas.save(os.path.join(args.out, f"grid_{name}.png"))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
