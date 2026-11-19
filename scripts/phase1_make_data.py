"""Phase 1 data: SD1.5 trajectories on GenAI-Bench training prompts for training a noise-aware scorer.

For each trajectory it stores, at every --save-every steps:
  proj_rgb  uint8 (S, 3, 64, 64)  Phi(x0_hat_t): the cheap linear RGB projection of the predicted clean latent
  x0_lat    fp16  (S, 4, 64, 64)  x0_hat_t itself (for latent-input models, StitchVM-like; --no-latents to skip)
and once per trajectory the distillation target:
  final_img_emb fp16 (768,)       CLIP ViT-L/14 image embedding of the final VAE-decoded image (normalised)
  text_emb      fp16 (768,)       CLIP text embedding of the prompt (normalised)

Prompts: the same shuffle as phase1_noisyclip.py (seed 0); the first --calib-prompts + --eval-prompts ids are held
out (Phi calibration + evaluation), everything else is training data. Phi is loaded from the evaluation run so the
scorer is trained and evaluated with the same projection.

Shard with --shard i --num-shards n (one SLURM job each). Output: <out>/shard_<i>.npz
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_noisyclip import Clip, decode, load_genai_bench, project, run_trajectories, sd_text  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--clip", default="openai/clip-vit-large-patch14")
    p.add_argument("--genai-bench", default="dataset/genai_bench")
    p.add_argument("--phi", default="logs/phase1_noisyclip/sd15_50steps/phi.npy")
    p.add_argument("--calib-prompts", type=int, default=32)
    p.add_argument("--eval-prompts", type=int, default=128)
    p.add_argument("--images-per-prompt", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument("--no-latents", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default="logs/phase1_data/sd15_50steps")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16, safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    clip = Clip(args.clip, device)
    phi = torch.from_numpy(np.load(args.phi)).to(device)

    prompts_by_id, _ = load_genai_bench(args.genai_bench)
    ids = sorted(prompts_by_id)
    random.Random(0).shuffle(ids)  # same order as phase1_noisyclip.py with its default --seed 0
    held_out = set(ids[:args.calib_prompts + args.eval_prompts])
    train_ids = [i for i in ids if i not in held_out]
    shard_ids = train_ids[args.shard::args.num_shards]
    flat = [pid for pid in shard_ids for _ in range(args.images_per_prompt)]
    save_steps = sorted(set(range(args.save_every - 1, args.num_steps, args.save_every)) | {args.num_steps - 1})
    S, N = len(save_steps), len(flat)
    print(f"shard {args.shard}/{args.num_shards}: {len(shard_ids)} prompts x {args.images_per_prompt} = {N} "
          f"trajectories, {S} saved steps each (held out: {len(held_out)} prompts)", flush=True)

    proj_rgb = np.zeros((N, S, 3, 64, 64), np.uint8)
    x0_lat = None if args.no_latents else np.zeros((N, S, 4, 64, 64), np.float16)
    final_img_emb = np.zeros((N, 768), np.float16)
    text_emb = np.zeros((N, 768), np.float16)

    neg = sd_text(pipe, [""], device)
    generator = torch.Generator(device=device).manual_seed(args.seed + args.shard)
    pos = {s: j for j, s in enumerate(save_steps)}

    for b0 in range(0, N, args.batch_size):
        bp = flat[b0:b0 + args.batch_size]
        B = len(bp)
        texts = [prompts_by_id[p] for p in bp]

        def cb(i, z_t, x0):
            if i in pos:
                j = pos[i]
                proj_rgb[b0:b0 + B, j] = (project(x0, phi) * 255).round().byte().cpu().numpy()
                if x0_lat is not None:
                    x0_lat[b0:b0 + B, j] = x0.half().cpu().numpy()

        lat, _ = run_trajectories(pipe, texts, neg, args, generator, cb)
        final_img_emb[b0:b0 + B] = clip.image(decode(pipe, lat)).half().cpu().numpy()
        text_emb[b0:b0 + B] = clip.text(texts).half().cpu().numpy()
        if (b0 // args.batch_size) % 10 == 0:
            sim = (final_img_emb[b0:b0 + B].astype(np.float32) * text_emb[b0:b0 + B].astype(np.float32)).sum(-1)
            print(f"[{b0 + B}/{N}] final CLIP score {sim.mean():.3f}", flush=True)

    pipe.scheduler.set_timesteps(args.num_steps)
    timesteps = pipe.scheduler.timesteps.cpu().numpy()[save_steps]
    data = dict(proj_rgb=proj_rgb, final_img_emb=final_img_emb, text_emb=text_emb, prompt_ids=np.array(flat),
                save_steps=np.array(save_steps), timesteps=timesteps, num_steps=args.num_steps)
    if x0_lat is not None:
        data["x0_lat"] = x0_lat
    path = os.path.join(args.out, f"shard_{args.shard}.npz")
    np.savez(path, **data)
    with open(os.path.join(args.out, f"shard_{args.shard}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
