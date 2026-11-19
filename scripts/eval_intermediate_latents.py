"""Evaluate the intermediate latents of SD 1.5 sampling.

At every DDIM step i the UNet's epsilon prediction gives a one-shot estimate of the clean latent
    x0_hat_i = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)
We decode x0_hat_i with the VAE, score it with the same reward as training (multi_score, e.g. 80*ocr + clipscore),
and compare it with the reward of the final image. That tells you, per step:

  * how good the early preview already is (mean reward of x0_hat_i)
  * how well it predicts the final reward     (Pearson / Spearman over all samples)
  * how well it ranks samples *within* a prompt group, which is what GRPO advantages use
    (mean within-group Spearman, and how often the step's best sample is also the final best)
  * how much compute you would have spent by stopping there (fraction of UNet evaluations + wall time)

Usage (from the repo root, on a GPU node):
    python scripts/eval_intermediate_latents.py --num-prompts 32 --images-per-prompt 8
    python scripts/eval_intermediate_latents.py --lora logs/ocr/sd1.5/checkpoints/checkpoint-240/lora

Outputs go to --out (default logs/intermediate_eval/<timestamp>):
    rewards.npz     per-sample arrays [N, num_steps] for every reward key + final rewards
    summary.json    per-step metrics (also printed as a table)
    curves.png      mean reward and correlation vs step
    grids/*.png     per prompt: rows = samples, columns = x0_hat at each step, last column = final image
"""
import argparse
import datetime
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from scipy.stats import pearsonr, spearmanr

import flow_grpo.rewards


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--lora", default=None, help="Path to a saved LoRA dir (e.g. .../checkpoint-240/lora)")
    p.add_argument("--prompts", default="dataset/ocr/test.txt")
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--images-per-prompt", type=int, default=8, help="Group size (like GRPO num_image_per_prompt)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=10, help="Training uses 10, eval uses 40")
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--eta", type=float, default=1.0,
                   help="DDIM eta (inside the SDE window if --sde-window-size > 0, else at every step)")
    p.add_argument("--sde-window-size", type=int, default=0,
                   help="Like training: eta only inside a random window of this many steps, eta=0 elsewhere. "
                        "0 = eta at every step")
    p.add_argument("--sde-window-range", type=int, nargs=2, default=(0, 5),
                   help="Range the window start is drawn from (same semantics as config.sample.sde_window_range)")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--ocr-weight", type=float, default=80.0)
    p.add_argument("--clip-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid-prompts", type=int, default=8, help="How many prompts get an image grid (0 = none)")
    p.add_argument("--grid-thumb", type=int, default=192, help="Thumbnail size in the grids")
    p.add_argument("--score-every", type=int, default=1,
                   help="Score (OCR+CLIP) the preview only every k steps (+ the last); others are NaN. OCR on CPU "
                        "costs ~15 min per scored step for 1024 images, so use e.g. 5 for 50-step runs.")
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_prompts(path, n, seed):
    with open(path) as f:
        prompts = [line.strip() for line in f if line.strip()]
    random.Random(seed).shuffle(prompts)
    return prompts[:n]


@torch.no_grad()
def encode_prompts(pipe, prompts, device):
    ids = pipe.tokenizer(
        prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt"
    ).input_ids.to(device)
    return pipe.text_encoder(ids)[0]


@torch.no_grad()
def decode(pipe, latents):
    img = pipe.vae.decode((latents / pipe.vae.config.scaling_factor).to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type="pt")  # float in [0, 1], NCHW


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def sample_with_previews(pipe, prompt_embeds, neg_embeds, args, generator, window_rng):
    """DDIM sampling that returns the decoded x0_hat after every step plus the final image and timings."""
    device = pipe._execution_device
    bs = prompt_embeds.shape[0]
    do_cfg = args.guidance_scale > 1.0
    cond = torch.cat([neg_embeds, prompt_embeds]) if do_cfg else prompt_embeds

    latents = pipe.prepare_latents(
        bs, pipe.unet.config.in_channels, args.resolution, args.resolution,
        prompt_embeds.dtype, device, generator,
    ).float()
    pipe.scheduler.set_timesteps(args.num_steps, device=device)

    if args.sde_window_size > 0:  # same window rule as sd15_pipeline_with_logprob
        lo, hi = args.sde_window_range
        start = window_rng.randint(lo, hi - args.sde_window_size)
        window = (start, start + args.sde_window_size)
    else:
        window = (0, args.num_steps)

    previews, unet_times, decode_times = [], [], []
    for i, t in enumerate(pipe.scheduler.timesteps):
        eta = args.eta if window[0] <= i < window[1] else 0.0
        sync(); t0 = time.perf_counter()
        inp = torch.cat([latents] * 2) if do_cfg else latents
        inp = pipe.scheduler.scale_model_input(inp, t)
        eps = pipe.unet(inp.to(pipe.unet.dtype), t, encoder_hidden_states=cond, return_dict=False)[0].float()
        if do_cfg:
            eps_u, eps_c = eps.chunk(2)
            eps = eps_u + args.guidance_scale * (eps_c - eps_u)
        out = pipe.scheduler.step(eps, t, latents, eta=eta, generator=generator)
        latents = out.prev_sample.float()
        sync(); unet_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        previews.append(decode(pipe, out.pred_original_sample).cpu())
        sync(); decode_times.append(time.perf_counter() - t0)

    final = decode(pipe, latents).cpu()
    return previews, final, unet_times, decode_times


def score(reward_fn, images, prompts):
    meta = [{"target_text": p} for p in prompts]  # same metadata contract as train_sd15.py
    t0 = time.perf_counter()
    details, _ = reward_fn(images, prompts, meta)
    dt = time.perf_counter() - t0
    def to_np(v):
        if torch.is_tensor(v):
            return v.detach().float().cpu().numpy().astype(np.float64)
        return np.asarray([float(x) for x in v], dtype=np.float64)  # lists may hold 0-d tensors

    return {k: to_np(v) for k, v in details.items()}, dt


def safe_corr(fn, a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(fn(a, b)[0])


def step_metrics(step_r, final_r, group_ids):
    """step_r, final_r: [N] totals. group_ids: [N] prompt index."""
    m = {
        "pearson": safe_corr(pearsonr, step_r, final_r),
        "spearman": safe_corr(spearmanr, step_r, final_r),
    }
    within, best_hit, worst_half_overlap = [], [], []
    for g in np.unique(group_ids):
        idx = group_ids == g
        s, f = step_r[idx], final_r[idx]
        within.append(safe_corr(spearmanr, s, f))
        if np.std(f) > 1e-12:
            best_hit.append(float(np.argmax(s) == np.argmax(f)))
            # if you pruned the bottom half by the step score, how many of the final bottom half would you drop
            k = len(s) // 2
            if k > 0:
                worst_half_overlap.append(len(set(np.argsort(s)[:k]) & set(np.argsort(f)[:k])) / k)
    m["within_group_spearman"] = float(np.nanmean(within)) if np.any(~np.isnan(within)) else float("nan")
    m["best_of_group_hit"] = float(np.mean(best_hit)) if best_hit else float("nan")
    m["bottom_half_overlap"] = float(np.mean(worst_half_overlap)) if worst_half_overlap else float("nan")
    return m


def save_grid(path, previews, final, sample_idx, thumb):
    cols = len(previews) + 1
    grid = Image.new("RGB", (cols * thumb, len(sample_idx) * thumb), "white")
    for r, n in enumerate(sample_idx):
        for c, img in enumerate([p[n] for p in previews] + [final[n]]):
            arr = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            grid.paste(Image.fromarray(arr).resize((thumb, thumb)), (c * thumb, r * thumb))
    grid.save(path)


def plot_curves(path, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps = [s["step"] for s in summary["per_step"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(steps, [s["mean_ocr"] for s in summary["per_step"]], marker="o", label="OCR of x0_hat")
    a1.axhline(summary["final"]["mean_ocr"], ls="--", c="gray", label="OCR of final image")
    a1.set_xlabel("step"); a1.set_ylabel("mean OCR score"); a1.legend(); a1.grid(alpha=0.3)
    for key, lab in [("pearson", "Pearson (all)"), ("spearman", "Spearman (all)"),
                     ("within_group_spearman", "Spearman within group"), ("best_of_group_hit", "best-of-group hit")]:
        a2.plot(steps, [s[key] for s in summary["per_step"]], marker="o", label=lab)
    a2.set_xlabel("step"); a2.set_ylabel("agreement with final reward"); a2.set_ylim(-0.1, 1.05)
    a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    args = parse_args()
    out = args.out or os.path.join("logs", "intermediate_eval", datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    os.makedirs(os.path.join(out, "grids"), exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda")
    pipe = StableDiffusionPipeline.from_pretrained(args.model, safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if args.lora:
        from peft import PeftModel
        pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora).merge_and_unload()
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    reward_fn = flow_grpo.rewards.multi_score(device, {"ocr": args.ocr_weight, "clipscore": args.clip_weight})

    prompts = load_prompts(args.prompts, args.num_prompts, args.seed)
    # GRPO-style groups: each prompt repeated images_per_prompt times, different noise
    flat_prompts = [p for p in prompts for _ in range(args.images_per_prompt)]
    group_ids = np.repeat(np.arange(len(prompts)), args.images_per_prompt)
    neg = encode_prompts(pipe, [""], device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    window_rng = random.Random(args.seed)  # same seed -> same windows for every model (paired comparison)

    step_scores = None  # dict key -> list over steps of list of arrays
    scored = set(range(args.score_every - 1, args.num_steps, args.score_every)) | {args.num_steps - 1}
    final_scores = {}
    unet_t = np.zeros(args.num_steps)
    decode_t = np.zeros(args.num_steps)
    score_t = np.zeros(args.num_steps)
    final_score_t = 0.0
    n_batches = 0
    grid_done = set()

    for b0 in range(0, len(flat_prompts), args.batch_size):
        batch = flat_prompts[b0:b0 + args.batch_size]
        embeds = encode_prompts(pipe, batch, device)
        previews, final, ut, dt = sample_with_previews(pipe, embeds, neg.repeat(len(batch), 1, 1), args, generator,
                                                       window_rng)
        unet_t += ut; decode_t += dt; n_batches += 1

        per_step = []
        fs, t = score(reward_fn, final, batch)
        final_score_t += t
        for i, img in enumerate(previews):
            if i in scored:
                s, t = score(reward_fn, img, batch)
                score_t[i] += t
            else:
                s = {k: np.full(len(batch), np.nan) for k in fs}
            per_step.append(s)

        if step_scores is None:
            step_scores = {k: [[] for _ in range(args.num_steps)] for k in fs}
        for i, s in enumerate(per_step):
            for k, v in s.items():
                step_scores[k][i].append(v)
        for k, v in fs.items():
            final_scores.setdefault(k, []).append(v)

        # image grids: one per prompt (rows = that prompt's samples present in this batch)
        for g in sorted(set(group_ids[b0:b0 + len(batch)])):
            if g in grid_done or len(grid_done) >= args.grid_prompts:
                continue
            rows = [n for n in range(len(batch)) if group_ids[b0 + n] == g]
            save_grid(os.path.join(out, "grids", f"prompt_{g:03d}.png"), previews, final, rows, args.grid_thumb)
            grid_done.add(g)

        print(f"[{b0 + len(batch)}/{len(flat_prompts)}] final mean ocr={np.mean(fs['ocr']):.3f} "
              f"step-wise ocr={[round(float(np.mean(s['ocr'])), 2) for i, s in enumerate(per_step) if i in scored]}",
              flush=True)

    step_arr = {k: np.stack([np.concatenate(v) for v in vs], axis=1) for k, vs in step_scores.items()}  # [N, S]
    final_arr = {k: np.concatenate(v) for k, v in final_scores.items()}  # [N]
    np.savez(os.path.join(out, "rewards.npz"), group_ids=group_ids, prompts=np.array(flat_prompts),
             **{f"step_{k}": v for k, v in step_arr.items()}, **{f"final_{k}": v for k, v in final_arr.items()})

    timesteps = pipe.scheduler.timesteps.cpu().tolist()
    cum_unet = np.cumsum(unet_t) / n_batches
    summary = {
        "final": {"mean_ocr": float(final_arr["ocr"].mean()), "mean_clip": float(final_arr["clipscore"].mean()),
                  "mean_total": float(final_arr["avg"].mean()),
                  "exact_ocr_frac": float((final_arr["ocr"] >= 1.0).mean())},
        "timing_per_batch_s": {"unet_full_trajectory": float(cum_unet[-1]),
                               "vae_decode_per_step": float(decode_t.mean() / n_batches),
                               "reward_per_step": float(score_t[sorted(scored)].mean() / n_batches),
                               "reward_final": final_score_t / n_batches},
        "per_step": [],
    }
    for i in sorted(scored):
        m = step_metrics(step_arr["avg"][:, i], final_arr["avg"], group_ids)
        m.update(step=i + 1, timestep=int(timesteps[i]),
                 unet_frac=(i + 1) / args.num_steps,
                 unet_time_frac=float(cum_unet[i] / cum_unet[-1]),
                 mean_ocr=float(step_arr["ocr"][:, i].mean()),
                 mean_clip=float(step_arr["clipscore"][:, i].mean()),
                 mean_total=float(step_arr["avg"][:, i].mean()),
                 ocr_spearman=safe_corr(spearmanr, step_arr["ocr"][:, i], final_arr["ocr"]))
        summary["per_step"].append(m)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_curves(os.path.join(out, "curves.png"), summary)

    print(f"\nN={len(flat_prompts)} samples ({len(prompts)} prompts x {args.images_per_prompt}), "
          f"{args.num_steps} DDIM steps, eta={args.eta}, lora={args.lora}")
    print(f"final image: ocr={summary['final']['mean_ocr']:.3f} clip={summary['final']['mean_clip']:.3f} "
          f"total={summary['final']['mean_total']:.2f} exact={summary['final']['exact_ocr_frac']:.2f}")
    print(f"per batch: full UNet trajectory {cum_unet[-1]:.2f}s | VAE decode {summary['timing_per_batch_s']['vae_decode_per_step']:.2f}s/step "
          f"| reward {summary['timing_per_batch_s']['reward_per_step']:.2f}s/step")
    hdr = f"{'step':>4} {'t':>4} {'unet%':>6} {'ocr':>6} {'clip':>6} {'total':>7} {'pear':>6} {'spear':>6} {'grpSp':>6} {'best':>5} {'bot50':>6}"
    print(hdr); print("-" * len(hdr))
    for m in summary["per_step"]:
        print(f"{m['step']:>4} {m['timestep']:>4} {100 * m['unet_time_frac']:>5.0f}% {m['mean_ocr']:>6.3f} {m['mean_clip']:>6.3f} "
              f"{m['mean_total']:>7.2f} {m['pearson']:>6.2f} {m['spearman']:>6.2f} {m['within_group_spearman']:>6.2f} "
              f"{m['best_of_group_hit']:>5.2f} {m['bottom_half_overlap']:>6.2f}")
    print(f"\ngrpSp = mean Spearman with final reward inside each prompt group; best = step's best sample is the final best; "
          f"bot50 = overlap of bottom-half by step score vs final score (0.5 = chance).\nSaved to {out}")


if __name__ == "__main__":
    main()
